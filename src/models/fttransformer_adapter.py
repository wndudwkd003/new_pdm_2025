# src/models/ft_transformer_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from rtdl_revisiting_models import FTTransformer  # pip install rtdl_revisiting_models

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split


class FTTransformerAdapter(BaseModelAdapter):
    """
    FT-Transformer (rtdl_revisiting_models) 기반 멀티-스텝 분류 어댑터.

    - 입력 x: (B, S, F) 혹은 (B, ...) 모양의 연속형 feature 텐서
      → batch 차원만 남기고 나머지는 전부 flatten 해서 하나의 tabular feature로 사용
    - 출력 logits: (B, H, C)
      → FT-Transformer의 d_out = H * C 로 설정 후 reshape
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: FTTransformer | None = None
        self.device = self.config.train.device

        self.horizon: int | None = None
        self.num_class: int | None = None
        self.n_cont_features: int | None = None

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None  # 타입을 특정하지 않음

    # ------------------------------------------------------------------
    # 핵심 학습 루프
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.horizon = int(train_data.get_horizon())
        self.num_class = int(train_data.meta.num_class)

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs: list[float] = []
        train_losses: list[float] = []
        valid_losses: list[float] = []

        num_epochs = self.config.train.epochs

        for epoch in range(num_epochs):
            train_loss = self.run_epoch(
                loader=tr_loader,
                split=Split.TRAIN,
            )
            valid_loss = self.run_epoch(
                loader=vl_loader,
                split=Split.VALID,
            )

            if self.optimizer is None:
                lr = float(self.config.train.learning_rate)
            else:
                lr = float(self.optimizer.param_groups[0]["lr"])

            print(
                f"[{self.config.model.model.name} Epoch {epoch + 1}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"LR: {lr:.6f}"
            )

            lrs.append(lr)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            # early stopping
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience = 0
                if self.model is not None:
                    best_state = {
                        k: v.cpu()
                        for k, v in self.model.state_dict().items()
                    }
            else:
                patience += 1

            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 최종 성능 계산
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

        train_metrics = compute_multitask_classification_metrics(
            tr_labels, tr_preds
        )
        valid_metrics = compute_multitask_classification_metrics(
            vl_labels, vl_preds
        )

        metric_name = "cross_entropy"

        H = self.horizon
        tasks = []
        for _ in range(H):
            tasks.append(
                {
                    Split.TRAIN.value: train_losses,
                    Split.VALID.value: valid_losses,
                }
            )

        loss_info = {
            "metric_name": metric_name,
            "tasks": tasks,
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": loss_info,
        }

        return results

    # ------------------------------------------------------------------
    # 테스트
    # ------------------------------------------------------------------
    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader
        )

        # overall metric
        metrics_overall = compute_multitask_classification_metrics(
            labels_all, preds_all
        )

        # pattern/ratio 별 metric
        patterns = test_data.config.data.missing_patterns
        ratios = test_data.ratios

        metrics_by_ratio: dict[str, dict[float, dict]] = {}

        for p_i, pattern in enumerate(patterns):
            p_val = pattern.value
            metrics_by_ratio[p_val] = {}

            for r_i, ratio in enumerate(ratios):
                mask = (pattern_idx_all == p_i) & (ratio_idx_all == r_i)
                if not np.any(mask):
                    continue

                y_sub = labels_all[mask]
                y_hat_sub = preds_all[mask]

                m = compute_multitask_classification_metrics(y_sub, y_hat_sub)
                metrics_by_ratio[p_val][ratio] = m

        results = {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

        return results

    # ------------------------------------------------------------------
    # 예측 공통 함수
    # ------------------------------------------------------------------
    def predict(
        self,
        loader: DataLoader,
    ):
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        desc = f"[{self.config.model.model.name} PRED]"

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, pattern_idx, ratio_idx = self._prepare_batch(batch)

            x_flat = self._flatten_x(x)

            with torch.no_grad():
                logits = self.model(x_flat, None)  # (B, H * C)
                B = logits.size(0)
                H = self.horizon
                C = self.num_class

                logits = logits.view(B, H, C)
                logits_flat = logits.view(B * H, C)
                y_flat = y.view(B * H)

                loss = F.cross_entropy(logits_flat, y_flat)

                preds = logits.argmax(dim=-1)  # (B, H)

            total_loss += float(loss.item())
            num_batches += 1

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_pattern_idx.append(pattern_idx.cpu())
            all_ratio_idx.append(ratio_idx.cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()          # (N, H)
        labels_all = torch.cat(all_labels, dim=0).numpy()        # (N, H)
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()  # (N,)
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()      # (N,)

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    # ------------------------------------------------------------------
    # 에폭 단위 학습
    # ------------------------------------------------------------------
    def run_epoch(
        self,
        loader: DataLoader,
        split: Split,
    ):
        is_train = (split == Split.TRAIN)

        if is_train:
            self.model_train_mode()
        else:
            self.model_eval_mode()

        desc = f"[{self.config.model.model.name} {split.name}]"

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, _, _ = self._prepare_batch(batch)

            # 첫 배치에서 모델이 없으면 여기서 생성
            if self.model is None:
                self._build_model_from_batch(x)

            x_flat = self._flatten_x(x)

            if is_train and self.optimizer is None:
                self.optimizer, self.scheduler = self.get_deeplearning_utils()

            if is_train and self.optimizer is None:
                raise ValueError("optimizer가 설정되지 않았습니다.")

            if is_train:
                self.optimizer.zero_grad()

            logits = self.model(x_flat, None)  # (B, H * C)
            B = logits.size(0)
            H = self.horizon
            C = self.num_class

            logits = logits.view(B, H, C)
            logits_flat = logits.view(B * H, C)
            y_flat = y.view(B * H)

            loss = F.cross_entropy(logits_flat, y_flat)

            if is_train:
                loss.backward()
                self.optimizer.step()

            num_batches += 1
            total_loss += float(loss.item())

        return total_loss / max(1, num_batches)

    # ------------------------------------------------------------------
    # 배치 준비 / 유틸
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        batch: dict,
    ):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        x_ori = batch["x_originals"].to(self.device)
        bemv = batch["bemv"].to(self.device)
        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    def _flatten_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, ...) → (B, n_cont_features) 로 flatten
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        return x_flat

    def _build_model_from_batch(
        self,
        x: torch.Tensor,
    ):
        """
        첫 배치의 x 모양을 보고 FT-Transformer를 구성.
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        n_cont_features = x_flat.size(1)

        self.n_cont_features = int(n_cont_features)

        if self.horizon is None or self.num_class is None:
            raise ValueError("horizon 또는 num_class가 설정되지 않았습니다.")

        d_out = int(self.horizon * self.num_class)

        ft_kwargs = FTTransformer.get_default_kwargs()
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=[],
            d_out=d_out,
            **ft_kwargs,
        ).to(self.device)

    def model_train_mode(self):
        if self.model is not None:
            self.model.train()

    def model_eval_mode(self):
        if self.model is not None:
            self.model.eval()

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------
    def save(
        self,
        path: Path,
    ):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. fit() 이후에 save()를 호출하십시오.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "n_cont_features": self.n_cont_features,
            "num_class": self.num_class,
            "horizon": self.horizon,
            "model_path": str(model_path),
        }

        self.save_meta(save_dir, meta)

        return model_path

    def load(
        self,
        path: Path,
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.n_cont_features = int(meta["n_cont_features"])
        self.num_class = int(meta["num_class"])
        self.horizon = int(meta["horizon"])

        model_path = Path(meta["model_path"])

        ft_kwargs = FTTransformer.get_default_kwargs()
        d_out = int(self.horizon * self.num_class)

        self.model = FTTransformer(
            n_cont_features=self.n_cont_features,
            cat_cardinalities=[],
            d_out=d_out,
            **ft_kwargs,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
