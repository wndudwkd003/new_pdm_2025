# src/models/ft_transformer_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from rtdl_revisiting_models import FTTransformer

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split


class FTTransformerAdapter(BaseModelAdapter):
    """
    rtdl_revisiting_models.FTTransformer 기반 테이블 단일 분류 어댑터.

    - 입력 x: (B, F) 또는 (B, ..., F) 형태의 연속형 feature 텐서
      → batch 차원만 남기고 나머지는 전부 flatten 해서 하나의 tabular feature로 사용
    - 출력 logits: (B, C)
      → C = num_class
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: FTTransformer | None = None
        self.device = self.config.train.device
        self.train_mode = self.config.model.stage

        self.feature_dim: int | None = None   # meta 기준 feature_dim
        self.input_dim: int | None = None     # 실제 입력 차원 (flatten 후)
        self.num_class: int | None = None

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        """
        단일 FT-Transformer로 테이블 분류를 수행.
        d_out = num_class 로 설정하여, 출력 (B, C)에 대해 cross_entropy 사용.
        """
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        # 메타에서 feature_dim / num_class 가져오기
        self.feature_dim = int(train_data.meta.feature_dim)
        self.num_class = int(train_data.meta.num_class)

        # 현재 테이블 구조에서는 inputs가 (N, F)이므로 입력 차원 = feature_dim
        self.input_dim = self.feature_dim

        # ★ 여기서 먼저 모델 생성 ★
        self.model = self._get_model(self.input_dim, self.num_class)

        # 그 다음에 optimizer / scheduler 생성
        optimizer, scheduler = self.get_deeplearning_utils()

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
                optimizer=optimizer,
                split=Split.TRAIN,
            )
            valid_loss = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
            )

            lr = float(optimizer.param_groups[0]['lr'])

            print(
                f"[{self.config.model.model.name} Epoch {epoch + 1}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"LR: {lr:.6f}"
            )

            lrs.append(lr)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            scheduler.step()

            # early stopping
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience = 0
                if self.model is not None:
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if patience >= max_patience:
                print(f"[{self.config.model.model.name}] Early stopping at epoch {epoch + 1}")
                break

        # best state 로드
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 최종 성능 계산
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        metric_name = "cross_entropy"

        # 단일 task 로 loss curve 기록
        tasks = [
            {
                Split.TRAIN.value: train_losses,
                Split.VALID.value: valid_losses,
            }
        ]

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
    # 단일 epoch 학습 / 검증
    # ------------------------------------------------------------------
    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
    ):
        is_train = (split == Split.TRAIN)

        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(self.config.model.model.name, split)

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, _, _ = self._prepare_batch(batch)
            # x: (B, F) 또는 (B, ..., F)
            # y: (B,)

            x_flat = self._flatten_x(x)  # (B, input_dim)

            logits = self.model(x_flat, None)  # (B, C)

            loss = F.cross_entropy(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_batches += 1
            total_loss += float(loss.item())

        return total_loss / max(1, num_batches)

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

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(te_loader)

        # overall metric
        metrics_overall = compute_classification_metrics(labels_all, preds_all)

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

                m = compute_classification_metrics(y_sub, y_hat_sub)
                metrics_by_ratio[p_val][ratio] = m

        results = {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

        return results

    # ------------------------------------------------------------------
    # 예측
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

        desc = self.get_desc(self.config.model.model.name, Split.TEST)

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, pattern_idx, ratio_idx = self._prepare_batch(batch)

            x_flat = self._flatten_x(x)  # (B, input_dim)

            with torch.no_grad():
                logits = self.model(x_flat, None)  # (B, C)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=1)       # (B,)

            total_loss += float(loss.item())
            num_batches += 1

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_pattern_idx.append(pattern_idx.cpu())
            all_ratio_idx.append(ratio_idx.cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    # ------------------------------------------------------------------
    # 배치 준비
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        batch: dict,
    ):
        x = batch["x"].to(self.device)                # (B, F) or (B, ..., F)
        y = batch["y"].to(self.device)                # (B,)
        x_ori = batch["x_originals"].to(self.device)  # 사용하지 않지만 유지
        bemv = batch["bemv"].to(self.device)          # 사용하지 않지만 유지
        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    def _flatten_x(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_flat = x.view(B, -1)
        return x_flat

    # ------------------------------------------------------------------
    # 모델 생성
    # ------------------------------------------------------------------
    def _get_model(
        self,
        input_dim: int,
        num_class: int | None,
    ) -> FTTransformer:
        if num_class is None:
            raise ValueError("num_class가 설정되지 않았습니다.")

        d_in = input_dim
        d_out = int(num_class)

        ft_kwargs = FTTransformer.get_default_kwargs()
        model = FTTransformer(
            n_cont_features=d_in,
            cat_cardinalities=[],
            d_out=d_out,
            **ft_kwargs,
        ).to(self.device)

        return model

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
            "feature_dim": self.feature_dim,
            "input_dim": self.input_dim,
            "num_class": self.num_class,
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

        self.feature_dim = int(meta["feature_dim"])
        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])

        model_path = Path(meta["model_path"])

        self.model = self._get_model(self.input_dim, self.num_class)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
