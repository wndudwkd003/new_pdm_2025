# src/models/tab_transformer_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split, StageType
from tab_transformer_pytorch import TabTransformer


class TabTransformerAdapter(BaseModelAdapter):
    """
    TabTransformer 기반 멀티-스텝 분류 어댑터.

    - 입력 x: (B, S, F) 혹은 (B, ...) 모양의 연속형 feature 텐서
      → batch 차원만 남기고 나머지는 전부 flatten 해서 하나의 tabular feature로 사용
    - 출력 logits: (B, H, C)
      → TabTransformer의 dim_out = H * C 로 설정 후 reshape
    """

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: TabTransformer | None = None

        self.device = self.config.train.device
        self.train_mode = self.config.model.stage

        self.feature_dim: int | None = None   # F (센서 수)
        self.seq_len: int | None = None       # S (윈도우 길이)
        self.input_dim: int | None = None     # S * F (TabTransformer 연속형 입력 차원)
        self.num_class: int | None = None
        self.horizon: int | None = None

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        """
        단일 TabTransformer로 horizon 전체(H-step)를 한 번에 예측.
        dim_out = H * C 로 설정하여, 출력 (B, H*C)을 (B, H, C)로 reshape.
        """
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        # horizon, num_class 설정
        self.num_class = int(train_data.meta.num_class)
        self.horizon = int(train_data.get_horizon())

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
                print(f"[{self.config.model.model.name}] Early stopping at epoch {epoch + 1}")
                break

        # best state 로드
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 최종 성능 계산 (train / valid)
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
    # 단일 epoch 학습 / 검증
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

            # 첫 배치에서 모델이 없으면 여기서 생성 (입력 차원 추론)
            if self.model is None:
                self._build_model_from_batch(x)

            # x: (B, S, F) → (B, S*F)
            B = x.size(0)
            x_flat = x.view(B, -1)

            # TabTransformer 입력: (x_categ, x_cont)
            x_categ = torch.empty(B, 0, dtype=torch.long, device=self.device)
            x_cont = x_flat

            if is_train and self.optimizer is None:
                self.optimizer, self.scheduler = self.get_deeplearning_utils()

            if is_train and self.optimizer is None:
                raise ValueError("optimizer가 설정되지 않았습니다.")

            if is_train:
                self.optimizer.zero_grad()

            logits = self.model(x_categ, x_cont)  # (B, H * C)

            H = self.horizon
            C = self.num_class
            logits = logits.view(B, H, C)         # (B, H, C)

            logits_flat = logits.view(B * H, C)   # (B*H, C)
            y_flat = y.view(B * H)                # (B*H,)

            loss = F.cross_entropy(logits_flat, y_flat)

            if is_train:
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

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
    # 예측 (전체 horizon 동시에)
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

            B = x.size(0)
            x_flat = x.view(B, -1)

            x_categ = torch.empty(B, 0, dtype=torch.long, device=self.device)
            x_cont = x_flat

            with torch.no_grad():
                logits = self.model(x_categ, x_cont)  # (B, H * C)

                H = self.horizon
                C = self.num_class

                logits = logits.view(B, H, C)        # (B, H, C)
                logits_flat = logits.view(B * H, C)  # (B*H, C)
                y_flat = y.view(B * H)               # (B*H,)

                loss = F.cross_entropy(logits_flat, y_flat)

                preds = logits.argmax(dim=-1)        # (B, H)

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
    # 배치 준비
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        batch: dict,
    ):
        x = batch["x"].to(self.device)                # (B, S, F)
        y = batch["y"].to(self.device)                # (B, H)
        x_ori = batch["x_originals"].to(self.device)  # 사용하지 않지만 유지
        bemv = batch["bemv"].to(self.device)          # 사용하지 않지만 유지
        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    # ------------------------------------------------------------------
    # TabTransformer 생성 (첫 배치에서 입력 차원 추론)
    # ------------------------------------------------------------------
    def _build_model_from_batch(
        self,
        x: torch.Tensor,
    ):
        """
        첫 배치의 x 모양을 보고 TabTransformer를 구성.
        x: (B, S, F) 또는 (B, ...) 로 가정.
        """
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, S*F 또는 기타)
        input_dim = x_flat.size(1)

        self.input_dim = int(input_dim)

        # S, F를 명시적으로 알고 싶다면, 3D인 경우에만 설정
        if x.dim() == 3:
            _, S, F_feat = x.shape
            self.seq_len = int(S)
            self.feature_dim = int(F_feat)
        else:
            self.seq_len = None
            self.feature_dim = None

        if self.horizon is None or self.num_class is None:
            raise ValueError("horizon 또는 num_class가 설정되지 않았습니다.")

        # 멀티-스텝: dim_out = H * C
        dim_out = int(self.horizon * self.num_class)

        self.model = self._get_model(
            input_dim=self.input_dim,
            dim_out=dim_out,
        ).to(self.device)

    def _get_model(
        self,
        input_dim: int,
        dim_out: int,
    ) -> TabTransformer:
        """
        num_continuous = input_dim (= S*F)
        dim_out = H * C
        """
        model = TabTransformer(
            categories=(),                # 카테고리 없음
            num_continuous=input_dim,     # flatten된 연속형 feature 수 (S*F)
            dim=32,
            dim_out=dim_out,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
        )
        return model

    # ------------------------------------------------------------------
    # 모드 전환
    # ------------------------------------------------------------------
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
            "feature_dim": self.feature_dim,
            "seq_len": self.seq_len,
            "input_dim": self.input_dim,
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

        self.feature_dim = int(meta["feature_dim"]) if meta["feature_dim"] is not None else None
        self.seq_len = int(meta["seq_len"]) if meta["seq_len"] is not None else None
        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])
        self.horizon = int(meta["horizon"])

        model_path = Path(meta["model_path"])

        dim_out = int(self.horizon * self.num_class)

        self.model = self._get_model(
            input_dim=self.input_dim,
            dim_out=dim_out,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
