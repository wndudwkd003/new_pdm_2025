# src/models/patchtst_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from transformers import PatchTSTConfig, PatchTSTModel

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split


class PatchTSTBackboneHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        out_dim: int,
    ):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=input_dim,
            context_length=seq_len,
        )

        self.backbone = PatchTSTModel(config)
        self.d_model = config.d_model
        self.head = nn.Linear(self.d_model, out_dim)

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        outputs = self.backbone(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
        )
        hidden = outputs.last_hidden_state

        if hidden.dim() != 4:
            raise ValueError(f"expected 4D hidden, got {hidden.shape}")

        pooled = hidden.mean(dim=(1, 2))  # (B, d_model)
        logits_flat = self.head(pooled)   # (B, out_dim)
        return logits_flat


class PatchTSTAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: PatchTSTBackboneHead | None = None
        self.device = self.config.train.device

        self.horizon: int | None = None
        self.num_class: int | None = None
        self.input_dim: int | None = None
        self.seq_len: int | None = None
        self.out_dim: int | None = None

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.horizon = train_data.get_horizon()
        self.num_class = train_data.meta.num_class
        self.out_dim = self.horizon * self.num_class

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
                lr = self.config.train.learning_rate
            else:
                lr = self.optimizer.param_groups[0]["lr"]

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

        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

        train_metrics = compute_multitask_classification_metrics(
            tr_labels, tr_preds
        )
        valid_metrics = compute_multitask_classification_metrics(
            vl_labels, vl_preds
        )

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
            "metric_name": "cross_entropy",
            "tasks": tasks,
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": loss_info,
        }

        return results

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

        metrics_overall = compute_multitask_classification_metrics(
            labels_all, preds_all
        )

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
            x, y, _, bemv, pattern_idx, ratio_idx = self._prepare_batch(batch)

            past_values = self._to_patch_input(x)
            past_mask = self._to_patch_mask(bemv)

            with torch.no_grad():
                logits_flat = self.model(
                    past_values=past_values,
                    past_observed_mask=past_mask,
                )

                B = logits_flat.size(0)
                H = self.horizon
                C = self.num_class

                if logits_flat.size(1) != H * C:
                    raise ValueError(
                        f"PatchTSTAdapter.predict: logits_flat dim {logits_flat.size(1)} "
                        f"!= H*C {H * C} (H={H}, C={C})"
                    )

                logits = logits_flat.view(B, H, C)

                logits_flat_ce = logits.view(B * H, C)
                y_flat = y.view(B * H)

                loss = F.cross_entropy(logits_flat_ce, y_flat)
                preds = logits.argmax(dim=-1)

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

    def run_epoch(
        self,
        loader: DataLoader,
        split: Split,
    ):
        is_train = split == Split.TRAIN

        if is_train:
            self.model_train_mode()
        else:
            self.model_eval_mode()

        desc = f"[{self.config.model.model.name} {split.name}]"

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, _, bemv, _, _ = self._prepare_batch(batch)

            past_values = self._to_patch_input(x)
            past_mask = self._to_patch_mask(bemv)

            self._build_model_from_batch(past_values)

            self.optimizer, self.scheduler = self.get_deeplearning_utils()

            if is_train:
                self.optimizer.zero_grad()

            logits_flat = self.model(
                past_values=past_values,
                past_observed_mask=past_mask,
            )

            B = logits_flat.size(0)
            H = self.horizon
            C = self.num_class

            if logits_flat.size(1) != H * C:
                raise ValueError(
                    f"PatchTSTAdapter.run_epoch: logits_flat dim {logits_flat.size(1)} "
                    f"!= H*C {H * C} (H={H}, C={C})"
                )

            logits = logits_flat.view(B, H, C)

            logits_flat_ce = logits.view(B * H, C)
            y_flat = y.view(B * H)

            loss = F.cross_entropy(logits_flat_ce, y_flat)

            if is_train:
                loss.backward()
                self.optimizer.step()

            num_batches += 1
            total_loss += float(loss.item())

        return total_loss / max(1, num_batches)

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

    def _to_patch_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() == 3:
            return x

        if x.dim() == 4:
            B, S, F, D = x.shape
            return x.view(B, S, F * D)

        raise ValueError(f"지원하지 않는 x 차원: {x.shape}")

    def _to_patch_mask(
        self,
        bemv: torch.Tensor,
    ) -> torch.Tensor:
        if bemv.dim() == 3:
            return bemv > 0

        if bemv.dim() == 4:
            B, S, F, D = bemv.shape
            if D == 1:
                return bemv[..., 0] > 0
            return (bemv > 0).all(dim=-1)

        raise ValueError(f"지원하지 않는 bemv 차원: {bemv.shape}")

    def _build_model_from_batch(
        self,
        past_values: torch.Tensor,
    ):
        _, S, F = past_values.shape

        self.seq_len = S
        self.input_dim = F

        if self.horizon is None or self.num_class is None:
            raise ValueError("horizon 또는 num_class가 설정되지 않았습니다.")

        self.out_dim = self.horizon * self.num_class

        self.model = PatchTSTBackboneHead(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            out_dim=self.out_dim,
        ).to(self.device)

        if self.model.head.out_features != self.out_dim:
            raise RuntimeError(
                f"PatchTSTBackboneHead head.out_features={self.model.head.out_features} "
                f"!= out_dim={self.out_dim}"
            )

    def model_train_mode(self):
        if self.model is not None:
            self.model.train()

    def model_eval_mode(self):
        if self.model is not None:
            self.model.eval()

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
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "num_class": self.num_class,
            "horizon": self.horizon,
            "out_dim": self.out_dim,
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

        self.input_dim = meta["input_dim"]
        self.seq_len = meta["seq_len"]
        self.num_class = meta["num_class"]
        self.horizon = meta["horizon"]
        self.out_dim = meta["out_dim"]

        model_path = Path(meta["model_path"])

        self.model = PatchTSTBackboneHead(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            out_dim=self.out_dim,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
