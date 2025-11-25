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
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split


class FTTransformerAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: FTTransformer | None = None
        self.device = self.config.train.device
        self.train_mode = self.config.model.stage



    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.feature_dim = train_data.meta.feature_dim
        self.num_class = train_data.meta.num_class
        self.horizon = train_data.get_horizon()
        self.sequence = train_data.meta.sequence

        self.model = self._get_model(self.feature_dim, self.num_class)

        optimizer, scheduler = self.get_deeplearning_utils()

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs: list[float] = []
        train_losses: list[float] = []
        valid_losses: list[float] = []


        for epoch in range(self.config.train.epochs):
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
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

            else:
                patience += 1

            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)

        # 최종 성능 계산
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

        train_metrics = compute_multitask_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_multitask_classification_metrics(vl_labels, vl_preds)

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

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(te_loader)

        # overall metric
        metrics_overall = compute_multitask_classification_metrics(labels_all, preds_all)

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

            x_flat = self._flatten_x(x)

            with torch.no_grad():
                logits = self.model(x_flat, None)  # (B, H * C)
                B = logits.size(0)
                H = self.horizon
                C = self.num_class

                logits = logits.view(B, H, C).permute(0, 2, 1)

                # y: (B, H)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=-1)  # (B, H)

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
            # y: (B, H)

            x_flat = self._flatten_x(x)

            logits = self.model(x_flat, None)  # (B, H * C)
            B = logits.size(0)
            H = self.horizon
            C = self.num_class

            logits = logits.view(B, H, C).permute(0, 2, 1)

            # y: (B, H)
            loss = F.cross_entropy(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

    def _flatten_x(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x_flat = x.view(B, -1)
        return x_flat


    def _get_model(
        self,
        input_dim: int,
        num_class: int,
    ):
        d_in = self.sequence * input_dim
        d_out = self.horizon * num_class

        ft_kwargs = FTTransformer.get_default_kwargs()
        model = FTTransformer(
            n_cont_features=d_in,
            cat_cardinalities=[],
            d_out=d_out,
            **ft_kwargs,
        ).to(self.device)

        return model



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
            "num_class": self.num_class,
            "horizon": self.horizon,
            "model_path": str(model_path),
            "sequence": self.sequence,
        }

        self.save_meta(save_dir, meta)

        return model_path

    def load(
        self,
        path: Path,
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.feature_dim = meta["feature_dim"]
        self.num_class = meta["num_class"]
        self.horizon = meta["horizon"]
        self.sequence = meta["sequence"]

        model_path = Path(meta["model_path"])

        self.model = self._get_model(self.feature_dim, self.num_class)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
