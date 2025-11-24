# src/models/MDBE_1_adapter.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics
from src.core.utils.losses import info_nce_loss
from src.params.data_model import Split, StageType

from src.core.models.MDBE_2 import HybridDoubleBranchEncoder


class MDBE_2_Adapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: HybridDoubleBranchEncoder | None = None
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

        if not self.model:
            self.model = self._get_model(self.feature_dim, self.num_class)
            self.model.to(self.device)

        optimizer, scheduler = self.get_deeplearning_utils()

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs = []
        train_losses = []
        valid_losses = []

        for epoch in range(self.config.train.epochs):
            train_loss = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer,
                split=Split.TRAIN
            )
            valid_loss = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID
            )

            lr = float(optimizer.param_groups[0]['lr'])

            print(f"[{self.config.model.model.name} Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | LR: {lr:.6f}")

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

        if self.train_mode == StageType.FINETUNE:
            _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
            _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

            train_metrics = compute_multitask_classification_metrics(tr_labels, tr_preds)
            valid_metrics = compute_multitask_classification_metrics(vl_labels, vl_preds)
            metric_name = "cross_entropy"

        else:
            train_metrics = {}
            valid_metrics = {}
            metric_name = "info_nce_recon_loss"

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

        if self.train_mode != StageType.FINETUNE:
            raise ValueError("테스트는 파인튜닝 단계에서만 가능합니다.")


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
        if self.train_mode != StageType.FINETUNE:
            raise ValueError("predict는 FINETUNE 단계에서만 사용할 수 있습니다.")

        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        for batch in tqdm(loader, desc=f"[{self.config.model.model.name} PRED]"):
            x, y, _, bemv, pattern_idx, ratio_idx = self._prepare_batch(batch)

            with torch.no_grad():
                out = self.model(x, bemv)
                logits = out["logits"]          # (B, H, C)
                B, H, C = logits.shape

                logits_flat = logits.reshape(B * H, C)
                y_flat = y.reshape(B * H)

                loss = F.cross_entropy(logits_flat, y_flat)

                preds = logits.argmax(dim=-1)   # (B, H)

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



    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,

    ):
        is_train = (split == Split.TRAIN)
        self.model.train() if is_train else self.model.eval()


        desc = f"[{self.config.model.model.name} {split.name}]"

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, x_ori, bemv, _, _ = self._prepare_batch(batch)

            clean_bemv = torch.ones_like(bemv)

            with torch.set_grad_enabled(is_train):

                if self.train_mode == StageType.PRETRAIN:
                    out1 = self.model(x, bemv)
                    out2 = self.model(x_ori, clean_bemv)


                    z_masked = out1["latent"]
                    z_clean = out2["latent"]

                    info_loss = info_nce_loss(
                        z_clean,
                        z_masked,
                        self.config.train.temperature
                    )


                    recon1 = out1["recon"]
                    recon2 = out2["recon"]

                    recon_loss = (F.mse_loss(recon1, x_ori) +
                                  F.mse_loss(recon2, x_ori))

                    loss = recon_loss + info_loss

                elif self.train_mode == StageType.FINETUNE:
                    out = self.model(x, bemv)

                    logits = out["logits"]
                    B, H, C = logits.shape
                    logits_flat = logits.reshape(B * H, C)
                    y_flat = y.reshape(B * H)

                    loss = F.cross_entropy(logits_flat, y_flat)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            num_batches += 1
            total_loss += loss.item()

        return total_loss / num_batches



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


    def _get_model(
        self,
        input_dim: int,
        num_class: int,
    ):
        model = HybridDoubleBranchEncoder(
            input_dim=input_dim,
            embed_dim=self.config.params.embed_dim,
            feature_hidden_dims=self.config.params.feature_hidden_dims,
            num_class=num_class,
            nhead=self.config.params.nhead,
            transformer_layers=self.config.params.transformer_layers,
            decoder_hidden_dim=self.config.params.decoder_hidden_dim,
            total_layer=self.config.params.total_layer,
            horizon=self.horizon,
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
            "model_path": str(model_path),
            "horizon": self.horizon,
        }

        self.save_meta(save_dir, meta)

        return model_path



    def load(
        self,
        path: Path,
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)
        feature_dim = int(meta["feature_dim"])
        num_class = int(meta["num_class"])
        model_path = Path(meta["model_path"])
        self.horizon = int(meta["horizon"])
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.device = self.config.train.device
        self.model = self._get_model(feature_dim, num_class)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        return True
