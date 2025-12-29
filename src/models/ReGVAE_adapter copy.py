# src/models/ft_transformer_adapter.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.core.models.ReGVAE import ReGVAE
from src.core.utils.losses import ReGVAEViewKLLoss

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split


class ReVGAEAdapter(BaseModelAdapter):

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: ReGVAE | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        model_cfg = self.config.model
        self.lambda_class = float(getattr(model_cfg, "lambda_class", 1.0))
        self.lambda_view = float(getattr(model_cfg, "lambda_view", 1.0))
        self.lambda_recon = float(getattr(model_cfg, "lambda_recon", 1.0))
        self.lambda_kl = float(getattr(model_cfg, "lambda_kl", 1.0))

        self.view_kl_loss = ReGVAEViewKLLoss()

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim = train_data.meta.input_dim
        self.num_class = train_data.meta.num_class

        self.model = self._get_model(self.input_dim, self.num_class)

        optimizer, scheduler = self.get_deeplearning_utils()

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = int(self.config.train.early_stopping_rounds)
        num_epochs = int(self.config.train.epochs)

        train_total: list[float] = []
        valid_total: list[float] = []

        train_ce: list[float] = []
        valid_ce: list[float] = []

        train_view: list[float] = []
        valid_view: list[float] = []

        train_kl: list[float] = []
        valid_kl: list[float] = []

        train_recon: list[float] = []
        valid_recon: list[float] = []

        for epoch in range(num_epochs):
            tr = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer,
                split=Split.TRAIN,
            )
            vl = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
            )

            lr = float(optimizer.param_groups[0]["lr"])

            print(
                f"[{self.config.model.model.name} Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, ce={tr['ce']:.4f}, view={tr['view']:.4f}, kl={tr['kl']:.4f}, recon={tr['recon']:.4f} | "
                f"Valid: total={vl['total']:.4f}, ce={vl['ce']:.4f}, view={vl['view']:.4f}, kl={vl['kl']:.4f}, recon={vl['recon']:.4f} | "
                f"LR: {lr:.6f}"
            )

            train_total.append(tr["total"])
            valid_total.append(vl["total"])

            train_ce.append(tr["ce"])
            valid_ce.append(vl["ce"])

            train_view.append(tr["view"])
            valid_view.append(vl["view"])

            train_kl.append(tr["kl"])
            valid_kl.append(vl["kl"])

            train_recon.append(tr["recon"])
            valid_recon.append(vl["recon"])

            scheduler.step()

            if best_valid_loss is None or vl["total"] < best_valid_loss:
                best_valid_loss = vl["total"]
                patience = 0
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if patience >= max_patience:
                print(
                    f"[{self.config.model.model.name}] Early stopping at epoch {epoch + 1}"
                )
                break

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader, split=Split.TRAIN)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader, split=Split.VALID)

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        metric_name = "total_loss"
        tasks = [
            {
                Split.TRAIN.value: train_total,
                Split.VALID.value: valid_total,
            }
        ]

        components = {
            "stage1": {
                "train": {
                    "ce": train_ce,
                    "view": train_view,
                    "kl": train_kl,
                    "recon": train_recon,
                },
                "valid": {
                    "ce": valid_ce,
                    "view": valid_view,
                    "kl": valid_kl,
                    "recon": valid_recon,
                },
            }
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": metric_name,
                "tasks": tasks,
                "components": components,
            },
        }

        return results

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
    ):
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(self.config.model.model.name, split)

        total_sum = 0.0
        ce_sum = 0.0
        view_sum = 0.0
        kl_sum = 0.0
        recon_sum = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x_missing, y, x_clean, _, _, _ = self._prepare_batch(batch)

            with torch.set_grad_enabled(is_train):

                out_clean = self.model(x_cont=x_clean, x_cat=None)
                logits_clean = out_clean["logits"]
                recon_clean = out_clean["recon"]
                mu_clean = out_clean["z_mu"]
                logvar_clean = out_clean["z_logvar"]

                out_missing = self.model(x_cont=x_missing, x_cat=None)
                logits = out_missing["logits"]
                recon_missing = out_missing["recon"]
                mu_missing = out_missing["z_mu"]
                logvar_missing = out_missing["z_logvar"]

                # (1) 분류: 결측 입력 기준 (원하면 clean logits도 같이 CE 넣으셔도 됩니다)
                loss_ce = 0.5 * (
                    F.cross_entropy(logits, y) + F.cross_entropy(logits_clean, y)
                )

                # (2) view KL + prior KL
                d_kl = self.view_kl_loss(
                    mu_clean=mu_clean,
                    logvar_clean=logvar_clean,
                    mu_missing=mu_missing,
                    logvar_missing=logvar_missing,
                )
                loss_view = d_kl["view"]
                loss_kl = d_kl["kl"]

                # (3) recon: 결측/온전 모두 원본(x_clean)에 복원하도록

                loss_recon = 0.5 * (
                    F.mse_loss(recon_missing, x_clean)
                    + F.mse_loss(recon_clean, x_clean)
                )

                # loss_recon = F.mse_loss(recon_missing, x_clean)

                loss_total = (
                    self.lambda_class * loss_ce
                    + self.lambda_view * loss_view
                    + self.lambda_kl * loss_kl
                    + self.lambda_recon * loss_recon
                )

                if is_train:
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

            num_batches += 1
            total_sum += float(loss_total.item())
            ce_sum += float(loss_ce.item())
            view_sum += float(loss_view.item())
            kl_sum += float(loss_kl.item())
            recon_sum += float(loss_recon.item())

        denom = max(1, num_batches)
        return {
            "total": total_sum / denom,
            "ce": ce_sum / denom,
            "view": view_sum / denom,
            "kl": kl_sum / denom,
            "recon": recon_sum / denom,
        }

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, split=Split.TEST
        )

        metrics_overall = compute_classification_metrics(labels_all, preds_all)

        patterns = test_data.config.data.missing_patterns
        ratios = test_data.ratios

        metrics_by_ratio: dict[str, dict[float, dict]] = {}

        for p_i, pattern in enumerate(patterns):
            p_val = pattern.value
            metrics_by_ratio[p_val] = {}

            for r_i, ratio in enumerate(ratios):
                mask = (pattern_idx_all == p_i) & (ratio_idx_all == r_i)
                if np.any(mask):
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

    def predict(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
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

        desc = self.get_desc(self.config.model.model.name, split)

        for batch in tqdm(loader, desc=desc):
            x_missing, y, _, _, pattern_idx, ratio_idx = self._prepare_batch(batch)

            with torch.no_grad():
                out = self.model(x_cont=x_missing, x_cat=None)
                logits = out["logits"]

                if logits is None:
                    raise ValueError("d_out가 None인 모델입니다. logits가 None 입니다.")

                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=1)

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
        num_class: int | None,
    ) -> ReGVAE:
        if num_class is None:
            raise ValueError("num_class is None")

        # ReGVAE에 get_default_kwargs가 동일하게 구현돼 있다고 가정합니다.
        ft_kwargs = ReGVAE.get_default_kwargs(n_blocks=3)

        model = ReGVAE(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=num_class,
            latent_dim=None,
            logits_from="mu",
            **ft_kwargs,
        ).to(self.device)

        return model

    def save(
        self,
        path: Path,
    ):
        if self.model is None:
            raise ValueError(
                "저장할 모델이 없습니다. fit() 이후에 save()를 호출하십시오."
            )

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
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

        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])

        model_path = Path(meta["model_path"])

        self.model = self._get_model(self.input_dim, self.num_class)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        return True
