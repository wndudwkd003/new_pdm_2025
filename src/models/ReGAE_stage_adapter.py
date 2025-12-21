# src/models/ft_transformer_adapter.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.core.models.ReGAE import FTTransformer
from src.core.utils.losses import info_nce_loss

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split


class ReGAEAdapter(BaseModelAdapter):

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: FTTransformer | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        model_cfg = self.config.model
        self.lambda_class = float(getattr(model_cfg, "lambda_class", 1.0))
        self.lambda_view = float(getattr(model_cfg, "lambda_view", 1.0))
        self.lambda_recon = float(getattr(model_cfg, "lambda_recon", 1.0))

        self.temperature = float(getattr(self.config.train, "temperature", 0.1))

    def _apply_stage_train_hparams(self, stage: int) -> None:
        if stage == 1:
            self.config.train.lr = float(self.config.train.lr_stage_1)
            self.config.train.lr_min = float(self.config.train.lr_min_stage_1)
        elif stage == 2:
            self.config.train.lr = float(self.config.train.lr_stage_2)
            self.config.train.lr_min = float(self.config.train.lr_min_stage_2)
        else:
            raise ValueError(f"stage must be 1 or 2, got: {stage}")

    def _get_deeplearning_utils_for_stage(self, stage: int):
        self._apply_stage_train_hparams(stage)
        optimizer, scheduler = self.get_deeplearning_utils()
        return optimizer, scheduler

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

        max_patience = int(self.config.train.early_stopping_rounds)
        num_epochs = int(self.config.train.epochs)

        # -----------------------
        # Stage 1 history
        # -----------------------
        train_total_s1: list[float] = []
        valid_total_s1: list[float] = []

        train_ce_s1: list[float] = []
        valid_ce_s1: list[float] = []

        train_info_s1: list[float] = []
        valid_info_s1: list[float] = []

        train_recon_s1: list[float] = []
        valid_recon_s1: list[float] = []

        # -----------------------
        # Stage 2 history
        # -----------------------
        train_total_s2: list[float] = []
        valid_total_s2: list[float] = []

        train_ce_s2: list[float] = []
        valid_ce_s2: list[float] = []

        train_info_s2: list[float] = []
        valid_info_s2: list[float] = []

        train_recon_s2: list[float] = []
        valid_recon_s2: list[float] = []

        # =========================================================
        # Stage 1: Info + Recon
        # =========================================================
        optimizer_s1, scheduler_s1 = self._get_deeplearning_utils_for_stage(stage=1)

        best_valid_loss_s1 = None
        best_state_s1 = None
        patience_s1 = 0

        for epoch in range(num_epochs):
            tr = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer_s1,
                split=Split.TRAIN,
                stage=1,
            )
            vl = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
                stage=1,
            )

            lr = float(optimizer_s1.param_groups[0]["lr"])
            print(
                f"[{self.config.model.model.name} Stage 1 Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, info={tr['info']:.4f}, recon={tr['recon']:.4f} | "
                f"Valid: total={vl['total']:.4f}, info={vl['info']:.4f}, recon={vl['recon']:.4f} | "
                f"LR: {lr:.6f}"
            )

            train_total_s1.append(tr["total"])
            valid_total_s1.append(vl["total"])

            train_ce_s1.append(tr["ce"])
            valid_ce_s1.append(vl["ce"])

            train_info_s1.append(tr["info"])
            valid_info_s1.append(vl["info"])

            train_recon_s1.append(tr["recon"])
            valid_recon_s1.append(vl["recon"])

            scheduler_s1.step()

            if best_valid_loss_s1 is None or vl["total"] < best_valid_loss_s1:
                best_valid_loss_s1 = vl["total"]
                patience_s1 = 0
                best_state_s1 = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_s1 += 1

            if patience_s1 >= max_patience:
                print(
                    f"[{self.config.model.model.name}] Stage 1 early stopping at epoch {epoch + 1}"
                )
                break

        if best_state_s1 is not None and self.model is not None:
            self.model.load_state_dict(best_state_s1)
            self.model.to(self.device)

        # =========================================================
        # Stage 2: CE only (fine-tune on missing input)
        # =========================================================
        optimizer_s2, scheduler_s2 = self._get_deeplearning_utils_for_stage(stage=2)

        best_valid_loss_s2 = None
        best_state_s2 = None
        patience_s2 = 0

        for epoch in range(num_epochs):
            tr = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer_s2,
                split=Split.TRAIN,
                stage=2,
            )
            vl = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
                stage=2,
            )

            lr = float(optimizer_s2.param_groups[0]["lr"])
            print(
                f"[{self.config.model.model.name} Stage 2 Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, ce={tr['ce']:.4f} | "
                f"Valid: total={vl['total']:.4f}, ce={vl['ce']:.4f} | "
                f"LR: {lr:.6f}"
            )

            train_total_s2.append(tr["total"])
            valid_total_s2.append(vl["total"])

            train_ce_s2.append(tr["ce"])
            valid_ce_s2.append(vl["ce"])

            train_info_s2.append(tr["info"])
            valid_info_s2.append(vl["info"])

            train_recon_s2.append(tr["recon"])
            valid_recon_s2.append(vl["recon"])

            scheduler_s2.step()

            if best_valid_loss_s2 is None or vl["total"] < best_valid_loss_s2:
                best_valid_loss_s2 = vl["total"]
                patience_s2 = 0
                best_state_s2 = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_s2 += 1

            if patience_s2 >= max_patience:
                print(
                    f"[{self.config.model.model.name}] Stage 2 early stopping at epoch {epoch + 1}"
                )
                break

        if best_state_s2 is not None and self.model is not None:
            self.model.load_state_dict(best_state_s2)
            self.model.to(self.device)

        # =========================================================
        # Evaluate (after Stage 2)
        # =========================================================
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader, split=Split.TRAIN)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader, split=Split.VALID)

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        # 기존 schema 유지: total curve는 stage1 + stage2를 이어붙임
        train_total = train_total_s1 + train_total_s2
        valid_total = valid_total_s1 + valid_total_s2

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
                    "ce": train_ce_s1,
                    "info": train_info_s1,
                    "recon": train_recon_s1,
                },
                "valid": {
                    "ce": valid_ce_s1,
                    "info": valid_info_s1,
                    "recon": valid_recon_s1,
                },
            },
            "stage2": {
                "train": {
                    "ce": train_ce_s2,
                    "info": train_info_s2,
                    "recon": train_recon_s2,
                },
                "valid": {
                    "ce": valid_ce_s2,
                    "info": valid_info_s2,
                    "recon": valid_recon_s2,
                },
            },
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
        stage: int,
    ):
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")

        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got: {stage}")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(f"{self.config.model.model.name}_stage{stage}", split)

        total_sum = 0.0
        ce_sum = 0.0
        info_sum = 0.0
        recon_sum = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x_missing, y, x_clean, _, _, _ = self._prepare_batch(batch)

            with torch.set_grad_enabled(is_train):
                # 항상 missing/clean 둘 다 forward
                out_missing = self.model(x_cont=x_missing, x_cat=None)
                z_missing = out_missing["embedding"]
                recon_missing = out_missing.get("recon")

                out_clean = self.model(x_cont=x_clean, x_cat=None)
                z_clean = out_clean["embedding"]
                recon_clean = out_clean.get("recon")

                # 항상 InfoNCE / Recon 계산
                loss_info = info_nce_loss(z_clean, z_missing, self.temperature)
                loss_recon = 0.5 * (
                    F.mse_loss(recon_missing, x_clean)
                    + F.mse_loss(recon_clean, x_clean)
                )

                # stage2에서만 CE 추가
                if stage == 2:
                    logits = out_missing["logits"]
                    loss_ce = F.cross_entropy(logits, y)
                else:
                    loss_ce = torch.zeros((), device=x_missing.device)

                # total
                loss_total = (
                    self.lambda_view * loss_info + self.lambda_recon * loss_recon
                )
                if stage == 2:
                    loss_total = loss_total + self.lambda_class * loss_ce

                if is_train:
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

            num_batches += 1
            total_sum += float(loss_total.item())
            ce_sum += float(loss_ce.item())
            info_sum += float(loss_info.item())
            recon_sum += float(loss_recon.item())

        denom = max(1, num_batches)
        return {
            "total": total_sum / denom,
            "ce": ce_sum / denom,
            "info": info_sum / denom,
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
    ) -> FTTransformer:
        if num_class is None:
            raise ValueError("num_class is None")

        ft_kwargs = FTTransformer.get_default_kwargs(n_blocks=3)

        model = FTTransformer(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=num_class,
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
