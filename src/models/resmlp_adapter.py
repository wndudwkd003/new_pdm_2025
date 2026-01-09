# src/models/resmlp_adapter.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rtdl_revisiting_models import ResNet

from src.configs.configs import Config
from src.datasets.data_class import Datasets
from src.models.base_model_adapter import BaseModelAdapter
from src.params.data_model import Split
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics


class ResMLPAdapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)

        self.model: ResNet | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.output_dim: int | None = None  # classification: num_class, regression: 1
        self.is_regression: bool = False

    # -------------------------
    # task inference
    # -------------------------
    def _is_regression_from_meta(self, data: Datasets) -> bool:
        meta = data.meta

        if isinstance(meta, dict):
            if "task" not in meta:
                return False
            t = str(meta["task"]).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False
            raise ValueError(f"Unknown meta.task: {meta['task']}")

        if hasattr(meta, "task"):
            t = str(meta.task).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False
            raise ValueError(f"Unknown meta.task: {meta.task}")

        return False

    def _infer_num_class_from_dataset(self, data: Datasets) -> int:
        y = data.imputed_dict["original"]["y"]
        y = np.asarray(y)

        uniq = np.unique(y)
        if uniq.ndim != 1:
            raise ValueError("y unique must be 1D")

        if uniq.size < 2:
            raise ValueError(f"num_class must be >=2, got uniq={uniq}")

        if int(uniq.min()) != 0:
            raise ValueError(
                f"labels must start at 0, got min={uniq.min()} uniq={uniq}"
            )
        if int(uniq.max()) != int(uniq.size - 1):
            raise ValueError(
                f"labels must be contiguous 0..C-1, got max={uniq.max()} size={uniq.size} uniq={uniq}"
            )

        return int(uniq.size)

    def _infer_input_dim_output_dim(
        self, train_data: Datasets, valid_data: Datasets
    ) -> tuple[int, int, bool]:
        input_dim = int(train_data.meta.input_dim)

        is_reg = self._is_regression_from_meta(train_data)
        is_reg_v = self._is_regression_from_meta(valid_data)
        if is_reg != is_reg_v:
            raise ValueError(
                f"train/valid task mismatch: train={is_reg} valid={is_reg_v}"
            )

        if is_reg:
            return input_dim, 1, True

        n_tr = self._infer_num_class_from_dataset(train_data)
        n_vl = self._infer_num_class_from_dataset(valid_data)
        if n_tr != n_vl:
            raise ValueError(
                f"train/valid num_class mismatch: train={n_tr}, valid={n_vl}"
            )

        if hasattr(train_data.meta, "num_class"):
            n_meta = int(train_data.meta.num_class)
            if n_meta > 0 and n_meta != n_tr:
                raise ValueError(
                    f"meta.num_class != y-derived: meta={n_meta}, y={n_tr}"
                )

        return input_dim, n_tr, False

    # -------------------------
    # losses / outputs
    # -------------------------
    def _pred_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            p = pred.squeeze(-1)
            yt = y.float()
            return F.mse_loss(p, yt)
        return F.cross_entropy(pred, y.long())

    def _pred_to_output(self, pred: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            return pred.squeeze(-1)
        return pred.argmax(dim=1)

    # -------------------------
    # fit / test
    # -------------------------
    def fit(self, train_data: Datasets, valid_data: Datasets):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim, self.output_dim, self.is_regression = (
            self._infer_input_dim_output_dim(train_data, valid_data)
        )

        self.model = self._get_model(self.input_dim, self.output_dim)
        optimizer, scheduler = self.get_deeplearning_utils()

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs: list[float] = []
        train_losses: list[float] = []
        valid_losses: list[float] = []

        num_epochs = self.config.train.epochs
        name = self.config.model.model.name

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

            lr = float(optimizer.param_groups[0]["lr"])
            msg = self.get_epoch_message(name, epoch, train_loss, valid_loss, lr)
            print(msg)

            lrs.append(lr)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            scheduler.step()

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience = 0
                if self.model is None:
                    raise ValueError("model is None")
                best_state = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"[{name}] Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            if self.model is None:
                raise ValueError("model is None")
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader, split=Split.TRAIN)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader, split=Split.VALID)

        if self.is_regression:
            train_metrics = compute_regression_metrics(tr_labels, tr_preds)
            valid_metrics = compute_regression_metrics(vl_labels, vl_preds)
            task_name = "regression"
            metric_name = "mse"
        else:
            train_metrics = compute_classification_metrics(tr_labels, tr_preds)
            valid_metrics = compute_classification_metrics(vl_labels, vl_preds)
            task_name = "classification"
            metric_name = "pred_loss"

        tasks = [{Split.TRAIN.value: train_losses, Split.VALID.value: valid_losses}]
        loss_info = {"metric_name": metric_name, "tasks": tasks, "lrs": lrs}

        results = {
            "split": Split.TRAIN.value,
            "task": task_name,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": loss_info,
        }
        return results

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
    ) -> float:
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(self.config.model.model.name, split)

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, _, _ = self._prepare_batch(batch)

            x_flat = self._flatten_x(x)
            pred = self.model(x_flat)
            loss = self._pred_loss(pred, y)

            if is_train:
                if optimizer is None:
                    raise ValueError("optimizer is None in train mode")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_batches += 1
            total_loss += float(loss.item())

        return total_loss / max(1, num_batches)

    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("model is None")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, split=Split.TEST
        )

        if self.is_regression:
            metrics_overall = compute_regression_metrics(labels_all, preds_all)
        else:
            metrics_overall = compute_classification_metrics(labels_all, preds_all)

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

                if self.is_regression:
                    m = compute_regression_metrics(y_sub, y_hat_sub)
                else:
                    m = compute_classification_metrics(y_sub, y_hat_sub)

                metrics_by_ratio[p_val][ratio] = m

        results = {
            "split": Split.TEST.value,
            "task": "regression" if self.is_regression else "classification",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }
        return results

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
    ):
        if self.model is None:
            raise ValueError("model is None")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_pattern_idx: list[torch.Tensor] = []
        all_ratio_idx: list[torch.Tensor] = []

        desc = self.get_desc(self.config.model.model.name, split)

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, pattern_idx, ratio_idx = self._prepare_batch(batch)
            x_flat = self._flatten_x(x)

            pred = self.model(x_flat)
            loss = self._pred_loss(pred, y)
            out_pred = self._pred_to_output(pred)

            total_loss += float(loss.item())
            num_batches += 1

            all_preds.append(out_pred.detach().cpu())
            all_labels.append(y.detach().cpu())

            if not torch.is_tensor(pattern_idx):
                raise ValueError("pattern_idx must be torch.Tensor")
            if not torch.is_tensor(ratio_idx):
                raise ValueError("ratio_idx must be torch.Tensor")

            all_pattern_idx.append(pattern_idx.detach().cpu())
            all_ratio_idx.append(ratio_idx.detach().cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    # -------------------------
    # batch / shape
    # -------------------------
    def _prepare_batch(self, batch: dict):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)

        x_ori = batch["x_originals"].to(self.device)
        bemv = batch["bemv"].to(self.device)

        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    def _flatten_x(self, x: torch.Tensor) -> torch.Tensor:
        B = int(x.size(0))
        x_flat = x.view(B, -1)
        if self.input_dim is not None:
            if int(x_flat.size(1)) != int(self.input_dim):
                raise ValueError(
                    f"input_dim mismatch: meta={self.input_dim} but batch has {int(x_flat.size(1))}"
                )
        return x_flat

    # -------------------------
    # model
    # -------------------------
    def _get_model(self, input_dim: int, output_dim: int | None) -> ResNet:
        if output_dim is None:
            raise ValueError("output_dim is None")

        n_blocks = 4
        d_block = 128
        d_hidden = None
        d_hidden_multiplier = 2.0
        dropout1 = 0.1  # hidden dropout
        dropout2 = 0.0  # residual dropout

        model = ResNet(
            d_in=int(input_dim),
            d_out=int(output_dim),
            n_blocks=int(n_blocks),
            d_block=int(d_block),
            d_hidden=d_hidden,
            d_hidden_multiplier=float(d_hidden_multiplier),
            dropout1=float(dropout1),
            dropout2=float(dropout2),
        ).to(self.device)

        return model

    # -------------------------
    # save / load
    # -------------------------
    def save(self, path: Path):
        if self.model is None:
            raise ValueError("model is None")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "task": "regression" if self.is_regression else "classification",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model_path": str(model_path),
        }
        self.save_meta(save_dir, meta)
        return model_path

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        if "task" not in meta:
            raise ValueError("saved meta has no 'task'")

        task = meta["task"]
        if task == "regression":
            self.is_regression = True
        elif task == "classification":
            self.is_regression = False
        else:
            raise ValueError(f"Unknown saved task: {task}")

        if "input_dim" not in meta:
            raise ValueError("saved meta has no 'input_dim'")
        if "output_dim" not in meta:
            raise ValueError("saved meta has no 'output_dim'")
        if "model_path" not in meta:
            raise ValueError("saved meta has no 'model_path'")

        self.input_dim = int(meta["input_dim"])
        self.output_dim = int(meta["output_dim"])

        model_path = Path(meta["model_path"])
        self.model = self._get_model(self.input_dim, self.output_dim)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        return True
