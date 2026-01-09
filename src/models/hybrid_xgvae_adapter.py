# src/models/hybrid_xgvae_adapter.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import joblib

from src.core.models.ReGVAE_xai import ReGVAE
from src.core.utils.losses import ReGVAEFinalStage1Loss, info_nce_loss
from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics
from src.params.data_model import Split
from src.utils.embedding_vis import visualize_missing_mu_tsne
from src.utils.xai_save import save_xai_artifacts

# --- NEW: utilities ---
from src.utils.hybrid_xgvae_utils import (
    MemoryBank,
    export_clean_memory_bank,
    resolve_xgb_from_config_or_auto,
    resolve_lgbm_from_config_or_auto,
    infer_num_class_from_y,
    retrieve_agg_mu_xai,
    aggregate_retrieved_x_raw,
)

GBDTModel = XGBClassifier | XGBRegressor | LGBMClassifier | LGBMRegressor


class HybridXGVAEAdapter(BaseModelAdapter):
    """
    Stage1: ReGVAE (representation)
    Stage2: GBDT  (XGB / LightGBM)

    Stage2 feature (현재 구성):
      feat = concat[
        x_missing,             # (B, F)
        x_retrieved_raw,       # (B, F)
        mu_q,                  # (B, D)
        fused(mu_q, mu_r),     # (B, D)
        pattern_idx,           # (B, 1)  (원하면 제거)
      ]
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.config: Config = config
        self.device = self.config.train.device

        # models
        self.model: ReGVAE | None = None
        self.gbdt_model: GBDTModel | None = None

        # task dims
        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.is_regression: bool = False

        # stage2 backend
        m = self.config.model
        hm = str(getattr(m, "hybrid_mode", "xgb")).lower().strip()
        if hm in ("lgbm", "light_gbm", "lightgbm"):
            hm = "lightgbm"
        if hm not in ("xgb", "lightgbm"):
            hm = "xgb"
        self.hybrid_mode: str = hm

        # loss config (stage1)
        self.use_my_loss = bool(getattr(m, "use_my_loss", False))
        self.use_stage_1_ce = bool(getattr(m, "use_stage_1_ce", False))
        self.lambda_stage1_ce = float(getattr(m, "lambda_stage1_ce", 1.0))
        self.tau = float(getattr(m, "tau", 0.1))

        self.lambda_contrast = float(getattr(m, "lambda_contrast", 1.0))
        self.lambda_mu = float(getattr(m, "lambda_mu", 1.0))
        self.lambda_prior = float(getattr(m, "lambda_prior", 1.0))
        self.lambda_recon = float(getattr(m, "lambda_recon", 1.0))

        self.stage1_loss = None
        if self.use_my_loss:
            self.stage1_loss = ReGVAEFinalStage1Loss(
                tau=float(getattr(m, "tau", 0.1)),
                lambda_cls=float(getattr(m, "lambda_cls", 1.0)),
                w_contrast=self.lambda_contrast,
                w_mu=self.lambda_mu,
                w_prior=self.lambda_prior,
                w_recon=self.lambda_recon,
                mu_align=getattr(m, "mu_align", "l2"),
            )

        # retrieval config
        self.retrieval_k = int(getattr(m, "retrieval_k", 32))
        self.retrieval_tau = float(getattr(m, "retrieval_tau", 0.07))
        self.retrieval_chunk = int(getattr(m, "retrieval_chunk", 200000))

        # visualization config
        self.vis_max_points = int(getattr(m, "vis_max_points", 5000))
        self.vis_pca_dim = int(getattr(m, "vis_pca_dim", 50))
        self.vis_perplexity = float(getattr(m, "vis_perplexity", 30.0))
        self.vis_seed = int(getattr(m, "vis_seed", 0))

        # memory bank (util)
        self.bank: MemoryBank | None = None
        self._last_memory_path: Path | None = None

    # -------------------------
    # task inference
    # -------------------------
    def _is_regression_from_meta(self, data: Datasets) -> bool:
        meta = data.meta

        # dict meta
        if isinstance(meta, dict):
            t = str(meta.get("task", "")).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False
            # num_class 힌트가 있으면 사용
            if "num_class" in meta:
                return int(meta["num_class"]) <= 1
            return False

        # object meta
        if hasattr(meta, "task"):
            t = str(getattr(meta, "task")).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False

        if hasattr(meta, "num_class"):
            return int(getattr(meta, "num_class")) <= 1

        return False

    def _infer_num_class_from_dataset(self, data: Datasets) -> int:
        y = np.asarray(data.imputed_dict["original"]["y"])
        uniq = np.unique(y)
        if uniq.ndim != 1 or uniq.size < 2:
            return 2  # 최소 fallback
        return int(uniq.size)

    def _infer_input_dim_output_dim(
        self, train_data: Datasets, valid_data: Datasets
    ) -> tuple[int, int, bool]:
        input_dim = int(train_data.meta.input_dim)

        is_reg_tr = self._is_regression_from_meta(train_data)
        is_reg_vl = self._is_regression_from_meta(valid_data)
        if is_reg_tr != is_reg_vl:
            # 실제론 데이터셋 구성 문제이므로 멈추는 게 안전합니다.
            raise ValueError(
                f"train/valid task mismatch: train={is_reg_tr} valid={is_reg_vl}"
            )

        if is_reg_tr:
            return input_dim, 1, True

        # 분류면 class 수를 y에서 추론
        n_tr = self._infer_num_class_from_dataset(train_data)
        n_vl = self._infer_num_class_from_dataset(valid_data)
        if n_tr != n_vl:
            raise ValueError(
                f"train/valid num_class mismatch: train={n_tr}, valid={n_vl}"
            )

        return input_dim, int(n_tr), False

    # -------------------------
    # pred helpers
    # -------------------------
    def _pred_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.is_regression:
            return F.mse_loss(pred.squeeze(-1), y.float())
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
        name = self.config.model.model.name

        # -------------------------
        # Stage 1
        # -------------------------
        stage1_epochs = int(self.config.train.epochs)
        opt1, sch1 = self._make_optimizer_scheduler(stage=1, num_epochs=stage1_epochs)

        best_valid = None
        best_state = None
        patience = 0
        max_patience = int(self.config.train.early_stopping_rounds)

        train_total_1: list[float] = []
        valid_total_1: list[float] = []
        train_contrast_1: list[float] = []
        valid_contrast_1: list[float] = []
        train_pred_1: list[float] = []
        valid_pred_1: list[float] = []
        train_view_1: list[float] = []
        valid_view_1: list[float] = []
        train_kl_1: list[float] = []
        valid_kl_1: list[float] = []
        train_recon_1: list[float] = []
        valid_recon_1: list[float] = []

        for epoch in range(stage1_epochs):
            tr = self.run_epoch(tr_loader, opt1, Split.TRAIN)
            vl = self.run_epoch(vl_loader, None, Split.VALID)

            lr = float(opt1.param_groups[0]["lr"])
            print(
                f"[{name} Stage1 Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, contrast={tr['contrast']:.4f}, pred={tr['pred']:.4f}, "
                f"view={tr['view']:.4f}, kl={tr['kl']:.4f}, recon={tr['recon']:.4f} | "
                f"Valid: total={vl['total']:.4f}, contrast={vl['contrast']:.4f}, pred={vl['pred']:.4f}, "
                f"view={vl['view']:.4f}, kl={vl['kl']:.4f}, recon={vl['recon']:.4f} | "
                f"LR={lr:.6f}"
            )

            train_total_1.append(tr["total"])
            valid_total_1.append(vl["total"])
            train_contrast_1.append(tr["contrast"])
            valid_contrast_1.append(vl["contrast"])
            train_pred_1.append(tr["pred"])
            valid_pred_1.append(vl["pred"])
            train_view_1.append(tr["view"])
            valid_view_1.append(vl["view"])
            train_kl_1.append(tr["kl"])
            valid_kl_1.append(vl["kl"])
            train_recon_1.append(tr["recon"])
            valid_recon_1.append(vl["recon"])

            sch1.step()

            if best_valid is None or vl["total"] < best_valid:
                best_valid = vl["total"]
                patience = 0
                best_state = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience += 1
                if max_patience > 0 and patience >= max_patience:
                    print(f"[{name}] Stage1 Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        root = (
            self.work_dir
            if self.work_dir is not None
            else Path(self.config.train.output_dir)
        )

        # -------------------------
        # memory bank export (util)
        # -------------------------
        mem_dir = root / "memory_bank"
        mem_path = export_clean_memory_bank(
            model=self.model,
            dataset=train_data,
            device=self.device,
            is_regression=self.is_regression,
            output_dim=int(self.output_dim or 1),
            save_dir=mem_dir,
            tag="train",
            bs=4096,
        )
        self._last_memory_path = mem_path
        print("memory saved:", mem_path)

        # embedding vis
        vis_dir = root / "embedding_vis"
        vis_path = visualize_missing_mu_tsne(
            model=self.model,
            loader=vl_loader,
            device=self.device,
            save_dir=vis_dir,
            tag="valid_stage1",
            vis_max_points=self.vis_max_points,
            vis_pca_dim=self.vis_pca_dim,
            vis_perplexity=self.vis_perplexity,
            vis_seed=self.vis_seed,
        )

        # -------------------------
        # Stage 2
        # -------------------------
        self._load_bank(mem_path)

        X_tr, y_tr, _, _ = self._build_hybrid_features(
            loader=tr_loader, split=Split.TRAIN, memory_path=mem_path, return_xai=False
        )
        X_vl, y_vl, _, _ = self._build_hybrid_features(
            loader=vl_loader, split=Split.VALID, memory_path=mem_path, return_xai=False
        )

        if self.hybrid_mode == "xgb":
            objective, eval_metric, num_class_param = resolve_xgb_from_config_or_auto(
                self.config.model, is_reg=self.is_regression, y_tr=y_tr, y_val=y_vl
            )

            if self.is_regression:
                gbdt = XGBRegressor(
                    n_estimators=self.config.train.tree_est,
                    objective=objective,
                    random_state=self.config.train.seed,
                    eval_metric=eval_metric,
                    early_stopping_rounds=self.config.train.early_stopping_rounds,
                    device=self.config.train.device,
                    tree_method=self.config.train.tree_method,
                )
            else:
                xgb_kwargs = dict(
                    n_estimators=self.config.train.tree_est,
                    objective=objective,
                    random_state=self.config.train.seed,
                    eval_metric=eval_metric,
                    early_stopping_rounds=self.config.train.early_stopping_rounds,
                    device=self.config.train.device,
                    tree_method=self.config.train.tree_method,
                )
                if num_class_param is not None:
                    xgb_kwargs["num_class"] = num_class_param
                gbdt = XGBClassifier(**xgb_kwargs)

            gbdt.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_vl, y_vl)])
            self.gbdt_model = gbdt

            ev = gbdt.evals_result()
            train_vals = ev["validation_0"][eval_metric]
            valid_vals = ev["validation_1"][eval_metric]

            y_tr_pred = gbdt.predict(X_tr)
            y_vl_pred = gbdt.predict(X_vl)

            if self.is_regression:
                train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
                valid_metrics = compute_regression_metrics(y_vl, y_vl_pred)
                task_name = "regression"
            else:
                train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
                valid_metrics = compute_classification_metrics(y_vl, y_vl_pred)
                task_name = "classification"

            stage2_meta = {
                "backend": "xgb",
                "objective": str(objective),
                "eval_metric": str(eval_metric),
                "feature_dim": int(X_tr.shape[1]),
            }

        else:
            objective, eval_metric, num_class_param = resolve_lgbm_from_config_or_auto(
                self.config.model, is_reg=self.is_regression, y_tr=y_tr, y_val=y_vl
            )
            device_type = str(self.config.train.device).lower()

            callbacks = []
            early_rounds = int(self.config.train.early_stopping_rounds)
            if early_rounds > 0:
                callbacks.append(
                    early_stopping(
                        stopping_rounds=early_rounds, first_metric_only=False
                    )
                )

            if self.is_regression:
                gbdt = LGBMRegressor(
                    n_estimators=self.config.train.tree_est,
                    objective=objective,
                    random_state=self.config.train.seed,
                    device_type=device_type,
                )
            else:
                class_weight = getattr(self.config.model, "lgbm_class_weight", None)
                lgb_kwargs = dict(
                    n_estimators=self.config.train.tree_est,
                    objective=objective,
                    random_state=self.config.train.seed,
                    device_type=device_type,
                    class_weight=class_weight,
                )
                if str(objective).lower() in ("multiclass", "multiclassova"):
                    # util이 num_class_param을 줄 수도 있고 없을 수도 있으므로 y에서 fallback
                    lgb_kwargs["num_class"] = int(
                        num_class_param or infer_num_class_from_y(y_tr, y_vl)
                    )
                gbdt = LGBMClassifier(**lgb_kwargs)

            gbdt.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_vl, y_vl)],
                eval_names=["train", "valid"],
                eval_metric=eval_metric,
                callbacks=callbacks,
            )
            self.gbdt_model = gbdt

            ev = gbdt.evals_result_
            train_vals = ev["train"][eval_metric]
            valid_vals = ev["valid"][eval_metric]

            y_tr_pred = gbdt.predict(X_tr)
            y_vl_pred = gbdt.predict(X_vl)

            if self.is_regression:
                train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
                valid_metrics = compute_regression_metrics(y_vl, y_vl_pred)
                task_name = "regression"
            else:
                train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
                valid_metrics = compute_classification_metrics(y_vl, y_vl_pred)
                task_name = "classification"

            stage2_meta = {
                "backend": "lightgbm",
                "objective": str(objective),
                "eval_metric": str(eval_metric),
                "device_type": str(device_type),
                "feature_dim": int(X_tr.shape[1]),
            }

        # -------------------------
        # XAI 저장 (Stage2)
        # -------------------------
        _, _, _, _, _, xai_valid = self.predict_xai(
            loader=vl_loader,
            split=Split.VALID,
            stage=2,
            memory_path=mem_path,
            max_batches=10,
        )
        xai_dir = root / "xai"
        save_xai_artifacts(
            xai=xai_valid,
            save_dir=xai_dir,
            tag="valid_stage2",
            top_rows=64,
            topk_feat=10,
        )

        results = {
            "split": Split.TRAIN.value,
            "task": task_name,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": "stage1_total_loss",
                "tasks": [
                    {
                        Split.TRAIN.value: train_total_1,
                        Split.VALID.value: valid_total_1,
                    },
                    {Split.TRAIN.value: train_vals, Split.VALID.value: valid_vals},
                ],
                "components": {
                    "stage1": {
                        "train": {
                            "contrast": train_contrast_1,
                            "pred": train_pred_1,
                            "view": train_view_1,
                            "kl": train_kl_1,
                            "recon": train_recon_1,
                        },
                        "valid": {
                            "contrast": valid_contrast_1,
                            "pred": valid_pred_1,
                            "view": valid_view_1,
                            "kl": valid_kl_1,
                            "recon": valid_recon_1,
                        },
                    },
                    "stage2_gbdt": stage2_meta,
                },
                "stage1_rep_loss": "contrast" if self.use_my_loss else "info_nce",
                "stage1_use_pred_loss": bool(self.use_stage_1_ce),
                "stage1_pred_weight": float(self.lambda_stage1_ce),
            },
            "artifacts": {
                "memory_bank": str(mem_path),
                "embedding_vis": str(vis_path),
                "xai_dir": str(xai_dir),
            },
        }
        return results

    def test(self, test_data: Datasets):
        if self.model is None or self.gbdt_model is None:
            raise ValueError("model/gbdt_model is not ready (call fit or load).")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        root = (
            self.work_dir
            if self.work_dir is not None
            else Path(self.config.train.output_dir)
        )

        mem_path = root / "memory_bank" / "memory_clean_train.pt"
        if not mem_path.exists():
            alt = root / Split.TRAIN.value / "save" / "memory_clean_train.pt"
            if alt.exists():
                mem_path = alt
        if not mem_path.exists():
            raise ValueError(f"train memory bank not found: {mem_path}")

        self._load_bank(mem_path)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, split=Split.TEST, stage=2, memory_path=mem_path, return_xai=False
        )

        _, _, _, _, _, xai_test = self.predict_xai(
            loader=te_loader,
            split=Split.TEST,
            stage=2,
            memory_path=mem_path,
            max_batches=10,
        )
        xai_dir = root / "xai"
        save_xai_artifacts(
            xai=xai_test,
            save_dir=xai_dir,
            tag="test_stage2",
            top_rows=64,
            topk_feat=10,
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
                if np.any(mask):
                    y_sub = labels_all[mask]
                    y_hat_sub = preds_all[mask]
                    if self.is_regression:
                        m = compute_regression_metrics(y_sub, y_hat_sub)
                    else:
                        m = compute_classification_metrics(y_sub, y_hat_sub)
                    metrics_by_ratio[p_val][ratio] = m

        return {
            "split": Split.TEST.value,
            "task": "regression" if self.is_regression else "classification",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
            "artifacts": {"xai_dir": str(xai_dir)},
        }

    # -------------------------
    # predict / xai
    # -------------------------
    @torch.no_grad()
    def predict_xai(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
        stage: int = 2,
        memory_path: Path | None = None,
        max_batches: int | None = None,
    ):
        return self.predict(
            loader=loader,
            split=split,
            stage=stage,
            memory_path=memory_path,
            return_xai=True,
            max_batches=max_batches,
        )

    def run_epoch(self, loader: DataLoader, optimizer, split: Split):
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(f"{self.config.model.model.name}_stage1", split)

        total_sum = 0.0
        contrast_sum = 0.0
        pred_sum = 0.0
        view_sum = 0.0
        kl_sum = 0.0
        recon_sum = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, x_ori, _, _, _, _, _, V = self._prepare_batch(batch)

            x_clean = x_ori[::V] if V > 1 else x_ori
            y_base = y[::V] if V > 1 else y

            with torch.set_grad_enabled(is_train):
                out_clean = self.model(x_cont=x_clean, x_cat=None, return_attn=False)
                out_missing = self.model(x_cont=x, x_cat=None, return_attn=False)

                # stage1 rep loss
                if self.use_my_loss and self.stage1_loss is not None:
                    y_for_loss = None if self.is_regression else y_base
                    loss_dict = self.stage1_loss(
                        mu_clean=out_clean["z_mu"],
                        logvar_clean=out_clean["z_logvar"],
                        mu_missing=out_missing["z_mu"],
                        logvar_missing=out_missing["z_logvar"],
                        x_clean=x_clean,
                        recon_clean=out_clean["recon"],
                        recon_missing=out_missing["recon"],
                        y_base=y_for_loss,
                        views_per_base=V,
                    )
                    loss_total = loss_dict["total"]
                    loss_contrast = loss_dict["contrast"]
                    loss_view = loss_dict["mu_align"]
                    loss_kl = loss_dict["prior"]
                    loss_recon = loss_dict["recon"]
                else:
                    mu_clean = out_clean["z_mu"]
                    mu_missing = out_missing["z_mu"]
                    mu_clean_rep = (
                        mu_clean.repeat_interleave(V, dim=0) if V > 1 else mu_clean
                    )
                    loss_contrast = info_nce_loss(
                        mu_clean_rep, mu_missing, temperature=self.tau
                    )
                    loss_total = loss_contrast
                    loss_view = torch.zeros((), device=self.device)
                    loss_kl = torch.zeros((), device=self.device)
                    loss_recon = torch.zeros((), device=self.device)

                # optional CE/MSE
                loss_pred = torch.zeros((), device=self.device)
                if self.use_stage_1_ce:
                    pred_clean = out_clean.get("logits", None)
                    pred_missing = out_missing.get("logits", None)
                    if pred_clean is not None and pred_missing is not None:
                        l_clean = self._pred_loss(pred_clean, y_base)
                        l_miss = self._pred_loss(pred_missing, y)
                        loss_pred = 0.5 * (l_clean + l_miss)
                        loss_total = loss_total + self.lambda_stage1_ce * loss_pred

                if is_train and optimizer is not None:
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

            num_batches += 1
            total_sum += float(loss_total.item())
            contrast_sum += float(loss_contrast.item())
            pred_sum += float(loss_pred.item())
            view_sum += float(loss_view.item())
            kl_sum += float(loss_kl.item())
            recon_sum += float(loss_recon.item())

        denom = max(1, num_batches)
        return {
            "total": total_sum / denom,
            "contrast": contrast_sum / denom,
            "pred": pred_sum / denom,
            "view": view_sum / denom,
            "kl": kl_sum / denom,
            "recon": recon_sum / denom,
        }

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
        stage: int = 2,
        memory_path: Path | None = None,
        return_xai: bool = False,
        max_batches: int | None = None,
    ):
        if self.model is None:
            raise ValueError("model is None")

        # -------- stage1: deep logits only --------
        if stage == 1:
            self.model.eval()

            total_loss = 0.0
            num_batches = 0

            all_preds: List[torch.Tensor] = []
            all_labels: List[torch.Tensor] = []
            all_pattern_idx: List[torch.Tensor] = []
            all_ratio_idx: List[torch.Tensor] = []

            xai_pack: Dict[str, List[torch.Tensor]] = {}
            if return_xai:
                xai_pack["attn_cls_feat"] = []

            desc = self.get_desc(self.config.model.model.name, split)
            for bi, batch in enumerate(tqdm(loader, desc=desc)):
                if max_batches is not None and bi >= max_batches:
                    break

                x, y, _, _, pattern_idx, ratio_idx, _, _, _ = self._prepare_batch(batch)
                out = self.model(x_cont=x, x_cat=None, return_attn=return_xai)
                pred = out.get("logits", None)
                if pred is None:
                    # logits가 없으면 stage1-only predict는 의미가 없어서 중단이 안전합니다.
                    raise ValueError(
                        "stage1 logits is None (check ReGVAE forward output)."
                    )

                loss = self._pred_loss(pred, y)
                out_pred = self._pred_to_output(pred)

                if return_xai:
                    attn_cls_feat = self._extract_cls_to_feature_attention(out)
                    xai_pack["attn_cls_feat"].append(attn_cls_feat.detach().cpu())

                total_loss += float(loss.item())
                num_batches += 1

                all_preds.append(out_pred.detach().cpu())
                all_labels.append(y.detach().cpu())
                all_pattern_idx.append(pattern_idx.detach().cpu())
                all_ratio_idx.append(ratio_idx.detach().cpu())

            avg_loss = total_loss / max(1, num_batches)

            preds_all = torch.cat(all_preds, dim=0).numpy()
            labels_all = torch.cat(all_labels, dim=0).numpy()
            pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
            ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

            if not return_xai:
                return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

            xai_out: Dict[str, np.ndarray] = {}
            for k, vs in xai_pack.items():
                if len(vs) > 0:
                    xai_out[k] = torch.cat(vs, dim=0).numpy()

            return (
                avg_loss,
                preds_all,
                labels_all,
                pattern_idx_all,
                ratio_idx_all,
                xai_out,
            )

        # -------- stage2 --------
        if self.gbdt_model is None:
            raise ValueError("gbdt_model is None")
        if memory_path is None:
            raise ValueError("stage2 requires memory_path")

        if self.bank is None:
            self._load_bank(memory_path)

        if return_xai:
            X, y, p, r, xai_out = self._build_hybrid_features(
                loader=loader,
                split=split,
                memory_path=memory_path,
                return_xai=True,
                max_batches=max_batches,
            )
        else:
            X, y, p, r = self._build_hybrid_features(
                loader=loader,
                split=split,
                memory_path=memory_path,
                return_xai=False,
                max_batches=max_batches,
            )
            xai_out = None

        y_hat = self.gbdt_model.predict(X)

        if self.is_regression:
            avg_loss = float(
                np.mean((y_hat.astype(np.float32) - y.astype(np.float32)) ** 2)
            )
        else:
            avg_loss = float(
                np.mean(
                    (y_hat.astype(np.int64) != y.astype(np.int64)).astype(np.float32)
                )
            )

        if not return_xai:
            return avg_loss, y_hat, y, p, r

        return avg_loss, y_hat, y, p, r, xai_out

    # -------------------------
    # hybrid feature builder (Stage2)
    # -------------------------
    @torch.no_grad()
    def _build_hybrid_features(
        self,
        loader: DataLoader,
        split: Split,
        memory_path: Path,
        return_xai: bool = False,
        max_batches: int | None = None,
    ):
        if self.model is None:
            raise ValueError("model is None")
        if self.bank is None:
            self._load_bank(memory_path)

        self.model.eval()

        feats_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        p_list: List[np.ndarray] = []
        r_list: List[np.ndarray] = []

        xai_pack: Dict[str, List[np.ndarray]] = {}
        if return_xai:
            xai_pack = {
                "gate_g": [],
                "retr_idx": [],
                "retr_sim": [],
                "retr_w": [],
                "attn_cls_feat": [],
                "retr_x_raw": [],
            }

        desc = self.get_desc(f"{self.config.model.model.name}_hybrid_feat", split)

        for bi, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches is not None and bi >= max_batches:
                break

            x, y_t, _, _, pattern_idx, ratio_idx, base_idx, _, _ = self._prepare_batch(
                batch
            )

            if return_xai:
                out = self.model(x_cont=x, x_cat=None, return_attn=True)
                mu_q = out["z_mu"]
                attn_cls_feat = self._extract_cls_to_feature_attention(out)
            else:
                enc = self.model.encode_only(x_cont=x, x_cat=None, return_attn=False)
                mu_q = enc["z_mu"]
                attn_cls_feat = None

            mu_r, best_idx, best_sim, w = retrieve_agg_mu_xai(
                bank=self.bank,
                mu_q=mu_q,
                device=self.device,
                retrieval_k=self.retrieval_k,
                retrieval_tau=self.retrieval_tau,
                retrieval_chunk=self.retrieval_chunk,
                exclude_idx=base_idx if split == Split.TRAIN else None,
            )

            fused, g = self.model.feat_gate(mu_q, mu_r, return_gate=True)
            x_r = aggregate_retrieved_x_raw(
                bank=self.bank,
                best_idx=best_idx,
                w=w,
                device=self.device,
            )

            feat = torch.cat(
                [
                    x.float(),  # x_missing
                    x_r.float(),  # retrieved raw aggregate
                    mu_q.float(),  # query embedding
                    fused.float(),  # gated embedding
                    pattern_idx.float().unsqueeze(1),  # optional
                ],
                dim=1,
            )
            feats_list.append(feat.detach().cpu().numpy().astype(np.float32))

            if self.is_regression:
                y_list.append(y_t.detach().cpu().numpy().astype(np.float32))
            else:
                y_list.append(y_t.detach().cpu().numpy().astype(np.int64))
            p_list.append(pattern_idx.detach().cpu().numpy().astype(np.int64))
            r_list.append(ratio_idx.detach().cpu().numpy().astype(np.int64))

            if return_xai:
                xai_pack["gate_g"].append(g.detach().cpu().numpy())
                xai_pack["retr_idx"].append(best_idx.detach().cpu().numpy())
                xai_pack["retr_sim"].append(best_sim.detach().cpu().numpy())
                xai_pack["retr_w"].append(w.detach().cpu().numpy())
                xai_pack["retr_x_raw"].append(x_r.detach().cpu().numpy())
                if attn_cls_feat is not None:
                    xai_pack["attn_cls_feat"].append(
                        attn_cls_feat.detach().cpu().numpy()
                    )

        X = (
            np.concatenate(feats_list, axis=0)
            if feats_list
            else np.zeros((0, 1), dtype=np.float32)
        )
        y = (
            np.concatenate(y_list, axis=0)
            if y_list
            else np.zeros((0,), dtype=np.float32)
        )
        p = np.concatenate(p_list, axis=0) if p_list else np.zeros((0,), dtype=np.int64)
        r = np.concatenate(r_list, axis=0) if r_list else np.zeros((0,), dtype=np.int64)

        if not return_xai:
            return X, y, p, r

        xai_out: Dict[str, np.ndarray] = {}
        for k, vs in xai_pack.items():
            if len(vs) > 0:
                xai_out[k] = np.concatenate(vs, axis=0)

        return X, y, p, r, xai_out

    # -------------------------
    # memory bank (util wrapper)
    # -------------------------
    @torch.no_grad()
    def _load_bank(self, path: Path):
        # bank은 내부적으로 x_raw 존재 체크까지 수행합니다(없으면 의미가 없으므로).
        self.bank = MemoryBank.load(path=path, device=self.device, to_device=True)

    # -------------------------
    # optim
    # -------------------------
    def _make_optimizer_scheduler(self, stage: int, num_epochs: int):
        if self.model is None:
            raise ValueError("model is None")

        if stage == 1:
            lr = self.config.train.lr_stage_1
            lr_min = self.config.train.lr_min_stage_1
        else:
            lr = self.config.train.lr_stage_2
            lr_min = self.config.train.lr_min_stage_2

        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, num_epochs), eta_min=lr_min
        )
        return opt, sch

    # -------------------------
    # attn extraction
    # -------------------------
    def _extract_cls_to_feature_attention(
        self, out: Dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        # return_xai=True에서만 호출되므로, 없으면 멈추는 게 안전합니다.
        attn_list = out.get("attn_enc", None)
        if attn_list is None or len(attn_list) == 0:
            raise ValueError("attn_enc is missing (return_attn=True required).")

        last = attn_list[-1]
        if last is None:
            raise ValueError("last attention is None")

        attn_mean = last.mean(dim=1)  # (B, T, T)
        cls_to_all = attn_mean[:, 0, :]  # (B, T)
        cls_to_feat = cls_to_all[:, 1:]  # (B, F_tokens)

        denom = torch.clamp(cls_to_feat.sum(dim=1, keepdim=True), min=1e-12)
        return cls_to_feat / denom

    def _prepare_batch(self, batch: dict):
        x = batch["x"].to(self.device)

        y_raw = batch["y"].to(self.device)
        y = y_raw.float() if self.is_regression else y_raw.long()

        x_ori = batch["x_originals"].to(self.device)
        bemv = batch["bemv"].to(self.device)
        pattern_idx = batch["pattern_idx"].to(self.device)
        ratio_idx = batch["ratio_idx"].to(self.device)
        base_idx = batch["base_idx"].to(self.device)
        V = batch["views_per_base"].item()
        B0 = batch["base_batch_size"].item()
        return x, y, x_ori, bemv, pattern_idx, ratio_idx, base_idx, B0, V

    # -------------------------
    # model
    # -------------------------
    def _get_model(self, input_dim: int, output_dim: int) -> ReGVAE:
        ft_kwargs = ReGVAE.get_default_kwargs()
        return ReGVAE(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=output_dim,
            latent_dim=None,
            logits_from="mu",
            **ft_kwargs,
        ).to(self.device)

    # -------------------------
    # save / load
    # -------------------------
    def save(self, path: Path):
        if self.model is None or self.gbdt_model is None:
            raise ValueError("model/gbdt_model is None")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        deep_path = save_dir / f"{self.config.model.model.name}_deep.pt"
        torch.save(self.model.state_dict(), deep_path)

        if self.hybrid_mode == "xgb":
            gbdt_path = save_dir / "xgb_model.json"
            self.gbdt_model.save_model(gbdt_path)  # type: ignore[attr-defined]
        else:
            gbdt_path = save_dir / "lgbm_model.pkl"
            joblib.dump(self.gbdt_model, gbdt_path)

        # memory bank copy (가능하면 포함)
        mem_dst = save_dir / "memory_clean_train.pt"
        mem_src = None
        if self._last_memory_path is not None and Path(self._last_memory_path).exists():
            mem_src = Path(self._last_memory_path)
        else:
            root = (
                self.work_dir
                if self.work_dir is not None
                else Path(self.config.train.output_dir)
            )
            cand = root / "memory_bank" / "memory_clean_train.pt"
            if cand.exists():
                mem_src = cand

        if mem_src is not None:
            try:
                shutil.copyfile(mem_src, mem_dst)
            except Exception:
                pass

        meta = {
            "task": "regression" if self.is_regression else "classification",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hybrid_mode": self.hybrid_mode,
            "deep_model_path": str(deep_path),
            "gbdt_model_path": str(gbdt_path),
            "memory_bank_path": str(mem_dst) if mem_dst.exists() else None,
        }
        self.save_meta(save_dir, meta)
        return deep_path

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.is_regression = meta.get("task", "classification") == "regression"

        hm = str(meta.get("hybrid_mode", "xgb")).lower().strip()
        if hm in ("lgbm", "light_gbm", "lightgbm"):
            hm = "lightgbm"
        if hm not in ("xgb", "lightgbm"):
            hm = "xgb"
        self.hybrid_mode = hm

        self.input_dim = int(meta["input_dim"])
        self.output_dim = int(meta["output_dim"])

        deep_path = Path(meta["deep_model_path"])
        self.model = self._get_model(self.input_dim, self.output_dim)
        state = torch.load(deep_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        gbdt_path = Path(meta["gbdt_model_path"])
        if self.hybrid_mode == "xgb":
            self.gbdt_model = XGBRegressor() if self.is_regression else XGBClassifier()
            self.gbdt_model.load_model(gbdt_path)  # type: ignore[union-attr]
        else:
            self.gbdt_model = joblib.load(gbdt_path)

        mem_path = meta.get("memory_bank_path", None)
        if mem_path:
            mp = Path(mem_path)
            if mp.exists():
                self._load_bank(mp)
                self._last_memory_path = mp

        return True
