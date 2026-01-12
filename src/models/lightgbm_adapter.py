# src/models/lightgbm_adapter.py

from __future__ import annotations

import numpy as np
from pathlib import Path

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import joblib

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics


class LightGBMAdapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)
        self.model: LGBMClassifier | LGBMRegressor | None = None
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

    # -------------------------
    # bemv -> NaN restore
    # -------------------------
    def _apply_nan_from_bemv(self, X: np.ndarray, bemv: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        assert isinstance(bemv, np.ndarray)
        assert X.shape == bemv.shape, f"X.shape={X.shape}, bemv.shape={bemv.shape}"

        if X.dtype.kind != "f":
            X = X.astype(np.float32)

        X_nan = X.copy()
        miss_mask = bemv == 0
        X_nan[miss_mask] = np.nan
        return X_nan

    def _stack_gbdt_xy_with_nan(self, data: Datasets) -> tuple[np.ndarray, np.ndarray]:
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            ratio_dict = data.imputed_dict[p_v]

            for ratio in data.ratios:
                d = ratio_dict[ratio]
                X_imp = d["X"]
                y = d["y"]
                bemv = d["bemv"]

                X_list.append(self._apply_nan_from_bemv(X_imp, bemv))
                y_list.append(y)

        X_cat = np.concatenate(X_list, axis=0)
        y_cat = np.concatenate(y_list, axis=0)
        return X_cat, y_cat

    def _get_ratio_xy_with_nan(
        self, data: Datasets, pattern_v: str, ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        d = data.imputed_dict[pattern_v][ratio]
        X_nan = self._apply_nan_from_bemv(d["X"], d["bemv"])
        return X_nan, d["y"]

    # -------------------------
    # helpers (classification)
    # -------------------------
    def _infer_num_class_from_y(self, y_tr: np.ndarray, y_val: np.ndarray) -> int:
        y_cat = np.concatenate([y_tr, y_val], axis=0)
        uniq = np.unique(y_cat)
        assert uniq.ndim == 1
        assert uniq.size >= 2, f"num_class must be >= 2, got uniq={uniq}"

        mn = int(uniq.min())
        mx = int(uniq.max())
        assert mn == 0, f"class labels must start at 0. got min={mn}, uniq={uniq}"
        assert mx == int(
            uniq.size - 1
        ), f"class labels must be contiguous 0..C-1. got max={mx}, size={uniq.size}, uniq={uniq}"
        return int(uniq.size)

    def _resolve_device_type(self) -> str:
        return str(self.config.train.device).lower()

    # -------------------------
    # train / eval
    # -------------------------
    def fit(self, train_data: Datasets, valid_data: Datasets):
        X_tr, y_tr = self._stack_gbdt_xy_with_nan(train_data)
        X_val, y_val = self._stack_gbdt_xy_with_nan(valid_data)

        self.is_regression = self._is_regression_from_meta(train_data)

        device_type = self._resolve_device_type()
        early_rounds = int(self.config.train.early_stopping_rounds)

        callbacks = []
        if early_rounds > 0:
            callbacks.append(
                early_stopping(
                    stopping_rounds=early_rounds,
                    first_metric_only=False,
                )
            )

        # -----------------------------
        # regression
        # -----------------------------
        if self.is_regression:
            objective_cfg = str(self.config.model.lgbm_objective).lower()
            metric_cfg = str(self.config.model.lgbm_metric).lower()

            objective = (
                "regression"
                if objective_cfg == "auto"
                else self.config.model.lgbm_objective
            )
            eval_metric = (
                "rmse" if metric_cfg == "auto" else self.config.model.lgbm_metric
            )

            if not str(objective).startswith("reg"):
                raise ValueError(
                    f"Regression task requires regression objective, got objective={objective}"
                )

            print("[LightGBMAdapter] task: regression")
            print(f"[LightGBMAdapter] device_type: {device_type}")
            print(f"[LightGBMAdapter] objective: {objective}")
            print(f"[LightGBMAdapter] eval_metric: {eval_metric}")

            lgb_kwargs = dict(
                n_estimators=self.config.train.tree_est,
                objective=objective,
                random_state=self.config.train.seed,
                device_type=device_type,
            )

            model = LGBMRegressor(**lgb_kwargs)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                eval_names=["train", "valid"],
                eval_metric=eval_metric,
                callbacks=callbacks,
            )

            self.model = model

            ev = model.evals_result_
            train_vals = ev["train"][eval_metric]
            valid_vals = ev["valid"][eval_metric]

            y_tr_pred = model.predict(X_tr)
            y_val_pred = model.predict(X_val)

            train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
            valid_metrics = compute_regression_metrics(y_val, y_val_pred)

            return {
                "split": Split.TRAIN.value,
                "task": "regression",
                f"{Split.TRAIN.value}_metrics": train_metrics,
                f"{Split.VALID.value}_metrics": valid_metrics,
                "loss": {
                    "metric_name": eval_metric,
                    "tasks": [
                        {Split.TRAIN.value: train_vals, Split.VALID.value: valid_vals}
                    ],
                },
            }

        # -----------------------------
        # classification
        # -----------------------------
        n_classes_y = self._infer_num_class_from_y(y_tr, y_val)

        objective_cfg = str(self.config.model.lgbm_objective).lower()
        metric_cfg = str(self.config.model.lgbm_metric).lower()

        if objective_cfg == "auto":
            objective = "binary" if n_classes_y == 2 else "multiclass"
        else:
            objective = self.config.model.lgbm_objective

        if metric_cfg == "auto":
            eval_metric = "binary_logloss" if objective == "binary" else "multi_logloss"
        else:
            eval_metric = self.config.model.lgbm_metric

        print("[LightGBMAdapter] task: classification")
        print(f"[LightGBMAdapter] num_class_from_y: {n_classes_y}")
        print(f"[LightGBMAdapter] device_type: {device_type}")
        print(f"[LightGBMAdapter] objective: {objective}")
        print(f"[LightGBMAdapter] eval_metric: {eval_metric}")

        lgb_kwargs = dict(
            n_estimators=self.config.train.tree_est,
            objective=objective,
            random_state=self.config.train.seed,
            device_type=device_type,
            class_weight=self.config.model.lgbm_class_weight,
        )

        if objective == "multiclass":
            lgb_kwargs["num_class"] = n_classes_y

        model = LGBMClassifier(**lgb_kwargs)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric=eval_metric,
            callbacks=callbacks,
        )

        self.model = model

        ev = model.evals_result_
        train_vals = ev["train"][eval_metric]
        valid_vals = ev["valid"][eval_metric]

        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        return {
            "split": Split.TRAIN.value,
            "task": "classification",
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": eval_metric,
                "tasks": [
                    {Split.TRAIN.value: train_vals, Split.VALID.value: valid_vals}
                ],
            },
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")
        return self.model.predict(X)

    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        X_all, y_all = self._stack_gbdt_xy_with_nan(test_data)
        y_pred_all = self.model.predict(X_all)

        if self.is_regression:
            metrics_overall = compute_regression_metrics(y_all, y_pred_all)
        else:
            metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            for ratio in test_data.ratios:
                X, y = self._get_ratio_xy_with_nan(test_data, p_v, ratio)
                y_pred = self.model.predict(X)

                if self.is_regression:
                    m = compute_regression_metrics(y, y_pred)
                else:
                    m = compute_classification_metrics(y, y_pred)

                by_ratio[p_v][ratio] = m

        return {
            "split": Split.TEST.value,
            "task": "regression" if self.is_regression else "classification",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": by_ratio,
        }

    def save(self, path: Path):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "lgbm_model.pkl"
        joblib.dump(self.model, model_path)

        meta = {
            "model_path": str(model_path),
            "task": "regression" if self.is_regression else "classification",
        }
        self.save_meta(save_dir, meta)

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]
        task = meta["task"] if "task" in meta else "classification"

        model = joblib.load(model_path)
        self.model = model
        self.is_regression = task == "regression"
        return True
