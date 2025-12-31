# src/models/xgboost_adapter.py

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from pathlib import Path

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics


class XGBoostAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.model: XGBClassifier | XGBRegressor | None = None
        self.is_regression: bool = False

    def _is_regression_from_meta(self, data: Datasets) -> bool:
        meta = data.meta
        if "task" not in meta:
            return False

        task = meta["task"]
        if task == "regression":
            return True
        if task == "classification":
            return False

        raise ValueError(f"Unknown meta.task: {task}")

    def _resolve_xgb_params(self, data: Datasets) -> tuple[str, str, int | None]:
        # returns: (objective, eval_metric, num_class_or_none)
        is_reg = self._is_regression_from_meta(data)

        if is_reg:
            return "reg:squarederror", "rmse", None

        num_class = data.get_num_class()
        if num_class == 2:
            return "binary:logistic", "logloss", None
        return "multi:softprob", "mlogloss", num_class

    def _resolve_from_config_or_auto(
        self, data: Datasets
    ) -> tuple[str, str, int | None]:
        auto_objective, auto_eval_metric, auto_num_class = self._resolve_xgb_params(
            data
        )

        objective_cfg = self.config.model.objective
        eval_metric_cfg = self.config.model.eval_metric

        objective = auto_objective if objective_cfg == "auto" else objective_cfg
        eval_metric = auto_eval_metric if eval_metric_cfg == "auto" else eval_metric_cfg

        return objective, eval_metric, auto_num_class

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        self.is_regression = self._is_regression_from_meta(train_data)

        objective, eval_metric, auto_num_class = self._resolve_from_config_or_auto(
            train_data
        )

        if self.is_regression:
            print("[XGBoostAdapter] task: regression")
            print(f"[XGBoostAdapter] objective={objective}, eval_metric={eval_metric}")

            model = XGBRegressor(
                n_estimators=self.config.train.tree_est,
                objective=objective,
                random_state=self.config.train.seed,
                eval_metric=eval_metric,
                early_stopping_rounds=self.config.train.early_stopping_rounds,
                device=self.config.train.device,
                tree_method=self.config.train.tree_method,
            )

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
            )

            self.model = model

            ev = model.evals_result()
            train_vals = ev["validation_0"][eval_metric]
            valid_vals = ev["validation_1"][eval_metric]

            loss_tasks = [
                {
                    Split.TRAIN.value: train_vals,
                    Split.VALID.value: valid_vals,
                }
            ]

            y_tr_pred = model.predict(X_tr)
            y_val_pred = model.predict(X_val)

            train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
            valid_metrics = compute_regression_metrics(y_val, y_val_pred)

            results = {
                "split": Split.TRAIN.value,
                "task": "regression",
                f"{Split.TRAIN.value}_metrics": train_metrics,
                f"{Split.VALID.value}_metrics": valid_metrics,
                "loss": {
                    "metric_name": eval_metric,
                    "tasks": loss_tasks,
                },
            }
            return results

        # -----------------------------
        # classification (default)
        # -----------------------------
        num_class = train_data.get_num_class()
        print("[XGBoostAdapter] task: classification")
        print(f"[XGBoostAdapter] num_class (from meta): {num_class}")
        print(f"[XGBoostAdapter] objective={objective}, eval_metric={eval_metric}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[XGBoostAdapter] num_class_from_y: {num_class_from_y}")

        xgb_kwargs = dict(
            n_estimators=self.config.train.tree_est,
            objective=objective,
            random_state=self.config.train.seed,
            eval_metric=eval_metric,
            early_stopping_rounds=self.config.train.early_stopping_rounds,
            device=self.config.train.device,
            tree_method=self.config.train.tree_method,
        )

        # multiclass에서만 num_class 전달
        if auto_num_class is not None:
            xgb_kwargs["num_class"] = auto_num_class

        model = XGBClassifier(**xgb_kwargs)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
        )

        self.model = model

        ev = model.evals_result()
        train_vals = ev["validation_0"][eval_metric]
        valid_vals = ev["validation_1"][eval_metric]

        loss_tasks = [
            {
                Split.TRAIN.value: train_vals,
                Split.VALID.value: valid_vals,
            }
        ]

        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        results = {
            "split": Split.TRAIN.value,
            "task": "classification",
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": eval_metric,
                "tasks": loss_tasks,
            },
        }
        return results

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")
        return self.model.predict(X)

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        X_all, y_all = test_data.get_data_for_gbdt()
        y_pred_all = self.predict(X_all)

        if self.is_regression:
            metrics_overall = compute_regression_metrics(y_all, y_pred_all)
        else:
            metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            ratio_dict = test_data.imputed_dict[p_v]

            for ratio in test_data.ratios:
                d = ratio_dict[ratio]
                X = d["X"]
                y = d["y"]

                y_pred = self.predict(X)

                if self.is_regression:
                    m = compute_regression_metrics(y, y_pred)
                else:
                    m = compute_classification_metrics(y, y_pred)

                by_ratio[p_v][ratio] = m

        results = {
            "split": Split.TEST.value,
            "task": "regression" if self.is_regression else "classification",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": by_ratio,
        }
        return results

    def save(self, path: Path):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "xgb_model.json"
        self.model.save_model(model_path)

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
        if task == "regression":
            model = XGBRegressor()
            self.is_regression = True
        elif task == "classification":
            model = XGBClassifier()
            self.is_regression = False
        else:
            raise ValueError(f"Unknown saved task: {task}")

        model.load_model(model_path)
        self.model = model

        return True
