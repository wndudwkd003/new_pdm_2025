# src/models/xgboost_adapter.py

import numpy as np
from xgboost import XGBClassifier
from pathlib import Path

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics


class XGBoostAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.model: XGBClassifier | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        # X: (N, F), y: (N,)
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        eval_metric = self.config.model.eval_metric
        num_class = train_data.get_num_class()

        print(f"[XGBoostAdapter] num_class (from meta): {num_class}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[XGBoostAdapter] num_class_from_y: {num_class_from_y}")

        model = XGBClassifier(
            n_estimators=self.config.train.epochs,
            objective=self.config.model.objective,
            num_class=num_class,
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

        y_tr_pred = model.predict(X_tr)    # (N,)
        y_val_pred = model.predict(X_val)  # (N,)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        results = {
            "split": Split.TRAIN.value,
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

        y_pred = self.model.predict(X)
        return y_pred

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        # 전체 테스트 데이터 기준 성능
        X_all, y_all = test_data.get_data_for_gbdt()   # (N, F), (N,)
        y_pred_all = self.predict(X_all)               # (N,)

        metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        # 패턴/ratio 별 성능
        by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            ratio_dict = test_data.imputed_dict[p_v]

            for ratio in test_data.ratios:
                d = ratio_dict[ratio]

                X = d["X"]  # (N, F)
                y = d["y"]  # (N,)

                y_pred = self.predict(X)  # (N,)

                m = compute_classification_metrics(y, y_pred)
                by_ratio[p_v][ratio] = m

        results = {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": by_ratio,
        }

        return results

    def save(
        self,
        path: Path
    ):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "xgb_model.json"
        self.model.save_model(model_path)

        meta = {
            "model_path": str(model_path),
        }

        self.save_meta(save_dir, meta)

    def load(
        self,
        path: Path
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]

        model = XGBClassifier()
        model.load_model(model_path)

        self.model = model

        return True
