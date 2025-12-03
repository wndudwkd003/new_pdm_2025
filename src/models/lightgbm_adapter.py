# src/models/lightgbm_adapter.py

import numpy as np
from pathlib import Path

from lightgbm import LGBMClassifier, early_stopping
import joblib

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics


class LightGBMAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.model: LGBMClassifier | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        # X: (N, F), y: (N,)
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        eval_metric = self.config.model.lgbm_metric
        num_class = train_data.get_num_class()

        print(f"[LightGBMAdapter] num_class (from meta): {num_class}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[LightGBMAdapter] num_class_from_y: {num_class_from_y}")

        device_type = self.config.train.device
        print(f"[LightGBMAdapter] device_type: {device_type}")

        model = LGBMClassifier(
            objective=self.config.model.lgbm_objective,
            num_class=num_class,
            random_state=self.config.train.seed,
            device_type=device_type,
            class_weight=self.config.model.lgbm_class_weight,
        )

        callbacks = []
        if self.config.train.early_stopping_rounds > 0:
            callbacks.append(
                early_stopping(
                    stopping_rounds=self.config.train.early_stopping_rounds,
                    first_metric_only=False,
                )
            )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric=eval_metric,
            callbacks=callbacks,
        )

        self.model = model

        # LightGBM은 evals_result_ 속성에 로그 저장
        ev = model.evals_result_
        # ev 구조: {"train": {metric_name: [...]}, "valid": {metric_name: [...]} }
        train_vals = ev["train"][eval_metric]
        valid_vals = ev["valid"][eval_metric]

        loss_tasks = [
            {
                Split.TRAIN.value: train_vals,
                Split.VALID.value: valid_vals,
            }
        ]

        # 예측
        y_tr_pred = model.predict(X_tr)   # (N,)
        y_val_pred = model.predict(X_val) # (N,)

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

        y_pred = self.model.predict(X)   # (N,)
        return y_pred

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        # 전체 테스트 데이터 기준 성능
        X_all, y_all = test_data.get_data_for_gbdt()   # (N, F), (N,)
        y_pred_all = self.model.predict(X_all)         # (N,)

        metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        # 패턴 / ratio 별 성능
        by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            ratio_dict = test_data.imputed_dict[p_v]

            for ratio in test_data.ratios:
                d = ratio_dict[ratio]

                X = d["X"]  # (N, F)
                y = d["y"]  # (N,)

                y_pred = self.model.predict(X)  # (N,)

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
        path: Path,
    ):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "lgbm_model.pkl"
        joblib.dump(self.model, model_path)

        meta = {
            "model_path": str(model_path),
        }

        self.save_meta(save_dir, meta)

    def load(
        self,
        path: Path,
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]

        model = joblib.load(model_path)

        self.model = model

        return True
