# src/models/lightgbm_adapter.py

import numpy as np
from pathlib import Path

from lightgbm import LGBMClassifier, early_stopping
import joblib

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics


class LightGBMAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.models: list[LGBMClassifier] | None = None
        self.horizon: int | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        # adapter --> 데이터 세트 반환 (numpy)
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        # LightGBM 전용 metric / objective 사용
        eval_metric = self.config.model.lgbm_metric
        horizon = train_data.get_horizon()
        self.horizon = horizon
        num_class = train_data.get_num_class()

        print(f"[LightGBMAdapter] horizon: {horizon}, num_class: {num_class}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[LightGBMAdapter] num_class_from_y: {num_class_from_y}")


        device_type = self.config.train.device
        print(f"[LightGBMAdapter] device_type: {device_type}")

        self.models = []
        loss_tasks: list[dict] = []
        train_pred_list: list[np.ndarray] = []
        valid_pred_list: list[np.ndarray] = []

        for t in range(horizon):
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
                y_tr[:, t],
                eval_set=[(X_tr, y_tr[:, t]), (X_val, y_val[:, t])],
                eval_names=["train", "valid"],
                eval_metric=eval_metric,
                callbacks=callbacks,
            )

            self.models.append(model)

            # LightGBM은 evals_result_ 속성에 로그 저장
            ev = model.evals_result_
            # ev 구조: {"train": {metric_name: [...]}, "valid": {metric_name: [...]} }
            train_vals = ev["train"][eval_metric]
            valid_vals = ev["valid"][eval_metric]

            loss_tasks.append(
                {
                    Split.TRAIN.value: train_vals,
                    Split.VALID.value: valid_vals,
                }
            )

            train_pred_list.append(model.predict(X_tr))
            valid_pred_list.append(model.predict(X_val))

        y_tr_pred = np.stack(train_pred_list, axis=1)
        y_val_pred = np.stack(valid_pred_list, axis=1)

        train_metrics = compute_multitask_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_multitask_classification_metrics(y_val, y_val_pred)

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
        if self.models is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        preds: list[np.ndarray] = []
        for model in self.models:
            y_hat = model.predict(X)
            preds.append(y_hat)

        y_pred = np.stack(preds, axis=1)
        return y_pred

    def test(
        self,
        test_data: Datasets,
    ):
        X_all, y_all = test_data.get_data_for_gbdt()
        y_pred_all = self.predict(X_all)

        metrics_overall = compute_multitask_classification_metrics(y_all, y_pred_all)

        by_ratio: dict = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            ratio_dict = test_data.imputed_dict[p_v]

            for ratio in test_data.ratios:
                d = ratio_dict[ratio]

                X = d["X"]
                y = d["y"]

                X_flat = test_data.get_flat_2d(X)
                y_pred = self.predict(X_flat)

                m = compute_multitask_classification_metrics(y, y_pred)
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
        """
        XGBoostAdapter와 구조를 맞추기 위해:
        - path / "save" 하위에 모델과 meta 저장
        - meta["save_model_dirs"]에 각 타임스텝 모델 경로 기록
        """
        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "horizon": self.horizon,
        }

        save_model_dirs: list[str] = []

        for t, model in enumerate(self.models):
            model_path = save_dir / f"lgbm_horizon_{t}.pkl"
            joblib.dump(model, model_path)
            save_model_dirs.append(str(model_path))

        meta["save_model_dirs"] = save_model_dirs

        self.save_meta(save_dir, meta)

    def load(
        self,
        path: Path,
    ):
        """
        XGBoostAdapter.load와 동일한 규칙:
        - 학습 시 save(path / Split.TRAIN.value)를 호출했다고 가정
        - 로드시에는 work_dir을 받아서 work_dir / "train" / "save"에서 읽기
        """
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        save_model_dirs = meta["save_model_dirs"]

        self.models = []

        for model_dir in save_model_dirs:
            model = joblib.load(model_dir)
            self.models.append(model)

        self.horizon = meta["horizon"]

        return len(self.models) == len(save_model_dirs) == self.horizon
