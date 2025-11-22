# src/models/xgboost_adapter.py

import numpy as np

from sklearn import tree
from xgboost import XGBClassifier
from pathlib import Path

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics


class XGBoostAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.models: list[XGBClassifier] | None = None


    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        # adapter --> 데이터 세트 반환 (numpy)
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()


        # XGBoost 설정
        eval_metric = self.config.model.eval_metric
        horizon = train_data.get_horizon()
        self.horizon = horizon
        num_class = train_data.get_num_class()
        print(f"[XGBoostAdapter] horizon: {horizon}, num_class: {num_class}")


        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[XGBoostAdapter] num_class_from_y: {num_class_from_y}")


        # 각 task(timestep) 별 모델 / history / 예측 저장
        self.models = []
        loss_tasks: list[dict] = []
        train_pred_list: list[np.ndarray] = []
        valid_pred_list: list[np.ndarray] = []


        # 시계열 예측 --> 멀티 태스크 분류로 처리
        for t in range(horizon):
            model = XGBClassifier(
                objective=self.config.model.objective,
                num_class=num_class,
                random_state=self.config.train.seed,
                eval_metric=eval_metric,
                early_stopping_rounds=self.config.train.early_stopping_rounds,
                device=self.config.train.device,
                tree_method="hist",
            )

            model.fit(
                X_tr, y_tr[:, t],
                eval_set=[(X_tr, y_tr[:, t]), (X_val, y_val[:, t])],
            )

            self.models.append(model)

            ev = model.evals_result()
            train_vals = ev["validation_0"][eval_metric]
            valid_vals = ev["validation_1"][eval_metric]

            loss_tasks.append(
                {
                    "train": train_vals,
                    "valid": valid_vals,
                }
            )

            train_pred_list.append(model.predict(X_tr))
            valid_pred_list.append(model.predict(X_val))

        y_tr_pred = np.stack(train_pred_list, axis=1)
        y_val_pred = np.stack(valid_pred_list, axis=1)


        train_metrics = compute_multitask_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_multitask_classification_metrics(y_val, y_val_pred)

        results = {
            "split": "train",
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "loss": {
                "metric_name": eval_metric,
                "tasks": loss_tasks, # {"train": [...], "valid": [...]}
            },
        }


        return results

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.models is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        preds = []
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

        by_ratio = {}

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
            "split": "test",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": by_ratio,
        }

        return results

    def save(
        self,
        path: Path
    ):
        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)


        meta = {
            "horizon": self.horizon,
        }

        save_model_dirs = []

        for t, model in enumerate(self.models):
            model_dir = save_dir / f"xgb_horizon_{t}.json"
            model.save_model(model_dir)

            save_model_dirs.append(str(model_dir))

        meta["save_model_dirs"] = save_model_dirs

        self.save_meta(save_dir, meta)



    def load(
        self,
        path: Path
    ):
        save_dir = path / "train" / "save"
        meta = self.load_meta(save_dir)

        save_model_dirs = meta["save_model_dirs"]

        self.models = []

        for model_dir in save_model_dirs:
            model = XGBClassifier()
            model.load_model(model_dir)
            self.models.append(model)

        return len(self.models) == len(save_model_dirs) == meta["horizon"]









