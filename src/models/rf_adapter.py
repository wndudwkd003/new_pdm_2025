# src/models/random_forest_adapter.py

import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
import joblib

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics


class RandomForestAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.models: list[RandomForestClassifier] | None = None
        self.horizon: int | None = None

    def _prepare_xy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        CPU RandomForest용 전처리:
        - 그냥 numpy array로만 맞춰줌
        """
        X_out = np.asarray(X)
        y_out = np.asarray(y)
        return X_out, y_out

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        """
        - train_data.get_data_for_gbdt() -> (X, y)
          X: (N, F), y: (N, T) (T = horizon)
        - 타임스텝별로 RF 하나씩 총 horizon 개 학습
        """
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        X_tr, y_tr = self._prepare_xy(X_tr, y_tr)
        X_val, y_val = self._prepare_xy(X_val, y_val)

        horizon = train_data.get_horizon()
        self.horizon = horizon
        num_class = train_data.get_num_class()

        print(f"[RandomForestAdapter] horizon: {horizon}, num_class: {num_class}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[RandomForestAdapter] num_class_from_y: {num_class_from_y}")

        # === 하이퍼파라미터 설정 ===
        # 트리 개수 = epochs
        n_estimators = self.config.train.epochs
        # CPU 병렬 처리 개수 = data.num_workers
        n_jobs = self.config.data.num_workers
        # 클래스 불균형 대응: balanced
        class_weight = self.config.model.lgbm_class_weight

        print(
            "[RandomForestAdapter] "
            f"n_estimators={n_estimators}, class_weight={class_weight}, n_jobs={n_jobs}"
        )

        self.models = []
        loss_tasks: list[dict] = []
        train_pred_list: list[np.ndarray] = []
        valid_pred_list: list[np.ndarray] = []

        for t in range(horizon):
            print(f"[RandomForestAdapter] training horizon step {t} / {horizon - 1}")

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=self.config.train.seed,
                class_weight=class_weight,
            )

            rf.fit(X_tr, y_tr[:, t])

            self.models.append(rf)

            y_tr_pred_t = rf.predict(X_tr)
            y_val_pred_t = rf.predict(X_val)

            # 간단히 accuracy를 loss 로그 형태로 저장
            train_acc = float(np.mean(y_tr[:, t] == y_tr_pred_t))
            valid_acc = float(np.mean(y_val[:, t] == y_val_pred_t))

            loss_tasks.append(
                {
                    Split.TRAIN.value: [train_acc],
                    Split.VALID.value: [valid_acc],
                }
            )

            train_pred_list.append(y_tr_pred_t)
            valid_pred_list.append(y_val_pred_t)

        y_tr_pred = np.stack(train_pred_list, axis=1)
        y_val_pred = np.stack(valid_pred_list, axis=1)

        train_metrics = compute_multitask_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_multitask_classification_metrics(y_val, y_val_pred)

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": "accuracy",
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

        X_in = np.asarray(X)

        preds: list[np.ndarray] = []
        for rf in self.models:
            y_hat = rf.predict(X_in)
            preds.append(y_hat)

        y_pred = np.stack(preds, axis=1)
        return y_pred

    def test(
        self,
        test_data: Datasets,
    ):
        X_all, y_all = test_data.get_data_for_gbdt()
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)

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
                X_flat = np.asarray(X_flat)
                y = np.asarray(y)

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
        LightGBMAdapter / XGBoostAdapter와 동일한 구조:
        - path / "save" 하위에 모델들과 meta 저장
        """
        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "horizon": self.horizon,
        }

        save_model_dirs: list[str] = []

        for t, rf in enumerate(self.models):
            model_path = save_dir / f"rf_horizon_{t}.pkl"
            joblib.dump(rf, model_path)
            save_model_dirs.append(str(model_path))

            meta["save_model_dirs"] = save_model_dirs

        self.save_meta(save_dir, meta)

    def load(
        self,
        path: Path,
    ):
        """
        LightGBMAdapter.load와 동일한 규칙:
        - 학습 시 save(path / Split.TRAIN.value)를 호출했다고 가정
        - 로드시에는 work_dir / "train" / "save"에서 읽기
        """
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        save_model_dirs = meta["save_model_dirs"]

        self.models = []

        for model_dir in save_model_dirs:
            rf = joblib.load(model_dir)
            self.models.append(rf)

        self.horizon = meta["horizon"]

        return len(self.models) == len(save_model_dirs) == self.horizon
