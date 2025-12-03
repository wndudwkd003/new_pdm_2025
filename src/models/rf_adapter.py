# src/models/random_forest_adapter.py

import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

from src.configs.configs import Config
from src.params.data_model import Split
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics


class RandomForestAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        # 단일 분류 모델
        self.model: RandomForestClassifier | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        """
        - train_data.get_data_for_gbdt() -> (X, y)
          X: (N, F), y: (N,)
        - warm_start 없이 한 번에 RandomForest 학습
        - loss history는 train / valid accuracy 한 점씩만 기록
        """
        # X: (N, F), y: (N,)
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        X_tr = np.asarray(X_tr)
        y_tr = np.asarray(y_tr)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        num_class = train_data.get_num_class()
        print(f"[RandomForestAdapter] num_class (from meta): {num_class}")

        num_class_from_y = int(max(y_tr.max(), y_val.max()) + 1)
        print(f"[RandomForestAdapter] num_class_from_y: {num_class_from_y}")

        # === 하이퍼파라미터 설정 ===
        # 여기서 epochs는 "학습 반복"이 아니라 "트리 개수"로 해석됨
        n_estimators = int(self.config.train.epochs)
        if n_estimators <= 0:
            raise ValueError("RandomForestAdapter: config.train.epochs 는 1 이상이어야 합니다.")

        n_jobs = self.config.data.num_workers
        class_weight_cfg = self.config.model.lgbm_class_weight  # None, "balanced", dict 등

        # "balanced" / "balanced_subsample"이면 직접 weight dict 계산
        if class_weight_cfg in ("balanced", "balanced_subsample"):
            classes = np.unique(y_tr)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_tr,
            )
            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        else:
            # None 이나 dict 는 그대로 사용
            class_weight = class_weight_cfg

        print(
            "[RandomForestAdapter] "
            f"n_estimators={n_estimators}, class_weight={class_weight}, n_jobs={n_jobs}"
        )

        # warm_start 사용하지 않음
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=self.config.train.seed,
            class_weight=class_weight,
        )

        rf.fit(X_tr, y_tr)
        self.model = rf

        # 최종 모델 기준 성능
        y_tr_pred = rf.predict(X_tr)   # (N,)
        y_val_pred = rf.predict(X_val) # (N,)

        train_acc = float(np.mean(y_tr == y_tr_pred))
        valid_acc = float(np.mean(y_val == y_val_pred))

        # history는 한 점짜리 curve
        loss_tasks = [
            {
                Split.TRAIN.value: [train_acc],
                Split.VALID.value: [valid_acc],
            }
        ]

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

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
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        X_in = np.asarray(X)
        y_pred = self.model.predict(X_in)  # (N,)
        return y_pred

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        # 전체 테스트 데이터 기준 성능
        X_all, y_all = test_data.get_data_for_gbdt()   # (N, F), (N,)
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)

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

                X = np.asarray(X)
                y = np.asarray(y)

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
        """
        XGBoostAdapter 와 동일한 구조:
        - path / "save" 하위에 모델과 meta 저장
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "rf_model.pkl"
        joblib.dump(self.model, model_path)

        meta = {
            "model_path": str(model_path),
        }

        self.save_meta(save_dir, meta)

    def load(
        self,
        path: Path,
    ):
        """
        - 학습 시 save(path / Split.TRAIN.value)를 호출했다고 가정
        - 로드시에는 work_dir / "train" / "save"에서 읽기
        """
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]

        rf = joblib.load(model_path)
        self.model = rf

        return True
