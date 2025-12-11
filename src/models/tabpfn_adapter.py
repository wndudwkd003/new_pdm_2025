# src/models/tabpfn_adapter.py

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss
from tabpfn import TabPFNClassifier

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split

class TabPFNAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.model: TabPFNClassifier | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        if len(X_tr) > 50000:
            indices = np.random.choice(len(X_tr), 50000, replace=False)
            X_tr = X_tr[indices]
            y_tr = y_tr[indices]

        self.model = TabPFNClassifier(
            device=self.config.train.device,
            random_state=self.config.train.seed,
            n_estimators=8,
        )

        self.model.fit(X_tr, y_tr)

        tr_proba = self.model.predict_proba(X_tr)
        val_proba = self.model.predict_proba(X_val)

        train_loss = log_loss(y_tr, tr_proba, labels=self.model.classes_)
        valid_loss = log_loss(y_val, val_proba, labels=self.model.classes_)

        loss_tasks = [
            {
                Split.TRAIN.value: [train_loss],
                Split.VALID.value: [valid_loss],
            }
        ]

        y_tr_pred = self.model.predict(X_tr)
        y_val_pred = self.model.predict(X_val)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": "logloss",
                "tasks": loss_tasks,
            },
        }

        return results

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted or loaded.")

        return self.model.predict(X)

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("Model is not fitted or loaded.")

        X_all, y_all = test_data.get_data_for_gbdt()
        y_pred_all = self.predict(X_all)

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
            raise ValueError("No model to save.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        # TabPFN이 요구하는 확장자: .tabpfn_fit
        model_name = "tabpfn_model.tabpfn_fit"
        save_path = save_dir / model_name

        # 이 함수 내부에서 save_fitted_tabpfn_model을 호출함
        self.model.save_fit_state(str(save_path))

        meta = {
            "model_path": str(save_path),
        }

        self.save_meta(save_dir, meta)
        return save_path


    def load(
        self,
        path: Path
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = Path(meta["model_path"])

        if not model_path.exists():
            print(f"[TabPFNAdapter] fitted model not found: {model_path}")
            return False

        self.model = TabPFNClassifier.load_from_fit_state(
            str(model_path),
            device=self.config.train.device,
        )

        return True

