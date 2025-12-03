# src/models/tabnet_adapter.py

import torch
import numpy as np
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split

class TabNetAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)
        self.model: TabNetClassifier | None = None

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        # 1. Unsupervised Pretraining
        self.pretrainer = TabNetPretrainer(
            device_name=self.config.train.device,
            seed=self.config.train.seed,
        )

        self.pretrainer.fit(
            X_train=X_tr,
            eval_set=[X_val],
            max_epochs=self.config.train.epochs,
            patience=self.config.train.early_stopping_rounds,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.data.num_workers,
            drop_last=False,
            pretraining_ratio=0.8
        )

        # 2. Supervised Finetuning
        self.model = TabNetClassifier(
            device_name=self.config.train.device,
            seed=self.config.train.seed,
        )

        self.model.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_name=[Split.TRAIN.value, Split.VALID.value],
            eval_metric=['logloss'],
            max_epochs=self.config.train.epochs,
            patience=self.config.train.early_stopping_rounds,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.data.num_workers,
            drop_last=False,
            from_unsupervised=self.pretrainer
        )

        history = self.model.history
        train_losses = history['loss']
        valid_losses = history[f'{Split.VALID.value}_logloss']

        loss_tasks = [
            {
                Split.TRAIN.value: train_losses,
                Split.VALID.value: valid_losses,
            }
        ]

        y_tr_pred = self.predict(X_tr)
        y_val_pred = self.predict(X_val)

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
            raise ValueError("Model is not trained or loaded.")

        return self.model.predict(X)

    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

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

        model_name = "tabnet_model"
        saved_filepath = self.model.save_model(str(save_dir / model_name))

        meta = {
            "model_path": str(saved_filepath),
        }

        self.save_meta(save_dir, meta)
        return saved_filepath

    def load(
        self,
        path: Path
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]

        self.model = TabNetClassifier()
        self.model.load_model(model_path)

        return True
