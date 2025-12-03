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

        # TabPFN은 학습 단계(Epoch)가 없으므로 config의 epoch 설정은 무시됩니다.
        self.model = TabPFNClassifier(
            device=self.config.train.device,
            seed=self.config.train.seed,
        )

        # TabPFN의 fit은 데이터를 저장하는 과정이므로 즉시 완료됨
        self.model.fit(X_tr, y_tr)

        # 학습 곡선이 없으므로, 최종 Loss만 계산하여 리스트로 반환 (인터페이스 호환용)
        # predict_proba를 사용하여 log_loss 계산
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

        # 최종 성능 계산
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

        # TabPFN은 sklearn estimator이므로 pickle/torch.save로 전체 객체 저장
        model_name = "tabpfn_model.pt"
        save_path = save_dir / model_name

        torch.save(self.model, save_path)

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

        model_path = meta["model_path"]

        self.model = torch.load(model_path)

        return True
