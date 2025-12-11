# src/models/deeptlf_adapter.py

import json
import numpy as np
from pathlib import Path

import torch
import xgboost as xgb

from deeptlf import DeepTFL, TreeDrivenEncoder
from deeptlf.deeptlf import NeuralNet

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split


class DeepTLFAdapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)

        self.model: DeepTFL | None = None
        self.input_dim: int | None = None        # DeepTFL NN 입력 차원
        self.num_class: int | None = None

        self.checkpoint_name = "checkpoint.pt"


    # ======================================================
    # 모델 생성
    # ======================================================
    def _get_model(self) -> DeepTFL:
        return DeepTFL(
            task="class",
            n_epoch=self.config.train.epochs,
            batch_size=self.config.train.batch_size,
            checkpoint_name=self.checkpoint_name,
        )


    # ======================================================
    # FIT
    # ======================================================
    def fit(self, train_data: Datasets, valid_data: Datasets):

        X_tr, y_tr = train_data.get_data_for_gbdt()
        X_val, y_val = valid_data.get_data_for_gbdt()

        # 데이터셋 input_dim 아님! DeepTFL 내부에서 NN 입력 차원 결정됨.
        self.num_class = train_data.meta.num_class

        self.model = self._get_model()
        self.model.fit(X_tr, y_tr, X_val, y_val)

        # ⬅ DeepTFL 내부 encoder.transform 결과 input_dim
        self.input_dim = self.model.input_shape

        y_tr_pred = self.model.predict(X_tr)
        y_val_pred = self.model.predict(X_val)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        loss_info = {
            "metric_name": "cross_entropy",
            "tasks": [
                {
                    Split.TRAIN.value: self.model.train_losses,
                    Split.VALID.value: self.model.val_losses,
                }
            ],
        }

        return {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": loss_info,
        }


    # ======================================================
    # ARRAY PREDICT
    # ======================================================
    def predict_array(self, X: np.ndarray):
        return self.model.predict(X)


    # ======================================================
    # predict(loader)
    # ======================================================
    def predict(self, loader):

        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        for batch in loader:
            x = batch["x"]
            y = batch["y"]
            pattern_idx = batch["pattern_idx"]
            ratio_idx = batch["ratio_idx"]

            preds = self.predict_array(x.cpu().numpy())
            batch_loss = 0.0  # DeepTFL은 loss 제공 없음

            total_loss += batch_loss
            num_batches += 1

            all_preds.append(torch.tensor(preds))
            all_labels.append(y.cpu())
            all_pattern_idx.append(pattern_idx.cpu())
            all_ratio_idx.append(ratio_idx.cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all


    # ======================================================
    # TEST
    # ======================================================
    def test(self, test_data: Datasets):

        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        X_all, y_all = test_data.get_data_for_gbdt()
        y_pred_all = self.predict_array(X_all)

        metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        metrics_by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p = pattern.value
            metrics_by_ratio[p] = {}

            for ratio, d in test_data.imputed_dict[p].items():
                X = d["X"]
                y = d["y"]
                pred = self.predict_array(X)
                metrics_by_ratio[p][ratio] = compute_classification_metrics(y, pred)

        return {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
        }


    # ======================================================
    # SAVE
    # ======================================================
    def save(self, path: Path):

        if self.model is None:
            raise ValueError("No model to save.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) XGBoost 저장
        xgb_path = save_dir / "xgb_model.json"
        self.model.xgb_model.save_model(str(xgb_path))

        # 2) Encoder 트리 저장
        trees = self.model.xgb_model.get_booster().get_dump(with_stats=False)
        encoder_path = save_dir / "encoder_trees.json"
        with open(encoder_path, "w") as f:
            json.dump(trees, f)

        # 3) NeuralNet checkpoint 저장
        nn_src = Path(self.checkpoint_name)
        nn_path = save_dir / "checkpoint.pt"
        if nn_src.exists():
            nn_path.write_bytes(nn_src.read_bytes())

        # ⭐ DeepTFL NN의 실제 input_dim 저장 (가장 중요)
        meta = {
            "input_dim": self.model.input_shape,
            "num_class": self.num_class,
            "xgb_model": str(xgb_path),
            "encoder_trees": str(encoder_path),
            "nn_checkpoint": str(nn_path),
        }

        self.save_meta(save_dir, meta)
        return save_dir


    # ======================================================
    # LOAD
    # ======================================================
    def load(self, path: Path):

        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        # ⬅️ 저장한 DeepTFL NN input_dim을 그대로 사용
        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])

        # ------------------------------
        # 1) 모델 생성
        # ------------------------------
        self.model = self._get_model()

        # ------------------------------
        # 2) XGBoost 복원
        # ------------------------------
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(meta["xgb_model"])

        # ------------------------------
        # 3) Encoder 복원
        # ------------------------------
        with open(meta["encoder_trees"], "r") as f:
            trees = json.load(f)

        encoder = TreeDrivenEncoder()
        encoder.fit(trees)

        # ------------------------------
        # 4) NeuralNet 복원
        # ------------------------------
        checkpoint = meta["nn_checkpoint"]

        nn_model = NeuralNet(
            input_dim=self.input_dim,
            hidden_dim=256,
            n_layers=4,
            num_classes=self.num_class,
            drop=0.23,
        ).to("cuda")

        state = torch.load(checkpoint, map_location="cuda")
        nn_model.load_state_dict(state)

        # ------------------------------
        # 5) DeepTFL 결합
        # ------------------------------
        self.model.xgb_model = xgb_model
        self.model.TDE_encoder = encoder
        self.model.nn_model = nn_model

        return True
