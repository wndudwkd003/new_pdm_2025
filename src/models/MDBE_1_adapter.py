# src/models/MDBE_1_adapter.py

import numpy as np

from sklearn import tree
from pathlib import Path

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.core.models.MDBE_1 import HybridDoubleBranchEncoder
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split


class MDBE_1_Adapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: HybridDoubleBranchEncoder | None = None


    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):

        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.model = self._get_model(train_data.meta.feature_dim, train_data.meta.num_class)

        optimizer, scheduler = self.get_deeplearning_utils()

        best_valid_loss = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs = []
        train_loss = []
        valid_loss = []

        for epoch in range(self.config.train.epochs):
            train_loss = self.run_epoch(Split.TRAIN)

            valid_loss = self.run_epoch(Split.VALID)









    def _get_model(
        self,
        input_dim: int,
        num_class: int,
    ):
        return HybridDoubleBranchEncoder(
            input_dim=input_dim,
            embed_dim=self.config.params.embed_dim,
            feature_hidden_dims=self.config.params.feature_hidden_dims,
            num_class=num_class,
            nhead=self.config.params.nhead,
            transformer_layers=self.config.params.transformer_layers,
        )


