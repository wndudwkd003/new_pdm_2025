from enum import Enum

from attr import dataclass

from src.models.xgboost_adapter import XGBoostAdapter


class DatasetType(Enum):
    MPTMS = "datasets/MPTMS/processed_data"
    CMAPSS = "datasets/c-mapss/processed_data"






class ModelType(Enum):
    XGBOOST = XGBoostAdapter


class StageType(Enum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
