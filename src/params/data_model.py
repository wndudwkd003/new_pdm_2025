from enum import Enum

class DatasetType(Enum):
    MPTMS = "datasets/MPTMS/processed_data"
    CMAPSS = "datasets/c-mapss/processed_data"

class ModelType(Enum):
    XGBOOST = "xgboost"

class StageType(Enum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
