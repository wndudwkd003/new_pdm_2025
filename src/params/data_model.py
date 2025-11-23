from enum import Enum
from dataclasses import dataclass, field

class DatasetType(Enum):
    MPTMS = "datasets/MPTMS/processed_data"
    CMAPSS = "datasets/c-mapss/processed_data"

class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOMFOREST = "randomforest"

    #
    MDBE_1 = "mdbe_1"
    MDBE_1_BALANCED = "mdbe_1_balanced"



    MDBE_1_XGB = "mdbe_1_xgb"

class StageType(Enum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"

class ModelSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class Split(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass
class ModelParams:
    embed_dim: int = 0
    feature_hidden_dims: list[int] = field(default_factory=list)
    nhead: int = 0
    transformer_layers: int = 0
    decoder_hidden_dim: int = 0
    total_layer: int = 0
