from enum import Enum
from dataclasses import dataclass, field


class DatasetType(Enum):
    MPTMS = "datasets/MPTMS/processed_data"
    CMAPSS = "datasets/c-mapss/processed_data"
    CMAPSS_R = "datasets/c-mapss-r/processed_data"
    FORD = "datasets/fordengine/processed_data"
    PDM_AIRI_BINARY = "datasets/pdm_ai4i_bin/processed_data"
    PDM_AIRI_MULTICLASS = "datasets/pdm_ai4i_multi/processed_data"

    SPF = "datasets/SteelPlatesFaults/processed_data"
    PMDD = "datasets/PMDD/processed_data"
    MetroPT3_24 = "datasets/MetroPT3_24/processed_data"
    MetroPT3_48 = "datasets/MetroPT3_48/processed_data"
    MetroPT3_72 = "datasets/MetroPT3_72/processed_data"

    IFDD = "datasets/IFDD/processed_data"

    FlowmetersA = "datasets/FlowmetersA/processed_data"
    FlowmetersB = "datasets/FlowmetersB/processed_data"
    FlowmetersC = "datasets/FlowmetersC/processed_data"
    FlowmetersD = "datasets/FlowmetersD/processed_data"

    EGSSD = "datasets/EGSSD/processed_data"

    CBM = "datasets/CBM/processed_data"
    CBMV3 = "datasets/CBMV3/processed_data"


class ModelType(Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOMFOREST = "randomforest"

    #

    TABTRANSFORMER = "tabtransformer"
    FTTRANSFORMER = "fttransformer"
    TABNET = "tabnet"
    MLP = "mlp"
    RESMLP = "resmlp"
    DEEPTLF = "deeptlf"
    TABPFN = "tabpfn"

    NAIM = "naim"

    #
    REGAE = "regae"
    REGAE_STAGE = "regae_stage"
    REGVAE = "regvae"

    HYBRID_XGVAE = "hybrid_xgvae"
    HYBRID_XGVAE_TS_1 = "hybrid_xgvae_ts_1"
    HYBRID_XGVAE_TS_2 = "hybrid_xgvae_ts_2"

    REGVAE_XAI = "regvae_xai"
    #
    # (deprecated)
    MDBE_1 = "mdbe_1"
    MDBE_1_BALANCED = "mdbe_1_balanced"
    MDBE_1_E2E = "mdbe_1_e2e"

    MDBE_1_XGB = "mdbe_1_xgb"

    MDBE_2 = "mdbe_2"

    MDBE_3_E2E = "mdbe_3_e2e"

    SAINT = "saint"
    AGATa = "agata"


class StageType(Enum):
    NONE = ""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"


class ModelSize(Enum):
    NONE = ""
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
