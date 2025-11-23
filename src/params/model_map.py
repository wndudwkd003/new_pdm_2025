
from src.params.data_model import ModelType, ModelSize, ModelParams
from src.models.xgboost_adapter import XGBoostAdapter
from src.models.lightgbm_adapter import LightGBMAdapter
from src.models.rf_adapter import RandomForestAdapter
from src.models.MDBE_1_adapter import MDBE_1_Adapter
from src.models.MDBE_1_balanced_adapter import MDBE_1_BALANCED_Adapter


MODEL_MAP = {
    # GBDT
    ModelType.XGBOOST: XGBoostAdapter,
    ModelType.LIGHTGBM: LightGBMAdapter,
    ModelType.RANDOMFOREST: RandomForestAdapter,

    ############################

    # Deep


    ############################

    # My

    ModelType.MDBE_1: MDBE_1_Adapter,
    ModelType.MDBE_1_BALANCED: MDBE_1_BALANCED_Adapter,
}






MODEL_SIZE_MAP = {
    ModelSize.SMALL: ModelParams(
        embed_dim=16,
        feature_hidden_dims=[32, 32, 16, 8],
        nhead=4,
        transformer_layers=2,
        decoder_hidden_dim=8,
        total_layer=1,
    ),

    ModelSize.MEDIUM: ModelParams(
        embed_dim=16,
        feature_hidden_dims=[64, 64, 32, 32, 16, 16],
        nhead=4,
        transformer_layers=4,
        decoder_hidden_dim=16,
        total_layer=2,
    ),

    ModelSize.LARGE: ModelParams(
        embed_dim=16,
        feature_hidden_dims=[512, 512, 128, 128, 32, 32],
        nhead=16,
        transformer_layers=6,
        decoder_hidden_dim=32,
        total_layer=3,
    ),
}
