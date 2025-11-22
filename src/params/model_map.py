
from src.params.data_model import ModelType, ModelSize, ModelParams
from src.models.xgboost_adapter import XGBoostAdapter
from src.models.MDBE_1_adapter import MDBE_1_Adapter

MODEL_MAP = {
    ModelType.XGBOOST: XGBoostAdapter,
    ModelType.MDBE_1: MDBE_1_Adapter,
}






MODEL_SIZE_MAP = {
    ModelSize.SMALL: ModelParams(
        embed_dim=16,
        feature_hidden_dims=[64, 64, 32, 32, 16, 16],
        nhead=4,
        transformer_layers=2,
        decoder_hidden_dim=64,
        total_layer=1,
    ),

    ModelSize.MEDIUM: ModelParams(
        embed_dim=32,
        feature_hidden_dims=[128, 128, 64, 64, 32, 32],
        nhead=8,
        transformer_layers=4,
        decoder_hidden_dim=128,
        total_layer=2,
    ),

    ModelSize.LARGE: ModelParams(
        embed_dim=64,
        feature_hidden_dims=[256, 256, 128, 128, 64, 64],
        nhead=16,
        transformer_layers=6,
        decoder_hidden_dim=256,
        total_layer=3,
    ),
}
