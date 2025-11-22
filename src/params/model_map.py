from dataclasses import dataclass, field
from re import M
from src.params.data_model import ModelType, ModelSize
from src.models.xgboost_adapter import XGBoostAdapter

MODEL_MAP = {
    ModelType.XGBOOST: XGBoostAdapter,

}


@dataclass
class ModelParams:
    embed_dim: int = 0
    encoder_hidden_dims: list[int] = field(default_factory=list)
    nhead: int = 0
    transformer_layers: int = 0
    decoder_hidden_dim: int = 0
    total_layer: int = 0



MODEL_SIZE_MAP = {
    ModelSize.SMALL: ModelParams(
        embed_dim=16,
        encoder_hidden_dims=[64, 64, 32, 32, 16, 16],
        nhead=4,
        transformer_layers=2,
        decoder_hidden_dim=64,
        total_layer=1,
    ),

    ModelSize.MEDIUM: ModelParams(
        embed_dim=32,
        encoder_hidden_dims=[128, 128, 64, 64, 32, 32],
        nhead=8,
        transformer_layers=4,
        decoder_hidden_dim=128,
        total_layer=2,
    ),

    ModelSize.LARGE: ModelParams(
        embed_dim=64,
        encoder_hidden_dims=[256, 256, 128, 128, 64, 64],
        nhead=16,
        transformer_layers=6,
        decoder_hidden_dim=256,
        total_layer=3,
    ),
}
