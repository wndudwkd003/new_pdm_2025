from src.params.data_model import ModelType, ModelSize, ModelParams

def get_xgb_adapter():
    from src.models.xgboost_adapter import XGBoostAdapter
    return XGBoostAdapter

def get_lightgbm_adapter():
    from src.models.lightgbm_adapter import LightGBMAdapter
    return LightGBMAdapter

# def get_rf_adapter():
#     from src.models.rf_adapter import RandomForestAdapter
#     return RandomForestAdapter

# def get_tabtransformer_adapter():
#     from src.models.tab_transformer_adapter import TabTransformerAdapter
#     return TabTransformerAdapter

# def get_fttransformer_adapter():
#     from src.models.fttransformer_adapter import FTTransformerAdapter
#     return FTTransformerAdapter

# def get_patchtst_adapter():
#     from src.models.patchtst_adapter import PatchTSTAdapter
#     return PatchTSTAdapter

# def get_mdbe_1_adapter():
#     from src.models.MDBE_1_adapter import MDBE_1_Adapter
#     return MDBE_1_Adapter

# def get_mdbe_1_balanced_adapter():
#     from src.models.MDBE_1_balanced_adapter import MDBE_1_BALANCED_Adapter
#     return MDBE_1_BALANCED_Adapter

# def get_mdbe_1_e2e_adapter():
#     from src.models.MDBE_1_e2e_adapter import MDBE_1_E2E_Adapter
#     return MDBE_1_E2E_Adapter

# def get_mdbe_2_adapter():
#     from src.models.MDBE_2_adapter import MDBE_2_Adapter
#     return MDBE_2_Adapter

# def get_mdbe_3_e2e_adapter():
#     from src.models.MDBE_3_e2e_adapter import MDBE_3_E2E_Adapter
#     return MDBE_3_E2E_Adapter



# ─────────────────────────────────────────────
# MODEL_MAP 정의
# ─────────────────────────────────────────────
MODEL_MAP = {
    # GBDT 계열
    ModelType.XGBOOST: get_xgb_adapter(),
    ModelType.LIGHTGBM: get_lightgbm_adapter(),
    # ModelType.RANDOMFOREST: get_rf_adapter(),

    # # Deep 계열
    # ModelType.TABTRANSFORMER: get_tabtransformer_adapter(),
    # ModelType.FTTRANSFORMER: get_fttransformer_adapter(),
    # ModelType.PATCHTST: get_patchtst_adapter(),

    # # Custom MDBE 계열
    # ModelType.MDBE_1: get_mdbe_1_adapter(),
    # ModelType.MDBE_1_BALANCED: get_mdbe_1_balanced_adapter(),
    # ModelType.MDBE_1_E2E: get_mdbe_1_e2e_adapter(),
    # ModelType.MDBE_2: get_mdbe_2_adapter(),
    # ModelType.MDBE_3_E2E: get_mdbe_3_e2e_adapter(),
}



# ─────────────────────────────────────────────
# MODEL_SIZE_MAP 정의
# ─────────────────────────────────────────────
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
