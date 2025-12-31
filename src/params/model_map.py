from src.params.data_model import ModelType, ModelSize, ModelParams


def get_xgb_adapter():
    from src.models.xgboost_adapter import XGBoostAdapter

    return XGBoostAdapter


def get_lightgbm_adapter():
    from src.models.lightgbm_adapter import LightGBMAdapter

    return LightGBMAdapter


def get_rf_adapter():
    from src.models.rf_adapter import RandomForestAdapter

    return RandomForestAdapter


def get_tabtransformer_adapter():
    from src.models.tab_transformer_adapter import TabTransformerAdapter

    return TabTransformerAdapter


def get_fttransformer_adapter():
    from src.models.fttransformer_adapter import FTTransformerAdapter

    return FTTransformerAdapter


def get_tabnet_adapter():
    from src.models.tabnet_adapter import TabNetAdapter

    return TabNetAdapter


def get_deeptlf_adapter():
    from src.models.deeptlf_adapter import DeepTLFAdapter

    return DeepTLFAdapter


def get_tabfpn_adapter():
    from src.models.tabpfn_adapter import TabPFNAdapter

    return TabPFNAdapter


def get_mlp_adapter():
    from src.models.mlp_adapter import MLPAdapter

    return MLPAdapter


def get_resmlp_adapter():
    from src.models.resmlp_adapter import ResMLPAdapter

    return ResMLPAdapter


def get_regae_adapter():
    from src.models.ReGAE_adapter import ReGAEAdapter

    return ReGAEAdapter


def get_regae_stage_adapter():
    from src.models.ReGAE_stage_adapter import ReGAEAdapter

    return ReGAEAdapter


def get_regvae_adapter():
    from src.models.ReGVAE_adapter import ReGVAEAdapter

    return ReGVAEAdapter


def get_naim_adapter():
    from src.models.naim_adapter import NAIMAdapter

    return NAIMAdapter


def get_hybrid_xgvae_adapter():
    from src.models.hybrid_xgvae_adapter import HybridXGVAEAdapter

    return HybridXGVAEAdapter


def get_hybrid_xgvae_ts_1_adapter():
    from src.models.hybrid_xgvae_ts_1_adapter import HybridXGVAE_TS_1_Adapter

    return HybridXGVAE_TS_1_Adapter


def get_hybrid_xgvae_ts_2_adapter():
    from src.models.hybrid_xgvae_ts_2_adapter import HybridXGVAE_TS_2_Adapter

    return HybridXGVAE_TS_2_Adapter


# def get_mdbe_1_adapter():
#     from src.models.MDBE_1_adapter import MDBE_1_Adapter
#     return MDBE_1_Adapter


def get_regvae_xai_adapter():
    from src.models.ReGVAE_xai_adapter import ReGVAEAdapterXAI

    return ReGVAEAdapterXAI


def get_saint_adapter():
    from src.models.saint_adapter import SAINTAdapter

    return SAINTAdapter


def get_agata_adapter():
    from src.models.agata_adapter import AGATaAdapter

    return AGATaAdapter


# ─────────────────────────────────────────────
# MODEL_MAP 정의
# ─────────────────────────────────────────────
MODEL_MAP = {
    # GBDT 계열
    ModelType.XGBOOST: get_xgb_adapter(),
    ModelType.LIGHTGBM: get_lightgbm_adapter(),
    ModelType.RANDOMFOREST: get_rf_adapter(),
    # # Deep 계열
    ModelType.TABTRANSFORMER: get_tabtransformer_adapter(),
    ModelType.FTTRANSFORMER: get_fttransformer_adapter(),
    ModelType.TABNET: get_tabnet_adapter(),
    ModelType.DEEPTLF: get_deeptlf_adapter(),
    ModelType.TABPFN: get_tabfpn_adapter(),
    ModelType.MLP: get_mlp_adapter(),
    ModelType.RESMLP: get_resmlp_adapter(),
    ModelType.NAIM: get_naim_adapter(),
    # ReGAE
    ModelType.REGAE: get_regae_adapter(),
    ModelType.REGVAE: get_regvae_adapter(),
    ModelType.REGAE_STAGE: get_regae_stage_adapter(),
    # # Custom MDBE 계열
    # ModelType.MDBE_1: get_mdbe_1_adapter(),
    # ModelType.MDBE_1_E2E: get_mdbe_1_e2e_adapter(),
    ModelType.HYBRID_XGVAE: get_hybrid_xgvae_adapter(),
    ModelType.HYBRID_XGVAE_TS_1: get_hybrid_xgvae_ts_1_adapter(),
    ModelType.HYBRID_XGVAE_TS_2: get_hybrid_xgvae_ts_2_adapter(),
    ModelType.REGVAE_XAI: get_regvae_xai_adapter(),
    ModelType.SAINT: get_saint_adapter(),
    ModelType.AGATa: get_agata_adapter(),
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
