from src.params.data_model import ModelType

from src.models.xgboost_adapter import XGBoostAdapter

MODEL_MAP = {
    ModelType.XGBOOST: XGBoostAdapter,
}
