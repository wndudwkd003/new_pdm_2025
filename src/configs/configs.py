# src/configs/configs.py

"""

모델 설정 파일


"""

from dataclasses import dataclass, field

from src.params.data_model import DatasetType, ModelType, StageType, ModelSize, ModelParams
from src.params.scenario import MissingScenario, MissingPattern, ImputeMethod # , StackMode


@dataclass
class StaticMeta:
    backward: int = 10
    forward: int = 30
    interval: int = 3




@dataclass
class Data:
    datasets: DatasetType = DatasetType.MPTMS
    skip_header: bool = True
    data_load_workers: int = 10
    num_workers: int = 10
    missing_patterns: list[MissingPattern] = field(default_factory=lambda: [MissingPattern.MCAR])
    missing_scenario: MissingScenario = MissingScenario.MULTI # SINGLE, MULTI
    target_missing_ratio: float = 0.5
    start_missing_ratio: float  = 0.0
    step_missing_ratio: float   = 0.1
    impute_method: ImputeMethod = ImputeMethod.ZERO
    # stack_mode: StackMode

@dataclass
class Train:
    epochs: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cuda"
    output_dir: str = "outputs"
    seed: int = 42
    early_stopping_rounds: int = 15
    lr_min: float = 1e-5
    tree_method: str = "hist"
    temperature: float = 0.7

@dataclass
class Model:
    model: ModelType = ModelType.XGBOOST
    stage: StageType = StageType.NONE
    other_prefix: str = ""
    save_work_dir: str = "outputs/2025-11-24_04-29-49_xgboost_0.0_to_0.5_0.1_step_none_none_multi_mcar_zero_mptms_30_10_3s" # ""
    model_size: ModelSize = ModelSize.NONE

    # xgboost
    eval_metric: str = "mlogloss"
    objective: str = "multi:softprob"


    # light gbm
    lgbm_objective: str = "multiclass"
    lgbm_metric: str = "multi_logloss"
    lgbm_class_weight: str = "balanced"




@dataclass
class Config:
    data: Data  = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    train: Train = field(default_factory=Train)
    params: ModelParams = field(default_factory=ModelParams)

@dataclass
class DatasetMeta:
    horizon: int = 0
    sequence: int = 0
    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    feature_dim: int = 0
    num_class: int = 0


