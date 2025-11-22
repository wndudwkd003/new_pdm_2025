# src/configs/configs.py

"""

모델 설정 파일


"""

from dataclasses import dataclass, field

from src.params.data_model import DatasetType, ModelType, StageType, ModelSize
from src.params.scenario import MissingScenario, MissingPattern, ImputeMethod # , StackMode


@dataclass
class Data:
    datasets: DatasetType = DatasetType.MPTMS
    split_ratio: float = 0.8
    skip_header: bool = True
    num_workers: int = 1
    missing_patterns: list[MissingPattern] = field(default_factory=lambda: [MissingPattern.MCAR])
    missing_scenario: MissingScenario = MissingScenario.MULTI # SINGLE, MULTI
    target_missing_ratio: float = 0.5
    start_missing_ratio: float  = 0.0
    step_missing_ratio: float   = 0.1
    impute_method: ImputeMethod = ImputeMethod.MEAN
    # stack_mode: StackMode

@dataclass
class Train:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda"
    output_dir: str = "outputs"
    seed: int = 42
    early_stopping_rounds: int = 15

@dataclass
class Model:
    model: ModelType = ModelType.XGBOOST
    eval_metric: str = "mlogloss"
    objective: str = "multi:softprob"
    stage: StageType = StageType.FINETUNE
    other_prefix: str = ""
    save_work_dir: str = "outputs/2025-11-22_03-47-58_xgboost-0.0_to_0.5_0.1step-finetune"
    model_size: ModelSize = ModelSize.SMALL

@dataclass
class Config:
    data: Data  = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    train: Train = field(default_factory=Train)

@dataclass
class DatasetMeta:
    horizon: int = 0
    sequence: int = 0
    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    feature_dim: int = 0
    num_class: int = 0


