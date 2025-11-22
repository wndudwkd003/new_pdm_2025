# src/configs/configs.py

"""

모델 설정 파일


"""

from dataclasses import dataclass, field
from turtle import st

from params.data_model import DatasetType, ModelType, StageType
from params.scenario import (
    MissingScenario, MissingPattern, StackMode, ImputeMethod
)


@dataclass
class Data:
    datasets: DatasetType
    split_ratio: float = 0.8
    skip_header: bool
    num_workers: int = 1


    missing_patterns: list[MissingPattern] = field(default_factory=list)


    missing_scenario: MissingScenario # single / multi

    target_missing_ratio: float # default, 이게 목표하는 비율임
    start_missing_ratio: float  # multi인 경우에 사용됨, 시작 비율
    step_missing_ratio: float   # 증가 비율


    stack_mode: StackMode


    impute_method: ImputeMethod

    mask_fill: float


@dataclass
class Train:
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: str = "outputs"
    seed: int
    early_stopping_rounds: int


@dataclass
class Model:
    model: ModelType = ModelType.XGBOOST

    eval_metric: str
    objective: str

    stage: StageType = StageType.FINETUNE

    other_prefix: str = ""

    save_work_dir: str = ""



@dataclass
class Config:
    data: Data
    model: Model
    train: Train



@dataclass
class DatasetMeta:
    horizon: int
    sequence: int
    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    feature_dim: int
    num_class: int

