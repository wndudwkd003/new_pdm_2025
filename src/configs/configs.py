# src/configs/configs.py

"""

모델 설정 파일


"""

from dataclasses import dataclass, field
from turtle import st

from params.data_model import DatasetType, ModelType
from params.scenario import (
    MissingScenario, MissingPattern, StackMode, ImputeMethod
)


@dataclass
class Data:
    datasets: DatasetType

    missing_pattern: MissingPattern


    missing_scenario: MissingScenario # single / multi

    target_missing_ratio: float # default, 이게 목표하는 비율임
    start_missing_ratio: float  # multi인 경우에 사용됨, 시작 비율
    step_missing_ratio: float   # 증가 비율


    stack_mode: StackMode


    impute_method: ImputeMethod




@dataclass
class Train:
    epochs: int
    batch_size: int
    lr: float
    device: str
    output_dir: str
    seed: int
    early_stopping_rounds: int


@dataclass
class Model:
    model: ModelType

    eval_metric: str
    objective: str



@dataclass
class Config:
    data: Data
    model: Model
    train: Train



@dataclass
class DatasetMeta:
    num_class: int
    horizon: int
