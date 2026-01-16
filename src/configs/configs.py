# src/configs/configs.py

"""

모델 설정 파일


"""

from dataclasses import dataclass, field
from typing import Literal, List

from lark import Tree
from regex import F

from src.params.data_model import (
    DatasetType,
    ModelType,
    StageType,
    ModelSize,
    ModelParams,
)
from src.params.scenario import (
    MissingScenario,
    MissingPattern,
    ImputeMethod,
)  # , StackMode


@dataclass
class Data:
    datasets: DatasetType = DatasetType.SPF
    skip_header: bool = True
    data_load_workers: int = 10
    num_workers: int = 5
    missing_patterns: list[MissingPattern] = field(
        default_factory=lambda: [MissingPattern.MCAR]
    )

    missing_ratio: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 0.8]
    )

    # ZERO, MEAN, MEDIAN, MICE, KNN, GAIN
    impute_method: ImputeMethod = ImputeMethod.MEDIAN
    use_on_batch: bool = True
    different_mode: bool = False
    # stack_mode: StackMode


@dataclass
class Train:
    tree_est: int = 8000  # default 100(1e-1) 1000(1e-2) 8000(1e-3)
    epochs: int = 1000  # 200
    batch_size: int = 512  # 8
    lr: float = 1e-3  # default 1e-1 1e-2 1e-3
    lr_stage_1: float = 1e-3
    lr_stage_2: float = 1e-3
    device: str = "cuda"
    output_dir: str = "outputs"
    seed: int = -1
    early_stopping_rounds: int = 50
    lr_min: float = 1e-5
    lr_min_stage_1: float = 1e-5
    lr_min_stage_2: float = 1e-5
    tree_method: str = "hist"


@dataclass
class Model:
    model: ModelType = ModelType.LIGHTGBM
    stage: StageType = StageType.NONE
    other_prefix: str = ""
    save_work_dir: str = ""
    model_size: ModelSize = ModelSize.NONE
    cross_ent_metric: str = "cross_entropy"

    hybrid_mode: str = "xgb"  # xgb | lightgbm
    use_regvae_saint_aug: bool = False

    use_my_loss: bool = True
    use_stage_1_ce: bool = False

    # xgboost
    eval_metric: str = "auto"
    objective: str = "auto"

    # light gbm
    # light gbm
    lgbm_objective: str = "auto"
    lgbm_metric: str = "auto"
    lgbm_class_weight: str = "balanced"

    lambda_kd: float = 1.0

    lambda_stage1_ce: float = 1.0

    # -------------------------
    # unified deep-model weights
    # -------------------------
    lambda_cls: float = 1.0
    lambda_recon: float = 1.0
    lambda_view: float = 1.0
    lambda_rscore: float = 1.0
    lambda_kl: float = 1.0

    # ReGVAE/ReGAE 계열에서 쓰는 추가 항목(이름 통일)
    lambda_contrast: float = 1.0
    lambda_mu: float = 1.0
    lambda_prior: float = 0.1

    mu_align: str = "mse"  # mse, cosine
    tau: float = 0.5  # temperature for contrastive loss

    # -------------------------
    # stage2 retrieval params
    # -------------------------
    retrieval_k: int = 4
    retrieval_tau: float = 0.5
    retrieval_chunk: int = 200000

    # -------------------------
    # embedding visualization params
    # -------------------------
    vis_max_points: int = 20000
    vis_pca_dim: int = 50
    vis_perplexity: float = 30.0
    vis_seed: int = 42

    # -------------------------
    # NAIM params
    # -------------------------
    naim_d_token: int = 198
    naim_embedder_initialization: str = "uniform"  # uniform, normal
    naim_bias: bool = True

    # attention mask type: (원 논문 구현 기준)
    # 0: 기본, 1: 변형, 2: mask + mask.T (식(10) 형태에 해당)
    naim_mask_type: int = 2
    naim_missing_value: str = "-inf"  # -inf or ~inf

    naim_num_heads: int = 8
    naim_feedforward_dim: int = 1000
    naim_dropout_rate: float = 0.1
    naim_activation: str = "relu"  # relu, gelu
    naim_num_layers: int = 4

    naim_extractor: bool = False
    naim_binary_sigmoid_head: bool = False

    # 범주형을 쓰는 경우에만 세팅 (MPTMS처럼 전부 연속형이면 빈 리스트)
    naim_cat_idxs: list[int] = field(default_factory=list)
    naim_cat_dims: list[int] = field(default_factory=list)


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    train: Train = field(default_factory=Train)
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class DatasetMeta:
    task: Literal["classification", "regression"] = "classification"
    y_col: str = "label"

    continuous_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    input_dim: int = 0
    num_class: int = 0
