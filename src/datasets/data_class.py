# src/datasets/data_class.py

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from src.imputer.base_imputer import BaseImputeAdapter
from src.configs.configs import Config, DatasetMeta

from src.missing_adapter.mcar_adapter import MCARAdapter

from src.imputer.mean_imputer import MeanImputer
from src.imputer.zero_imputer import ZeroImputer

from src.datasets.dl_collator import DefaultMissingCollator

from src.params.data_model import Split, DatasetType
from src.params.scenario import MissingScenario, StackMode, MissingPattern, ImputeMethod

from src.datasets.zscore_meta import ZScoreMeta
from src.datasets.balanced_batch_sampler import BalancedClassBatchSampler
from src.imputer.gain_imputer import GAINImputer
from src.imputer.mice_imputer import MICEImputer
from src.imputer.knn_imputer import KNNImputer
from src.imputer.median_imputer import MedianImputer

MISSING_MAP = {
    MissingPattern.MCAR: MCARAdapter,
}

IMPUTER_MAP = {
    ImputeMethod.MEAN: MeanImputer,
    ImputeMethod.ZERO: ZeroImputer,
    ImputeMethod.GAIN: GAINImputer,
    ImputeMethod.MICE: MICEImputer,
    ImputeMethod.KNN: KNNImputer,
    ImputeMethod.MEDIAN: MedianImputer,
}


# =========================================================
# meta helpers
# =========================================================
def _normalize_task(meta_json: dict) -> str:
    """
    meta.json의 task 표기를 내부 표준('classification'/'regression')으로 정규화합니다.
    - 'regress' 포함: regression
    - 'class' 포함 또는 binary/multiclass 계열: classification
    - task가 없으면 num_class 힌트로 추정 (<=1 -> regression, else classification)
    """
    raw = meta_json.get("task", None)
    num_class_hint = meta_json.get("num_class", None)

    if raw is None:
        if num_class_hint is not None:
            try:
                nc = int(num_class_hint)
                return "regression" if nc <= 1 else "classification"
            except Exception:
                return "classification"
        return "classification"

    t = str(raw).strip().lower()

    if "regress" in t:
        return "regression"

    if "class" in t:
        return "classification"

    if t in {"binary", "multiclass", "multi-class", "multi_class"}:
        return "classification"

    # 알 수 없는 문자열이면 기본은 classification으로 둡니다(불필요한 raise 제거)
    return "classification"


def _pick_y_col(meta_json: dict, y_df: pd.DataFrame) -> str:
    """
    y 컬럼명을 최대한 안전하게 선택합니다.
    우선순위:
      1) meta_json["y_col"] (존재 & 실제 컬럼에 있으면)
      2) y_df가 1컬럼이면 그 컬럼
      3) 'label' 컬럼이 있으면 'label'
      4) 그 외 첫 번째 컬럼
    """
    y_col = meta_json.get("y_col", None)
    if y_col is not None and str(y_col) in y_df.columns:
        return str(y_col)

    if y_df.shape[1] == 1:
        return str(y_df.columns[0])

    if "label" in y_df.columns:
        return "label"

    return str(y_df.columns[0])


def _coerce_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    숫자 변환 불가능한 값은 NaN으로 강제 변환합니다.
    (문자열 컬럼이 섞여 있어도 학습 파이프라인이 깨지지 않게)
    """
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ensure_contiguous_labels(y: np.ndarray) -> tuple[np.ndarray, int]:
    """
    분류 라벨을 0..C-1 contiguous로 정규화합니다.
    이미 contiguous면 그대로 둡니다.
    """
    y = np.asarray(y, dtype=np.int64)
    uniq = np.unique(y)

    if uniq.size == 0:
        return y, 0

    # 이미 0..C-1 contiguous이면 OK
    if int(uniq.min()) == 0 and int(uniq.max()) == int(uniq.size - 1):
        return y, int(uniq.size)

    # 아니면 재매핑
    mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
    y2 = np.vectorize(lambda v: mapping[int(v)], otypes=[np.int64])(y)
    return y2.astype(np.int64), int(np.unique(y2).size)


def _to_class_labels(series: pd.Series) -> np.ndarray:
    """
    분류 라벨을 int64로 변환합니다.
    - 숫자형이면 int64 캐스팅
    - 문자열/혼합이면 factorize로 0..C-1 생성
    """
    s = series
    if pd.api.types.is_numeric_dtype(s.dtype):
        y = s.to_numpy(dtype=np.int64)
        return y
    codes, _ = pd.factorize(s.astype(str))
    return codes.astype(np.int64)


# =========================================================
# multiprocessing missing helper
# =========================================================
def _apply_missing_single_ratio(args):
    ratio, X, seed, pattern = args  # pattern: MissingPattern Enum

    Adapter = MISSING_MAP[pattern]
    adapter = Adapter(ratio=ratio, seed=seed)
    X_missing = adapter.transform(X)
    return ratio, X_missing


class Datasets(Dataset):
    def __init__(
        self,
        config: Config,
        split: Split,
        zscore_meta: ZScoreMeta | None = None,
        imputer_dict: dict[str, BaseImputeAdapter] | None = None,
    ):
        super().__init__()

        self.config = config
        self.split = split

        use_dataset = config.data.datasets
        self.data_name = use_dataset.name
        self.data_dir = use_dataset.value
        self.zscore_meta = zscore_meta

        self.imputer_dict_external = imputer_dict
        self.imputer_dict = None

        # CSV 데이터 로드
        X_raw, y_raw, meta = self.load_data()
        self.meta = meta
        self.per_data_size = X_raw.shape[0]

        # 결측 시나리오 적용
        self.missing_dict = self.apply_missing_scenario(X_raw, y_raw)

        # Imputation 적용
        self.imputed_dict = self.apply_imputation(self.missing_dict)

        # Z-score 계산
        if split == Split.TRAIN:
            self.zscore_meta = self.make_zscore_data()

        # Z-score 적용
        self.imputed_dict = self.apply_zscore(self.zscore_meta)

    # ----------------- 필수 메서드 -----------------
    def __len__(self):
        return self.per_data_size

    def __getitem__(self, idx: int):
        return {"base_idx": idx}

    # ----------------- 내부 메서드 -----------------
    def apply_missing_scenario(self, X: np.ndarray, y: np.ndarray):
        ratios = [self.config.data.target_missing_ratio]

        if self.config.data.missing_scenario == MissingScenario.MULTI:
            start, target, step = (
                self.config.data.start_missing_ratio,
                self.config.data.target_missing_ratio,
                self.config.data.step_missing_ratio,
            )
            ratios = np.arange(start, target + step * 0.5, step).round(4).tolist()

        self.ratios = ratios

        missing_dict = {"original": {"X": X, "y": y}}

        total_size = 0
        use_mp_for_missing = self.config.data.data_load_workers > 1

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            missing_dict[pattern_v] = {}

            if use_mp_for_missing:
                num_workers = self.config.data.data_load_workers
                args_iter = [
                    (ratio, X, self.config.train.seed, pattern) for ratio in ratios
                ]

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for ratio, X_missing in tqdm(
                        executor.map(_apply_missing_single_ratio, args_iter),
                        total=len(args_iter),
                        desc=f"Applying missing ({pattern.name})",
                        leave=False,
                    ):
                        missing_dict[pattern_v][ratio] = {"X": X_missing, "y": y}
                        total_size += X_missing.shape[0]
            else:
                for ratio in tqdm(
                    ratios, desc=f"Applying missing ({pattern.name})", leave=False
                ):
                    adapter = MISSING_MAP[pattern](
                        ratio=ratio, seed=self.config.train.seed
                    )
                    X_missing = adapter.transform(X)
                    missing_dict[pattern_v][ratio] = {"X": X_missing, "y": y}
                    total_size += X_missing.shape[0]

        self.total_size = total_size
        return missing_dict

    def apply_imputation(self, missing_dict: dict):
        impute_method = self.config.data.impute_method
        Imputer = IMPUTER_MAP[impute_method]

        imputed_dict = {
            "original": {
                "X": missing_dict["original"]["X"],
                "y": missing_dict["original"]["y"],
            }
        }

        self.imputer_dict = {}
        use_external = self.imputer_dict_external is not None

        for pattern in tqdm(
            self.config.data.missing_patterns, desc=f"Imputing ({impute_method.name})"
        ):
            pattern_v = pattern.value
            ratio_dict = missing_dict[pattern_v]

            if not use_external:
                X_fit = np.concatenate(
                    [ratio_dict[ratio]["X"] for ratio in self.ratios], axis=0
                )
                imputer: BaseImputeAdapter = Imputer()
                imputer.fit(X_fit)
            else:
                imputer = self.imputer_dict_external[pattern_v]

            self.imputer_dict[pattern_v] = imputer

            imputed_dict[pattern_v] = {}
            for ratio in self.ratios:
                X_miss = ratio_dict[ratio]["X"]
                y = ratio_dict[ratio]["y"]
                X_imp, bemv = imputer.transform(X_miss)

                imputed_dict[pattern_v][ratio] = {"X": X_imp, "y": y, "bemv": bemv}

        return imputed_dict

    def get_data_for_gbdt(self):
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]
            for ratio in self.ratios:
                d = ratio_dict[ratio]
                X_list.append(d["X"])
                y_list.append(d["y"])

        X_cat = np.concatenate(X_list, axis=0)
        y_cat = np.concatenate(y_list, axis=0)
        return X_cat, y_cat

    def get_loader_for_deep(self, shuffle: bool = True):
        self.build_dl_view()
        collator = DefaultMissingCollator(self)

        loader = DataLoader(
            self,
            batch_size=self.config.train.batch_size,
            shuffle=shuffle,
            num_workers=self.config.data.num_workers,
            collate_fn=collator,
        )
        return loader

    def build_dl_view(self):
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        bemv_list: list[np.ndarray] = []
        pattern_idx_list: list[np.ndarray] = []
        ratio_idx_list: list[np.ndarray] = []

        patterns = self.config.data.missing_patterns
        ratios = self.ratios

        for p_idx, pattern in enumerate(patterns):
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]

            for r_idx, ratio in enumerate(
                tqdm(ratios, desc=f"Building DL view ({pattern.name})", leave=False)
            ):
                d = ratio_dict[ratio]
                X_imp = d["X"]
                y = d["y"]
                bemv = d["bemv"]

                B = X_imp.shape[0]
                X_list.append(X_imp)
                y_list.append(y)
                bemv_list.append(bemv)

                pattern_idx_list.append(np.full((B,), p_idx, dtype=np.int64))
                ratio_idx_list.append(np.full((B,), r_idx, dtype=np.int64))

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        bemv_all = np.concatenate(bemv_list, axis=0)
        pattern_all = np.concatenate(pattern_idx_list, axis=0)
        ratio_all = np.concatenate(ratio_idx_list, axis=0)

        self.inputs = X_all.astype(np.float32)
        if self.meta.task == "regression":
            self.targets = y_all.astype(np.float32)
        else:
            self.targets = y_all.astype(np.int64)

        self.bemv = bemv_all.astype(np.float32)
        self.pattern_idx = pattern_all
        self.ratio_idx = ratio_all
        self.N = self.inputs.shape[0]

    def get_num_class(self):
        # 불필요한 raise 제거: regression이면 0 반환
        return int(getattr(self.meta, "num_class", 0))

    def load_data(self):
        base_dir = Path(self.data_dir)
        split_dir = base_dir / self.split.value

        meta_path = base_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        # ---- task normalize ----
        task = _normalize_task(meta_json)

        # ---- csv paths ----
        X_path = split_dir / "X.csv"
        y_path = split_dir / "y.csv"

        # ---- load X ----
        X_df = pd.read_csv(X_path)

        conti_cols = meta_json.get("continuous_cols", None)
        cate_cols = meta_json.get("categorical_cols", None)

        conti_cols = conti_cols if isinstance(conti_cols, list) else []
        cate_cols = cate_cols if isinstance(cate_cols, list) else []

        cols_meta = conti_cols + cate_cols
        if cols_meta:
            used_cols = [c for c in cols_meta if c in X_df.columns]
            if not used_cols:
                used_cols = list(X_df.columns)
        else:
            used_cols = list(X_df.columns)

        X_sel = _coerce_numeric_frame(X_df[used_cols])
        X = X_sel.astype(np.float32).to_numpy()

        # meta에는 실제 사용 컬럼 기준으로 기록
        conti_used = (
            [c for c in conti_cols if c in used_cols] if conti_cols else list(used_cols)
        )
        cate_used = [c for c in cate_cols if c in used_cols]

        # ---- load y ----
        y_df = pd.read_csv(y_path)
        y_col = _pick_y_col(meta_json, y_df)

        y_series = y_df[y_col]

        if task == "regression":
            y = pd.to_numeric(y_series, errors="coerce").astype(np.float32).to_numpy()
            num_class = 0
        else:
            y0 = _to_class_labels(y_series)
            y, inferred_nc = _ensure_contiguous_labels(y0)

            # meta num_class가 있어도 데이터에 맞춰 inferred를 우선(불필요한 assert/raise 제거)
            num_class = inferred_nc

        meta = DatasetMeta(
            task=task,  # 'classification' or 'regression'
            y_col=y_col,  # 실제 사용한 y 컬럼명
            continuous_cols=conti_used,
            categorical_cols=cate_used,
            input_dim=int(X.shape[1]),  # 실제 사용한 feature 수로 확정
            num_class=int(num_class),
        )

        return X, y, meta

    def make_zscore_data(self) -> ZScoreMeta:
        X_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]
            for ratio in self.ratios:
                X_list.append(ratio_dict[ratio]["X"])

        X_all = np.concatenate(X_list, axis=0)

        mean = X_all.mean(axis=0).astype(np.float32)
        std = X_all.std(axis=0).astype(np.float32)

        mask = std == 0.0
        std = std + mask.astype(np.float32)

        return ZScoreMeta(mean=mean.tolist(), std=std.tolist())

    def apply_zscore(self, zscore_meta: ZScoreMeta):
        mean = np.asarray(zscore_meta.mean, dtype=np.float32)
        std = np.asarray(zscore_meta.std, dtype=np.float32)

        X0 = self.imputed_dict["original"]["X"]
        self.imputed_dict["original"]["X"] = (X0 - mean) / std

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]
            for ratio in self.ratios:
                X_imp = ratio_dict[ratio]["X"]
                ratio_dict[ratio]["X"] = (X_imp - mean) / std

        return self.imputed_dict
