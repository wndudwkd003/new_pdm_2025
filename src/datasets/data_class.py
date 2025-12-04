# src/datasets/data_class.py

import os
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm
import pandas as pd
from src.imputer.base_imputer import BaseImputeAdapter
from src.configs.configs import Config, DatasetMeta

from src.missing_adapter.mcar_adapter import MCARAdapter

from src.imputer.mean_imputer import MeanImputer
from src.imputer.zero_imputer import ZeroImputer

from src.datasets.dl_collator import DefaultMissingCollator

from src.params.data_model import Split, DatasetType
from src.params.scenario import (
    MissingScenario, StackMode,
    MissingPattern, ImputeMethod
)

from src.datasets.zscore_meta import ZScoreMeta




MISSING_MAP = {
    MissingPattern.MCAR: MCARAdapter,
}

IMPUTER_MAP = {
    ImputeMethod.MEAN: MeanImputer,
    ImputeMethod.ZERO: ZeroImputer,
}

def _apply_missing_single_ratio(args):
    ratio, X, seed, pattern = args   # pattern: MissingPattern Enum

    Adapter = MISSING_MAP[pattern]
    adapter = Adapter(
        ratio=ratio,
        seed=seed,
    )

    X_missing = adapter.transform(X)
    return ratio, X_missing



def _load_single_sample(args):
    idx, sample, skip_header = args

    csv_paths   = sample["input_files"]["csvs"]
    label_paths = sample["target_files"]["labels"]

    rows = []
    ys = []

    for csv_path in csv_paths:
        row = np.loadtxt(
            csv_path,
            delimiter=",",
            dtype=np.float32,
            skiprows=skip_header,
        )
        rows.append(row)

    for label_path in label_paths:
        with open(label_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        state = d["annotations"][0]["tagging"][0]["state"]
        ys.append(state)

    x_ts = np.stack(rows, axis=0)
    y_vec = np.array(ys, dtype=np.int64)

    return idx, x_ts, y_vec




class Datasets(Dataset):
    def __init__(
        self,
        config: Config,
        split: Split,
        zscore_meta: ZScoreMeta | None = None,
    ):
        super().__init__()

        self.config = config
        self.split = split

        use_dataset = config.data.datasets
        self.data_name = use_dataset.name
        self.data_dir = use_dataset.value
        self.zscore_meta = zscore_meta


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




    def numpy_from_samples(self, samples: list[dict]):
        ds_type = self.config.data.datasets
        print(f"DatasetType: {ds_type.name}")

        if ds_type == DatasetType.MPTMS:
            return self.numpy_from_samples_mptms(samples)
        elif ds_type in (DatasetType.FORD, DatasetType.CMAPSS):
            return self.numpy_from_samples_ford(samples)

        else:
            raise ValueError(f"지원하지 않는 DatasetType: {ds_type}")


    def numpy_from_samples_mptms(self, samples: list[dict]):
        # 헤더 스킵 여부는 그대로 설정 (자동 감지 안 함)
        skip_header = 1 if self.config.data.skip_header else 0

        num_samples = len(samples)
        X_results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * num_samples

        # ProcessPoolExecutor를 사용해서 병렬로 샘플 로딩
        num_workers = self.config.data.data_load_workers

        args_iter = [
            (idx, sample, skip_header)
            for idx, sample in enumerate(samples)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, x_ts, y_vec in tqdm(
                executor.map(_load_single_sample, args_iter),
                total=num_samples,
                desc=f"Loading CSV/labels ({self.split.value})"
            ):
                X_results[idx] = (x_ts, y_vec)

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for idx in range(num_samples):
            x_ts, y_vec = X_results[idx]

            X_list.append(x_ts)
            y_list.append(y_vec)

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        return X, y

    def numpy_from_samples_ford(self, samples: list[dict]):
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for sample in tqdm(samples, desc=f"Processing samples ({self.split.value})"):
            # X: (S, F)
            x_ts = np.asarray(sample["input"]["X"], dtype=np.float32)

            # y: (T,) 또는 스칼라 → np.array 로 통일
            y_raw = sample["target"]["y"]
            y_vec = np.asarray(y_raw, dtype=np.int64)


            X_list.append(x_ts)
            y_list.append(y_vec)

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        return X, y




    # ----------------- 필수 메서드 -----------------

    def __len__(self):
        return self.per_data_size

    def __getitem__(self, idx: int):
        return {"base_idx": idx}

    # ----------------- 내부 메서드 -----------------

    def apply_missing_scenario(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        # 기본 비율
        ratios = [self.config.data.target_missing_ratio]

        # 다중 시나리오인 경우 비율을 여러 개 설정
        if self.config.data.missing_scenario == MissingScenario.MULTI:
            start, target, step = (
                self.config.data.start_missing_ratio,
                self.config.data.target_missing_ratio,
                self.config.data.step_missing_ratio,
            )
            ratios = np.arange(start, target + step * 0.5, step).round(4).tolist()

        self.ratios = ratios

        # recon 용도 원본 데이터
        missing_dict = {
            "original": {
                "X": X,
                "y": y,
            }
        }

        total_size = 0

        # 데이터셋 종류 상관 없이, worker 수가 1보다 크면 멀티프로세싱 사용
        use_mp_for_missing = self.config.data.data_load_workers > 1

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            missing_dict[pattern_v] = dict()

            if use_mp_for_missing:
                # ---- 멀티프로세싱 버전 ----
                num_workers = self.config.data.data_load_workers

                args_iter = [
                    (ratio, X, self.config.train.seed, pattern)
                    for ratio in ratios
                ]

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    for ratio, X_missing in tqdm(
                        executor.map(_apply_missing_single_ratio, args_iter),
                        total=len(args_iter),
                        desc=f"Applying missing ({pattern.name})",
                        leave=False,
                    ):
                        missing_dict[pattern_v][ratio] = {
                            "X": X_missing,
                            "y": y,
                        }
                        total_size += X_missing.shape[0]

            else:
                # ---- 단일 프로세스 버전 (기존 로직) ----
                for i, ratio in enumerate(
                    tqdm(ratios, desc=f"Applying missing ({pattern.name})", leave=False)
                ):
                    missing_dict[pattern_v][ratio] = dict()

                    adapter = MISSING_MAP[pattern](
                        ratio=ratio,
                        seed=self.config.train.seed,
                    )

                    X_missing = adapter.transform(X)
                    missing_dict[pattern_v][ratio]["X"] = X_missing
                    missing_dict[pattern_v][ratio]["y"] = y

                    total_size += X_missing.shape[0]

        self.total_size = total_size

        return missing_dict




    def apply_imputation(
        self,
        missing_dict: dict,
    ):
        impute_method = self.config.data.impute_method

        Imputer = IMPUTER_MAP[impute_method]

        imputed_dict = {
            "original": {
                "X": missing_dict["original"]["X"],
                "y": missing_dict["original"]["y"],
            }
        }

        # 패턴별로 impute
        for pattern in tqdm(self.config.data.missing_patterns, desc=f"Imputing ({impute_method.name})"):
            pattern_v = pattern.value

            ratio_dict = missing_dict[pattern_v]

            X_fit_list = []

            for ratio in self.ratios:
                X_miss = ratio_dict[ratio]["X"]
                X_fit_list.append(X_miss)

            X_fit = np.concatenate(X_fit_list, axis=0)

            imputer: BaseImputeAdapter = Imputer()
            imputer.fit(X_fit)

            imputed_dict[pattern_v] = dict()

            for ratio in self.ratios:
                X_miss = ratio_dict[ratio]["X"]
                y = ratio_dict[ratio]["y"]

                X_imp, bemv = imputer.transform(X_miss)

                imputed_dict[pattern_v][ratio] = {
                    "X": X_imp,
                    "y": y,
                    "bemv": bemv,
                }

        return imputed_dict


    def get_data_for_gbdt(self):
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value

            ratio_dict = self.imputed_dict[pattern_v]

            for ratio in self.ratios:
                d = ratio_dict[ratio]

                X_imp = d["X"]   # (N, F)
                y = d["y"]       # (N,)

                X_list.append(X_imp)
                y_list.append(y)

        X_cat = np.concatenate(X_list, axis=0)   # (M, F)
        y_cat = np.concatenate(y_list, axis=0)   # (M,)

        return X_cat, y_cat


    def get_loader_for_deep(
        self,
        shuffle: bool = True,
    ):
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

            for r_idx, ratio in enumerate(tqdm(ratios, desc=f"Building DL view ({pattern.name})", leave=False)):
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
        self.targets = y_all.astype(np.int64)
        self.bemv = bemv_all.astype(np.float32)
        self.pattern_idx = pattern_all
        self.ratio_idx = ratio_all

        self.N = self.inputs.shape[0]

    def get_num_class(self):
        return self.meta.num_class


    def load_data(self):
        base_dir = Path(self.data_dir)
        split_dir = base_dir / self.split.value

        # meta read
        meta_path = base_dir / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        conti_cols = meta_json["continuous_cols"]
        cate_cols = meta_json["categorical_cols"]
        num_class = meta_json["num_class"]
        input_dim = meta_json["input_dim"]

        # csv path
        X_path = split_dir / "X.csv"
        y_path = split_dir / "y.csv"

        # load
        X_df = pd.read_csv(X_path)
        X = X_df[conti_cols + cate_cols].astype(np.float32).values

        y_df = pd.read_csv(y_path)
        y = y_df["label"].astype(np.int64).values

        meta = DatasetMeta(
            continuous_cols=conti_cols,
            categorical_cols=cate_cols,
            input_dim=input_dim,
            num_class=num_class,
        )

        # (N, F), (N, )
        return X, y, meta


    def make_zscore_data(self) -> ZScoreMeta:
        X_list: list[np.ndarray] = []

        # imputed_dict 에 들어있는 모든 X 모아서 통계 계산
        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]

            for ratio in self.ratios:
                X_imp = ratio_dict[ratio]["X"]   # (N, F)
                X_list.append(X_imp)

        X_all = np.concatenate(X_list, axis=0)   # (M, F)

        mean = X_all.mean(axis=0).astype(np.float32)  # (F,)
        std = X_all.std(axis=0).astype(np.float32)    # (F,)

        # 분산 0인 경우 1로 치환 (branch 없이 mask로 처리)
        mask = (std == 0.0)
        std = std + mask.astype(np.float32)

        return ZScoreMeta(
            mean=mean.tolist(),
            std=std.tolist(),
        )


    def apply_zscore(self, zscore_meta: ZScoreMeta):
        mean = np.asarray(zscore_meta.mean, dtype=np.float32)  # (F,)
        std = np.asarray(zscore_meta.std, dtype=np.float32)    # (F,)

        # original
        X0 = self.imputed_dict["original"]["X"]   # (N, F)
        X0 = (X0 - mean) / std
        self.imputed_dict["original"]["X"] = X0

        # 패턴별 / ratio별
        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = self.imputed_dict[pattern_v]

            for ratio in self.ratios:
                X_imp = ratio_dict[ratio]["X"]    # (N, F)
                X_imp = (X_imp - mean) / std
                ratio_dict[ratio]["X"] = X_imp

        return self.imputed_dict






