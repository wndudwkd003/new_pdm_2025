
# src/datasets/data_class.py

import os
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm

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





MISSING_MAP = {
    MissingPattern.MCAR: MCARAdapter,
}

IMPUTER_MAP = {
    ImputeMethod.MEAN: MeanImputer,
    ImputeMethod.ZERO: ZeroImputer,
}


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
    ):
        super().__init__()

        self.config = config
        self.split = split

        use_dataset = config.data.datasets
        self.data_name = use_dataset.name
        self.data_dir = use_dataset.value



        # jsonl 파일 로드
        samples, meta = self.load_data()
        self.meta = meta

        # 실제 데이터 로드
        X_raw, y_raw = self.numpy_from_samples(samples)
        self.per_data_size = X_raw.shape[0]

        # 결측 시나리오 적용
        self.missing_dict = self.apply_missing_scenario(X_raw, y_raw)

        # imputation 적용
        self.imputed_dict = self.apply_imputation(self.missing_dict)


    def load_data(self):
        ds_type = self.config.data.datasets

        if ds_type == DatasetType.MPTMS:
            return self.load_data_mptms()

        elif ds_type == DatasetType.FORD:
            return self.load_data_ford()

        else:
            raise ValueError(f"지원하지 않는 DatasetType: {ds_type}")

    def numpy_from_samples(
        self,
        samples: list[dict],
    ):
        """
        DatasetType 에 따라 적절한 numpy 변환 함수를 호출하는 공용 메서드.
        __init__ 에서 항상 이 함수를 통해 X, y 를 얻도록 한다.
        """
        ds_type = self.config.data.datasets

        if ds_type == DatasetType.MPTMS:
            return self._numpy_from_samples_mptms(samples)

        elif ds_type == DatasetType.FORD:
            return self._numpy_from_samples_ford(samples)

        else:
            raise ValueError(f"지원하지 않는 DatasetType: {ds_type}")


    def _numpy_from_samples_mptms(
        self,
        samples: list[dict],
    ):
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



    def _numpy_from_samples_ford(
        self,
        samples: list[dict],
    ):
        """
        Ford: JSONL이 X(그리고 y)를 직접 들고 있는 구조를 가정.

        예시 구조:
        {
            "sample_id": "...",
            "input": {
                "X": [[...], [...], ...]   # (S, F)
            },
            "target": {
                "y": [...]                 # (T,) 또는 스칼라
            },
            "metadata": {...}
        }

        y 위치/키가 다르면 아래 "target"]["y"] 부분만
        실제 JSONL 구조에 맞게 한 번만 수정하시면 됩니다.
        """

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for sample in samples:
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

    def get_flat_2d(self, X: np.ndarray):
        B, S, F = X.shape
        X_2d = X.reshape(B, S * F)
        return X_2d

    def apply_missing_scenario(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
       # 기본 비율
        ratios = [self.config.data.target_missing_ratio]


        # 다중 시나리오인 경우 비율을 여러개 설정함
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

        # 각 비율에 따라 결측 패턴 순차 적용
        for pattern in self.config.data.missing_patterns:
            Adapter = MISSING_MAP[pattern]
            missing_dict[pattern.value] = dict()

            for i, ratio in enumerate(
                tqdm(ratios, desc=f"Applying missing ({pattern.name})", leave=False)
            ):
                missing_dict[pattern.value][ratio] = dict()

                 # 어댑터 생성 및 변환
                adapter = Adapter(
                    ratio=ratio,
                    seed=self.config.train.seed # 시드 고정
                )

                X_missing = adapter.transform(X)
                missing_dict[pattern.value][ratio]["X"] = X_missing
                missing_dict[pattern.value][ratio]["y"] = y

                total_size += X_missing.shape[0]

        self.total_size = total_size            # 결측 시나리오 적용된 전체 데이터 수

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


    def get_data_for_gbdt(
        self,
    ):

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value

            ratio_dict = self.imputed_dict[pattern_v]

            for ratio in self.ratios:
                d = ratio_dict[ratio]

                X_imp = d["X"]
                y = d["y"]

                X_list.append(X_imp)
                y_list.append(y)

        X_cat = np.concatenate(X_list, axis=0)
        y_cat = np.concatenate(y_list, axis=0)

        B, S, F = X_cat.shape
        X_cat = X_cat.reshape(B, S * F)

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

                X_imp = d["X"]      # (B, S, F)
                y = d["y"]          # (B, T)
                bemv = d["bemv"]    # (B, S, F)

                B = X_imp.shape[0]

                X_list.append(X_imp)
                y_list.append(y)
                bemv_list.append(bemv)

                pattern_idx_list.append(
                    np.full((B,), p_idx, dtype=np.int64)
                )
                ratio_idx_list.append(
                    np.full((B,), r_idx, dtype=np.int64)
                )

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




    def get_horizon(self):
        return self.meta.horizon

    def get_num_class(self):
        return self.meta.num_class
