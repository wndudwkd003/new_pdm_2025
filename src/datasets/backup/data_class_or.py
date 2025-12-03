# src/datasets/data_class.py


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

from src.params.data_model import Split
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


    def load_data(
        self,
    ):
        data_path = Path(self.data_dir) / self.split.value
        jsonl_files = list(data_path.glob("*.jsonl"))

        samples: list[dict] = []

        for fpath in jsonl_files:
            with open(fpath, "r", encoding="utf-8") as f:
                objs = [json.loads(line.strip()) for line in f]
                samples.extend(objs)

        # meta 파일은 다 동일해서 0번 샘플에서 가져옴
        _meta = samples[0]["metadata"]

        continuous_cols = _meta["continuous_cols"]
        categorical_cols = _meta["categorical_cols"]
        feature_dim = len(continuous_cols) + len(categorical_cols)


        # meta 정보 추가할 일 있으면 여기에 추가하면 됨
        meta = DatasetMeta(
            horizon=_meta["backward"],
            sequence=_meta["forward"],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            feature_dim=feature_dim,
            num_class=_meta["num_class"],
        )

        return samples, meta


    # todo: 이거 데이터에 따라서 다르게 불러오는 방법으로 변경하기
    def numpy_from_samples(
        self,
        samples: list[dict],
    ):

        X_list = []
        y_list = []

        skip_header = 1 if self.config.data.skip_header else 0

        for sample in tqdm(samples, desc=f"Loading CSV/labels ({self.split.value})"):
            csv_paths   = sample["input_files"]["csvs"]
            label_paths = sample["target_files"]["labels"]

            rows = []
            ys = []

            for csv_path in csv_paths:
                # row = np.genfromtxt(
                #     csv_path,
                #     delimiter=',',
                #     dtype=np.float32,
                #     skip_header=skip_header,
                # )
                # rows.append(row)

                row = np.loadtxt(
                    csv_path,
                    delimiter=',',
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

            X_list.append(x_ts)
            y_list.append(y_vec)

        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)

        return X, y







