# src/datasets/data_class.py

from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

from src.configs.configs import Config, DatasetMeta

from src.missing_adapter.base_missing_adapter import BaseMissingAdapter
from src.missing_adapter.mcar_adapter import MCARAdapter

from src.params.literals import Split
from src.params.scenario import MissingScenario, StackMode, MissingPattern



MISSING_MAP = {
    MissingPattern.MCAR: MCARAdapter,

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

        # 결측 시나리오 적용
        X_missing, y_missing = self.apply_missing_scenario(X_raw, y_raw)





    # ----------------- 필수 메서드 -----------------

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


    # ----------------- 내부 메서드 -----------------

    @staticmethod
    def get_missing_adapter(missing_pattern: MissingPattern):
        return MISSING_MAP[missing_pattern]


    def apply_missing_scenario(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        for missing_pattern in self.config.data.missing_patterns:
            missing_adapter = self.get_missing_adapter(missing_pattern)


    def as_numpy(self):
        return self.inputs, self.targets

    def get_horizon(self):
        return self.meta.horizon

    def get_num_class(self):
        return self.meta.num_class


    def load_data(
        self,
    ):
        data_path = Path(self.data_dir) / self.split
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

        for sample in samples:
            csv_paths   = sample["input_files"]["csvs"]
            label_paths = sample["target_files"]["labels"]

            rows = []
            ys = []

            for csv_path in csv_paths:
                row = np.genfromtxt(
                    csv_path,
                    delimiter=',',
                    dtype=np.float32,
                    skip_header=skip_header,
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







