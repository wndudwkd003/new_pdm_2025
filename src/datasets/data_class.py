# src/datasets/data_class.py

import numpy as np
from torch.utils.data import Dataset

from dataclasses import dataclass

from src.configs.configs import DatasetMeta

@dataclass
class DatasetClass(Dataset):
    inputs: np.ndarray
    targets: np.ndarray
    meta: DatasetMeta

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def as_numpy(self):
        return self.inputs, self.targets

    def get_horizon(self):
        return self.meta.horizon

    def get_num_class(self):
        return self.meta.num_class
