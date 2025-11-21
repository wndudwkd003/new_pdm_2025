# src/imputer/base_imputer.py

import numpy as np
from abc import ABC, abstractmethod


class BaseImputeAdapter(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
