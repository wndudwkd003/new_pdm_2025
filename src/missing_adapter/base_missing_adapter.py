# src/missing_adapter/base_missing_adapter.py

import numpy as np

from abc import ABC, abstractmethod

class BaseMissingAdapter(ABC):
    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        pass
