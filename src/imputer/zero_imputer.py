# src/imputer/zero_imputer.py

import numpy as np

from src.imputer.base_imputer import BaseImputeAdapter


class ZeroImputer(BaseImputeAdapter):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray):
        pass

    def transform(self, X: np.ndarray):
        X_out = X.copy().astype(np.float32)
        nan_mask = np.isnan(X_out)
        X_out[nan_mask] = 0.0
        bemv = (~nan_mask).astype(np.float32)
        return X_out, bemv
