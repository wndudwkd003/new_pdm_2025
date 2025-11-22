# src/imputer/mean_imputer.py

import numpy as np

from src.imputer.base_imputer import BaseImputeAdapter

class MeanImputer(BaseImputeAdapter):
    def __init__(self):
        self.feature_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        B, S, F = X.shape

        X_2d = X.reshape(B * S, F)

        means = np.nanmean(X_2d, axis=0, dtype=np.float32)
        self.feature_mean = means

    def transform(self, X: np.ndarray):
        if self.feature_mean is None:
            raise ValueError("먼저 fit 해라")

        X_out = X.copy()

        B, S, F = X_out.shape
        nan_mask = np.isnan(X_out)
        mean_3d = np.broadcast_to(self.feature_mean, (B, S, F))

        X_out[nan_mask] = mean_3d[nan_mask]

        # bemv: "Binary Encoding Missing Values"
        # 1=관측, 0=결측 이라면:
        bemv = (~nan_mask).astype(np.float32)

        return X_out, bemv



