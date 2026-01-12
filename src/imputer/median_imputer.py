import numpy as np
from src.imputer.base_imputer import BaseImputeAdapter


class MedianImputer(BaseImputeAdapter):
    def __init__(self):
        self.feature_median: np.ndarray | None = None
        self._all_nan_cols: np.ndarray | None = None  # (F,) bool

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError(
                f"MedianImputer.fit expects 2D array (N, F), but got {X.shape}"
            )

        Xf = np.asarray(X, dtype=np.float32)
        all_nan_cols = np.isnan(Xf).all(axis=0)
        self._all_nan_cols = all_nan_cols

        # np.nanmedian는 전부 NaN인 컬럼이면 warning/NaN이 나올 수 있어서 보호
        med = np.nanmedian(Xf, axis=0).astype(np.float32)

        if all_nan_cols.any():
            med = med.copy()
            med[all_nan_cols] = (
                0.0  # 전부 NaN인 컬럼은 0으로 고정(원하시면 다른 값으로)
            )

        self.feature_median = med

    def transform(self, X: np.ndarray):
        if self.feature_median is None:
            raise ValueError("먼저 fit 해라")

        if X.ndim != 2:
            raise ValueError(
                f"MedianImputer.transform expects 2D array (N, F), but got {X.shape}"
            )

        X_out = np.asarray(X, dtype=np.float32).copy()
        nan_mask = np.isnan(X_out)
        bemv = (~nan_mask).astype(np.float32)

        N, F = X_out.shape
        med_2d = np.broadcast_to(self.feature_median, (N, F))
        X_out[nan_mask] = med_2d[nan_mask]

        return X_out, bemv
