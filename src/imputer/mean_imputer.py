# src/imputer/mean_imputer.py

import numpy as np

from src.imputer.base_imputer import BaseImputeAdapter


class MeanImputer(BaseImputeAdapter):
    def __init__(self):
        self.feature_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        # 테이블 데이터 전용: X는 (N, F) 형태라고 가정
        if X.ndim != 2:
            raise ValueError(f"MeanImputer.fit expects 2D array (N, F), but got shape {X.shape}")

        N, F = X.shape
        X_2d = X.astype(np.float32)

        # feature별 평균 (F,)
        means = np.nanmean(X_2d, axis=0, dtype=np.float32)
        self.feature_mean = means

    def transform(self, X: np.ndarray):
        if self.feature_mean is None:
            raise ValueError("먼저 fit 해라")

        if X.ndim != 2:
            raise ValueError(f"MeanImputer.transform expects 2D array (N, F), but got shape {X.shape}")

        X_out = X.copy().astype(np.float32)

        N, F = X_out.shape
        nan_mask = np.isnan(X_out)

        # self.feature_mean: (F,)
        # → (N, F)로 브로드캐스트
        mean_2d = np.broadcast_to(self.feature_mean, (N, F))

        X_out[nan_mask] = mean_2d[nan_mask]

        # bemv: 관측=1, 결측=0
        bemv = (~nan_mask).astype(np.float32)

        return X_out, bemv
