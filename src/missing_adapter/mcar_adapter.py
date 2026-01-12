# src/missing_adapter/mcar_adapter.py

import numpy as np


class MCARAdapter:
    def __init__(
        self,
        ratio: float,
        seed: int,
    ):
        self.ratio = ratio
        self.seed = seed

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        X: (N, F) 가정 (시계열 X, sequence X)
        ratio 비율로 완전 무작위(MCAR) 결측을 만든다.
        """
        # X: (N, F)
        N, F = X.shape

        rng = np.random.default_rng(self.seed)

        # True 인 위치를 NaN으로 만든다.
        mask = rng.random(size=(N, F)) < self.ratio  # (N, F)

        X_missing = X.copy()
        X_missing[mask] = np.nan

        return X_missing
