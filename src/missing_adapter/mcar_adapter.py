# src/missing_adapter/mcar_adapter.py

import numpy as np
from src.missing_adapter.base_missing_adapter import BaseMissingAdapter


class MCARAdapter(BaseMissingAdapter):
    def __init__(
        self,
        ratio: float,
        seed: int,
    ):
        self.ratio = ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def transform(
        self,
        X: np.ndarray,
    ):
        X_out = X.copy()
        X_out = X_out.astype(np.float32)

        B, S, F = X.shape

        if self.ratio == 0.0:
            return X_out

        else:
            mfc = int(F * self.ratio) # masked feature count

            for b in range(B):
                for s in range(S):
                    cols = self.rng.choice(F, size=mfc, replace=False)
                    X_out[b, s, cols] = np.nan

        return X_out

