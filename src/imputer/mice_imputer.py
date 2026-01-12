import numpy as np
from src.imputer.base_imputer import BaseImputeAdapter
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  <-- 반드시 먼저

from sklearn.impute import IterativeImputer


class MICEImputer(BaseImputeAdapter):
    def __init__(
        self,
        *,
        max_iter: int = 10,
        random_state: int = 42,
        sample_posterior: bool = False,
        n_nearest_features: int | None = None,
        initial_strategy: str = "mean",
        tol: float = 1e-3,
    ):
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.sample_posterior = bool(sample_posterior)
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = str(initial_strategy)
        self.tol = float(tol)

        self.imputer: IterativeImputer | None = None
        self._all_nan_cols: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError(
                f"MICEImputer.fit expects 2D array (N, F), but got {X.shape}"
            )

        Xf = np.asarray(X, dtype=np.float32)

        all_nan_cols = np.isnan(Xf).all(axis=0)
        self._all_nan_cols = all_nan_cols
        if all_nan_cols.any():
            Xf = Xf.copy()
            Xf[:, all_nan_cols] = 0.0

        self.imputer = IterativeImputer(
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            sample_posterior=self.sample_posterior,
            n_nearest_features=self.n_nearest_features,
            initial_strategy=self.initial_strategy,
            skip_complete=True,
        )
        self.imputer.fit(Xf)
        return self

    def transform(self, X: np.ndarray):
        if self.imputer is None:
            raise ValueError("먼저 fit 해라")

        if X.ndim != 2:
            raise ValueError(
                f"MICEImputer.transform expects 2D array (N, F), but got {X.shape}"
            )

        X_in = np.asarray(X, dtype=np.float32)
        nan_mask = np.isnan(X_in)
        bemv = (~nan_mask).astype(np.float32)

        Xf = X_in
        if self._all_nan_cols is not None and self._all_nan_cols.any():
            Xf = Xf.copy()
            Xf[:, self._all_nan_cols] = 0.0

        X_out = self.imputer.transform(Xf).astype(np.float32)
        if np.isnan(X_out).any():
            X_out = np.nan_to_num(X_out, nan=0.0).astype(np.float32)

        return X_out, bemv
