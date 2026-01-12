import numpy as np
from src.imputer.base_imputer import BaseImputeAdapter


from sklearn.impute import KNNImputer as SklearnKNNImputer  # ★ alias


class KNNImputer(BaseImputeAdapter):  # ★ 래퍼 이름을 KNNImputer로
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "uniform",  # "uniform" | "distance"
        metric: str = "nan_euclidean",
    ):
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.metric = str(metric)

        self.imputer: SklearnKNNImputer | None = None
        self._all_nan_cols: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError(
                f"KNNImputer.fit expects 2D array (N, F), but got {X.shape}"
            )

        Xf = np.asarray(X, dtype=np.float32)

        all_nan_cols = np.isnan(Xf).all(axis=0)
        self._all_nan_cols = all_nan_cols
        if all_nan_cols.any():
            Xf = Xf.copy()
            Xf[:, all_nan_cols] = 0.0

        self.imputer = SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        self.imputer.fit(Xf)
        return self

    def transform(self, X: np.ndarray):
        if self.imputer is None:
            raise ValueError("먼저 fit 해라")

        if X.ndim != 2:
            raise ValueError(
                f"KNNImputer.transform expects 2D array (N, F), but got {X.shape}"
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
