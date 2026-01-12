# src/imputer/gain_imputer.py

from __future__ import annotations

import numpy as np

from src.imputer.base_imputer import BaseImputeAdapter


class GAINImputer(BaseImputeAdapter):
    """
    HyperImpute의 GAIN 구현(GainImputation)을 직접 사용합니다.
    - fit(X): torch 텐서로 변환 후 (가능하면) CUDA에서 학습
    - transform(X): (X_out, bemv) 반환
      * bemv: 관측=1, 결측=0, shape=(N, F)
    """

    def __init__(
        self,
        *,
        batch_size: int = 256,
        n_epochs: int = 1000,
        hint_rate: float = 0.9,
        loss_alpha: float = 10.0,
        device: str = "auto",  # "auto" | "cpu" | "cuda" | "cuda:0" ...
    ):
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.hint_rate = float(hint_rate)
        self.loss_alpha = float(loss_alpha)
        self.device = device

        self._model = None
        self._fitted = False
        self._n_features: int | None = None

    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_model(self):
        try:
            from hyperimpute.plugins.imputers.plugin_gain import GainImputation
        except Exception as e:
            raise ImportError(
                "hyperimpute가 설치되어 있지 않습니다. `pip install hyperimpute` 를 실행해 주세요."
            ) from e

        return GainImputation(
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            hint_rate=self.hint_rate,
            loss_alpha=self.loss_alpha,
        )

    def fit(self, X: np.ndarray):
        import torch

        if X.ndim != 2:
            raise ValueError(f"GAINImputer.fit expects 2D array (N, F), got {X.shape}")

        self._n_features = int(X.shape[1])
        self._model = self._build_model()

        dev = self._resolve_device()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)

        self._model.fit(X_t)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray):
        import torch

        if not self._fitted or self._model is None:
            raise ValueError("먼저 fit 해라")

        if X.ndim != 2:
            raise ValueError(
                f"GAINImputer.transform expects 2D array (N, F), got {X.shape}"
            )

        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(
                f"feature dim mismatch: fitted F={self._n_features}, got F={X.shape[1]}"
            )

        X_in = np.asarray(X, dtype=np.float32)
        nan_mask = np.isnan(X_in)
        bemv = (~nan_mask).astype(np.float32)

        dev = self._resolve_device()
        X_t = torch.as_tensor(X_in, device=dev)

        with torch.no_grad():
            out_t = self._model.transform(X_t)

        X_out = out_t.detach().to("cpu").numpy().astype(np.float32, copy=False)
        return X_out, bemv
