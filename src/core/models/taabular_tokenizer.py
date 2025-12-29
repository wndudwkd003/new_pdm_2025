# src/core/models/tabular_tokenizer.py

from __future__ import annotations

import enum
import math
from typing import List, Optional

import torch
from torch import Tensor, nn


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        return cls(initialization)

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Feature-wise tokenizer (Embedding + optional bias).
    - value가 주어지면 (numerical feature present) embedding을 value로 스케일링합니다.
    - padding_idx를 지정하면 해당 인덱스 벡터는 0으로 고정됩니다.
    """

    category_offsets: Tensor

    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool,
        padding_idx: Optional[int] = None,
        initialization: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert len(cardinalities) > 0
        assert d_token > 0

        initialization_ = None
        if initialization is not None:
            initialization_ = _TokenInitialization.from_str(initialization)

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

        self.padding_idx = padding_idx
        self.embeddings = nn.Embedding(
            sum(cardinalities),
            d_token,
            padding_idx=padding_idx,
            norm_type=2,
            max_norm=1,
        )
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        if initialization_ is not None:
            initialization_.apply(self.embeddings.weight, d_token)
            if self.bias is not None:
                initialization_.apply(self.bias, d_token)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embeddings.weight[self.padding_idx].fill_(0)

    @property
    def n_tokens(self) -> int:
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor, value: Tensor | None = None) -> Tensor:
        # x: (B,) or (B, 1) with integer indices
        x = self.embeddings(x.long() + self.category_offsets[None])

        if value is not None:
            # value: (B,) -> (1, B, d)
            value = value.unsqueeze(0).unsqueeze(2).repeat(1, 1, x.shape[2])
            x = value * x

        if self.bias is not None:
            x = x + self.bias[None]
        return x
