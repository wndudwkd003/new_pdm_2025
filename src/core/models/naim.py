# src/core/models/naim.py

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import Sigmoid
import torch.nn.functional as F

from src.core.models.taabular_tokenizer import CategoricalFeatureTokenizer


__all__ = ["NAIM"]


class TabularMasker:
    def __init__(self, mask_type: int = 0, missing_value: str = "-inf"):
        self.mask_type = mask_type
        missing_value_options = {"-inf": -torch.inf, "~inf": -1e9}
        self.missing_value = missing_value_options[missing_value]

    def _tabular_sample_mask(self, sample: Tensor) -> Tensor:
        mask = torch.clone(sample)
        mask[~torch.isnan(sample)] = 0
        mask[torch.isnan(sample)] = 1
        return mask

    def mask(self, data: Tensor):
        masks_list = []

        for sample in data:
            # (F,) -> bool (missing=1)
            sample_mask = self._tabular_sample_mask(sample).to(torch.bool)

            # (F,F): missing feature j면 모든 row에서 column j가 True
            sample_mask = sample_mask.repeat(sample_mask.shape[0], 1)

            if self.mask_type == 1:
                sample_mask = ~sample_mask
                sample_mask = ~(sample_mask.mul(sample_mask.T).to(torch.bool))

            masks_list.append(sample_mask.unsqueeze(dim=0))  # (1,F,F)

        masks = torch.cat(masks_list, dim=0)  # (B,F,F) dtype=bool

        # 핵심: baddbmm에 넣을 additive mask는 float이어야 함
        masks = masks.to(dtype=data.dtype)

        # True 위치를 missing_value(-inf)로, False는 0 유지
        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        if self.mask_type != 2:
            return masks, None

        return masks, masks.transpose(-2, -1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        bias: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert input_size % num_heads == 0

        activation_options = dict(relu=F.relu, gelu=F.gelu)

        self.input_size = input_size
        self.num_heads = num_heads
        self.bias = bias
        self.activation = activation_options[activation]
        self.dropout_rate = dropout_rate

        self.linear_q = torch.nn.Linear(input_size, input_size, bias)
        self.linear_k = torch.nn.Linear(input_size, input_size, bias)
        self.linear_v = torch.nn.Linear(input_size, input_size, bias)
        self.linear_o = torch.nn.Linear(input_size, input_size, bias)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, mask2: Tensor = None
    ) -> Tensor:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = torch.repeat_interleave(mask, self.num_heads, 0)
            mask = mask.to(dtype=q.dtype)  # 추가
        if mask2 is not None:
            mask2 = torch.repeat_interleave(mask2, self.num_heads, 0)
            mask2 = mask2.to(dtype=q.dtype)  # 추가

        y, attn_scores = self._scaled_dot_product_attention(
            q, k, v, attn_mask=mask, attn_mask_2=mask2
        )
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return (
            x.reshape(batch_size, seq_len, self.num_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return (
            x.reshape(batch_size, self.num_heads, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        attn_mask_2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        if attn_mask_2 is not None:
            attn = torch.add(attn, attn_mask_2)
            attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)

        output = torch.bmm(attn, v)
        return output, attn


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ff_dim: int,
        num_heads: int,
        bias: bool = False,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(
            emb_dim,
            num_heads,
            bias=bias,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.layer_norm_2 = torch.nn.LayerNorm(emb_dim)

        activation_options = dict(relu=torch.nn.ReLU, gelu=torch.nn.GELU)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ff_dim),
            activation_options[activation](),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, emb_dim),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None, mask2: Tensor | None = None
    ) -> Tensor:
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x, mask=mask, mask2=mask2)
        x = self.layer_norm_2(x)
        x = x + self.ff(x)
        return x


class NAIM(torch.nn.Module):
    """
    NAIM model for tabular data.

    입력 x는 반드시 결측이 NaN으로 표시되어야 합니다.
    - 수치형 feature: NaN이면 missing
    - 범주형 feature: NaN이면 missing (padding idx로 매핑)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        cat_idxs: list[int] | None,
        cat_dims: list[int] | None,
        d_token: int,
        embedder_initialization: str,
        bias: bool,
        mask_type: int = 0,
        missing_value: str = "-inf",
        num_heads: int = 12,
        feedforward_dim: int = 1000,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        num_layers: int = 12,
        extractor: bool = False,
        binary_sigmoid_head: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.extractor = extractor
        self.binary_sigmoid_head = binary_sigmoid_head

        if extractor:
            self.output_size = input_size * d_token
        else:
            self.output_size = output_size

        self.cat_idxs = (
            cat_idxs if (cat_idxs is not None and len(cat_idxs) > 0) else [-1]
        )
        self.cat_dims = (
            cat_dims if (cat_dims is not None and len(cat_dims) > 0) else [-1]
        )

        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias

        self.mask_type = mask_type
        self.missing_value = missing_value

        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_layers = num_layers

        # per-feature tokenizer
        j = 0
        self.embeddings = torch.nn.ModuleList()
        common_params = dict(
            d_token=self.d_token,
            bias=self.bias,
            initialization=self.embedder_initialization,
        )

        for i in range(input_size):
            is_cat = i in self.cat_idxs

            if is_cat:
                # cat_dims[j]는 해당 feature의 cardinality-1의 max index를 의미한다고 가정
                # (0..cat_dims[j]-1 이 실제 범주), cat_dims[j]를 missing(pad)으로 사용
                feature_params = dict(
                    cardinalities=[self.cat_dims[j] + 1],
                    padding_idx=self.cat_dims[j],
                )
                j = j + (i != self.cat_idxs[-1])
            else:
                # numerical: {present, missing} 두 토큰 중 missing이 padding_idx=1
                feature_params = dict(cardinalities=[2], padding_idx=1)

            self.embeddings.append(
                CategoricalFeatureTokenizer(**common_params, **feature_params)
            )

        self.attention_mask = TabularMasker(self.mask_type, self.missing_value)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.encoder = torch.nn.ModuleList(
            [
                EncoderBlock(
                    self.d_token,
                    self.feedforward_dim,
                    self.num_heads,
                    bias=self.bias,
                    activation=self.activation,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(self.d_token)

        if not self.extractor:
            head = torch.nn.Linear(self.input_size * self.d_token, self.output_size)
            if self.binary_sigmoid_head and self.output_size == 1:
                self.classifier = torch.nn.Sequential(head, Sigmoid())
            else:
                self.classifier = torch.nn.Sequential(head)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F) with NaNs for missing
        j = 0
        embeddings = torch.empty((x.shape[0], 0, self.d_token), device=x.device)

        for feature_idx in range(x.shape[1]):
            if feature_idx in self.cat_idxs:
                single_feature = torch.nan_to_num(
                    x[:, feature_idx], nan=self.cat_dims[j]
                ).to(torch.int64)
                feature_values = None
                j += 1
            else:
                # numerical: single_feature는 (missing 여부) 0/1 index, value는 present 값
                single_feature = torch.isnan(x[:, feature_idx]).to(torch.int64)
                feature_values = torch.nan_to_num(x[:, feature_idx], nan=0)

            single_feature_embedding = self.embeddings[feature_idx](
                single_feature, feature_values
            )
            single_feature_embedding = torch.swapaxes(
                single_feature_embedding, 0, 1
            )  # (B, 1, d)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)

        masks, masks2 = self.attention_mask.mask(x)

        for encoder_layer in self.encoder:
            embeddings = encoder_layer(embeddings, mask=masks, mask2=masks2)

        embeddings = self.norm(embeddings)
        features = embeddings.reshape(embeddings.shape[0], -1)

        if self.extractor:
            return features

        logits = self.classifier(features)
        return logits
