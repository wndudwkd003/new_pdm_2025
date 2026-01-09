# src/core/models/ReGVAE.py

from __future__ import annotations

import math
import typing
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int) -> None:
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, however: {n_features=}")
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")
        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )
        x = x[..., None] * self.weight
        x = x + self.bias[None]
        return x


class CategoricalEmbeddings(nn.Module):
    def __init__(
        self, cardinalities: List[int], d_embedding: int, bias: bool = True
    ) -> None:
        super().__init__()
        if not cardinalities:
            raise ValueError("cardinalities must not be empty")
        if any(v <= 0 for v in cardinalities):
            i, value = next((i, v) for i, v in enumerate(cardinalities) if v <= 0)
            raise ValueError(
                "cardinalities must contain only positive values,"
                f" however: cardinalities[{i}]={value}"
            )
        if d_embedding <= 0:
            raise ValueError(f"d_embedding must be positive, however: {d_embedding=}")

        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d_embedding) for c in cardinalities]
        )
        self.bias = (
            Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"The input must have at least two dimensions, however: {x.ndim=}"
            )
        n_features = len(self.embeddings)
        if x.shape[-1] != n_features:
            raise ValueError(
                "The last input dimension must equal the number of categorical features."
                f" However: {x.shape[-1]=}, len(cardinalities)={n_features}"
            )

        x = torch.stack(
            [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        )
        if self.bias is not None:
            x = x + self.bias
        return x


_LINFORMER_KV_COMPRESSION_SHARING = Literal["headwise", "key-value"]


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, however: {n_heads=}")
        if d_embedding % n_heads:
            raise ValueError(
                "d_embedding must be a multiple of n_heads,"
                f" however: {d_embedding=}, {n_heads=}"
            )

        super().__init__()
        self.W_q = nn.Linear(d_embedding, d_embedding)
        self.W_k = nn.Linear(d_embedding, d_embedding)
        self.W_v = nn.Linear(d_embedding, d_embedding)
        self.W_out = nn.Linear(d_embedding, d_embedding) if n_heads > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self._n_heads = n_heads

        if linformer_kv_compression_ratio is not None:
            if n_tokens is None:
                raise ValueError(
                    "If linformer_kv_compression_ratio is not None,"
                    " then n_tokens also must not be None"
                )
            if linformer_kv_compression_sharing not in typing.get_args(
                _LINFORMER_KV_COMPRESSION_SHARING
            ):
                raise ValueError(
                    "Valid values of linformer_kv_compression_sharing include:"
                    f" {typing.get_args(_LINFORMER_KV_COMPRESSION_SHARING)},"
                    f" however: {linformer_kv_compression_sharing=}"
                )
            if (
                linformer_kv_compression_ratio <= 0.0
                or linformer_kv_compression_ratio >= 1.0
            ):
                raise ValueError(
                    "linformer_kv_compression_ratio must be from the open interval"
                    f" (0.0, 1.0), however: {linformer_kv_compression_ratio=}"
                )

            def make_linformer_kv_compression() -> nn.Linear:
                return nn.Linear(
                    n_tokens,
                    max(int(n_tokens * linformer_kv_compression_ratio), 1),
                    bias=False,
                )

            self.key_compression = make_linformer_kv_compression()
            self.value_compression = (
                make_linformer_kv_compression()
                if linformer_kv_compression_sharing == "headwise"
                else None
            )
        else:
            if n_tokens is not None:
                raise ValueError(
                    "If linformer_kv_compression_ratio is None, then n_tokens also must be None"
                )
            if linformer_kv_compression_sharing is not None:
                raise ValueError(
                    "If linformer_kv_compression_ratio is None, then linformer_kv_compression_sharing also must be None"
                )
            self.key_compression = None
            self.value_compression = None

        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self._n_heads
        return (
            x.reshape(batch_size, n_tokens, self._n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self._n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        return_attn: bool = False,
    ):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        if self.key_compression is not None:
            k = self.key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = (
                self.key_compression
                if self.value_compression is None
                else self.value_compression
            )(v.transpose(1, 2)).transpose(1, 2)

        batch_size = int(len(q))
        d_head_key = k.shape[-1] // self._n_heads
        d_head_value = v.shape[-1] // self._n_heads
        n_q_tokens = int(q.shape[1])

        q = self._reshape(q)
        k = self._reshape(k)

        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)

        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        if return_attn:
            n_k_tokens = int(attention_probs.shape[-1])
            attn = attention_probs.reshape(
                batch_size, self._n_heads, n_q_tokens, n_k_tokens
            )
            return x, attn

        return x


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError(
                "For the ReGLU activation, the last input dimension must be a multiple of 2,"
                f" however: {x.shape[-1]=}"
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


_TransformerFFNActivation = Literal["ReLU", "ReGLU"]


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        ffn_activation: _TransformerFFNActivation = "ReGLU",
        residual_dropout: float,
        n_tokens: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ) -> None:
        if ffn_activation not in typing.get_args(_TransformerFFNActivation):
            raise ValueError(
                "ffn_activation must be one of"
                f" {typing.get_args(_TransformerFFNActivation)}."
                f" However: {ffn_activation=}"
            )
        if ffn_d_hidden is None:
            if ffn_d_hidden_multiplier is None:
                raise ValueError(
                    "If ffn_d_hidden is None, then ffn_d_hidden_multiplier must not be None"
                )
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
        else:
            if ffn_d_hidden_multiplier is not None:
                raise ValueError(
                    "If ffn_d_hidden is not None, then ffn_d_hidden_multiplier must be None"
                )

        super().__init__()
        ffn_use_reglu = ffn_activation == "ReGLU"

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "att_norm": (
                            nn.LayerNorm(d_block) if layer_idx > 0 else nn.Identity()
                        ),
                        "att": MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens,
                            linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                            linformer_kv_compression_sharing=linformer_kv_compression_sharing,
                        ),
                        "att_drop": nn.Dropout(residual_dropout),
                        "ffn_norm": nn.LayerNorm(d_block),
                        "ffn": _named_sequential(
                            (
                                "linear1",
                                nn.Linear(
                                    d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)
                                ),
                            ),
                            ("activation", _ReGLU() if ffn_use_reglu else nn.ReLU()),
                            ("dropout", nn.Dropout(ffn_dropout)),
                            ("linear2", nn.Linear(ffn_d_hidden, d_block)),
                        ),
                        "ffn_drop": nn.Dropout(residual_dropout),
                    }
                )
                for layer_idx in range(n_blocks)
            ]
        )

        self.output = (
            None
            if d_out is None
            else _named_sequential(
                ("normalization", nn.LayerNorm(d_block)),
                ("activation", nn.ReLU()),
                ("linear", nn.Linear(d_block, d_out)),
            )
        )

    def forward(self, x: Tensor, return_attn: bool = False) -> Dict[str, Tensor | None]:
        if x.ndim != 3:
            raise ValueError(
                f"The input must have exactly three dimension, however: {x.ndim=}"
            )

        attn_list = [] if return_attn else None

        for block in self.blocks:
            block = cast(nn.ModuleDict, block)

            x_identity = x
            x = block["att_norm"](x)

            if return_attn:
                att, att_map = block["att"](x, x, return_attn=True)
                attn_list.append(att_map)
            else:
                att = block["att"](x, x)

            att = block["att_drop"](att)
            x = x_identity + att

            x_identity = x
            x = block["ffn_norm"](x)
            x = block["ffn"](x)
            x = block["ffn_drop"](x)
            x = x_identity + x

        tokens = x
        embedding = tokens[:, 0]
        logits = self.output(embedding) if self.output is not None else None
        return {
            "logits": logits,
            "embedding": embedding,
            "tokens": tokens,
            "attn": attn_list,
        }


class FTTransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int,
        d_block: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: Optional[int] = None,
        ffn_d_hidden_multiplier: Optional[float],
        ffn_dropout: float,
        ffn_activation: _TransformerFFNActivation = "ReGLU",
        residual_dropout: float,
        n_tokens_dec: Optional[int] = None,
        n_tokens_mem: Optional[int] = None,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[
            _LINFORMER_KV_COMPRESSION_SHARING
        ] = None,
    ) -> None:
        if ffn_activation not in typing.get_args(_TransformerFFNActivation):
            raise ValueError(
                "ffn_activation must be one of"
                f" {typing.get_args(_TransformerFFNActivation)}."
                f" However: {ffn_activation=}"
            )
        if ffn_d_hidden is None:
            if ffn_d_hidden_multiplier is None:
                raise ValueError(
                    "If ffn_d_hidden is None, then ffn_d_hidden_multiplier must not be None"
                )
            ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
        else:
            if ffn_d_hidden_multiplier is not None:
                raise ValueError(
                    "If ffn_d_hidden is not None, then ffn_d_hidden_multiplier must be None"
                )

        super().__init__()
        ffn_use_reglu = ffn_activation == "ReGLU"

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_att_norm": nn.LayerNorm(d_block),
                        "self_att": MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens_dec,
                            linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                            linformer_kv_compression_sharing=linformer_kv_compression_sharing,
                        ),
                        "self_att_drop": nn.Dropout(residual_dropout),
                        "cross_att_norm": nn.LayerNorm(d_block),
                        "cross_att": MultiheadAttention(
                            d_embedding=d_block,
                            n_heads=attention_n_heads,
                            dropout=attention_dropout,
                            n_tokens=n_tokens_mem,
                            linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                            linformer_kv_compression_sharing=linformer_kv_compression_sharing,
                        ),
                        "cross_att_drop": nn.Dropout(residual_dropout),
                        "ffn_norm": nn.LayerNorm(d_block),
                        "ffn": _named_sequential(
                            (
                                "linear1",
                                nn.Linear(
                                    d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)
                                ),
                            ),
                            ("activation", _ReGLU() if ffn_use_reglu else nn.ReLU()),
                            ("dropout", nn.Dropout(ffn_dropout)),
                            ("linear2", nn.Linear(ffn_d_hidden, d_block)),
                        ),
                        "ffn_drop": nn.Dropout(residual_dropout),
                    }
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        return_attn: bool = False,
    ) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"decoder x must be 3D, however: {x.ndim=}")
        if memory.ndim != 3:
            raise ValueError(f"decoder memory must be 3D, however: {memory.ndim=}")
        if x.shape[0] != memory.shape[0]:
            raise ValueError("decoder x and memory batch size mismatch")

        self_attn_list = [] if return_attn else None
        cross_attn_list = [] if return_attn else None

        for block in self.blocks:
            block = cast(nn.ModuleDict, block)

            x_identity = x
            x = block["self_att_norm"](x)

            if return_attn:
                att, att_map = block["self_att"](x, x, return_attn=True)
                self_attn_list.append(att_map)
            else:
                att = block["self_att"](x, x)

            att = block["self_att_drop"](att)
            x = x_identity + att

            x_identity = x
            x = block["cross_att_norm"](x)

            if return_attn:
                att, att_map = block["cross_att"](x, memory, return_attn=True)
                cross_attn_list.append(att_map)
            else:
                att = block["cross_att"](x, memory)

            att = block["cross_att_drop"](att)
            x = x_identity + att

            x_identity = x
            x = block["ffn_norm"](x)
            x = block["ffn"](x)
            x = block["ffn_drop"](x)
            x = x_identity + x

        out: Dict[str, Tensor] = {"tokens": x}
        if return_attn:
            out["self_attn"] = cast(Tensor, torch.stack(self_attn_list, dim=0))
            out["cross_attn"] = cast(Tensor, torch.stack(cross_attn_list, dim=0))
        return out


class _CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def forward(self, batch_dims: Tuple[int, ...]) -> Tensor:
        if not batch_dims:
            raise ValueError("The input must be non-empty")
        return self.weight.expand(*batch_dims, 1, -1)


class ReGVAE(nn.Module):
    """
    FTTransformer + (mu, logvar) head + reparameterization

    - encoder: CLS+features -> embedding (CLS token)
    - posterior head: embedding -> (mu, logvar) [B, latent_dim]
    - z sample: reparam(mu, logvar)
    - decoder: learnable queries (F tokens) cross-attend to encoder memory
    - recon head: decoder tokens -> per-feature reconstruction

    반환:
      logits: 분류용 로짓
      embedding: downstream에서 쓰기 위한 대표 벡터 (기본 mu 또는 z)
      z_mu, z_logvar, z: VAE latent 관련
      tokens_enc, tokens_dec, recon: AE 관련
    """

    def __init__(
        self,
        *,
        n_cont_features: int,
        cat_cardinalities: List[int],
        d_out: Optional[int],
        latent_dim: Optional[int] = None,
        logits_from: Literal["mu", "z"] = "mu",
        _is_default: bool = False,
        **backbone_kwargs: Any,
    ) -> None:
        if n_cont_features < 0:
            raise ValueError(
                f"n_cont_features must be non-negative, however: {n_cont_features=}"
            )
        if n_cont_features == 0 and not cat_cardinalities:
            raise ValueError(
                "At least one type of features must be presented, however:"
                f" {n_cont_features=}, {cat_cardinalities=}"
            )
        if "n_tokens" in backbone_kwargs:
            raise ValueError('backbone_kwargs must not contain key "n_tokens"')

        super().__init__()

        d_block: int = int(backbone_kwargs["d_block"])
        self.n_cont_features = int(n_cont_features)
        self.n_cat_features = int(len(cat_cardinalities))
        self._is_default = _is_default

        self.cls_embedding = _CLSEmbedding(d_block)

        self.cont_embeddings = (
            LinearEmbeddings(n_cont_features, d_block) if n_cont_features > 0 else None
        )
        self.cat_embeddings = (
            CategoricalEmbeddings(cat_cardinalities, d_block, True)
            if cat_cardinalities
            else None
        )

        total_feat = self.n_cont_features + self.n_cat_features
        enc_tokens_len = 1 + total_feat
        dec_tokens_len = total_feat

        self.dec_queries = Parameter(torch.empty(total_feat, d_block))
        d_rsqrt = d_block**-0.5
        nn.init.uniform_(self.dec_queries, -d_rsqrt, d_rsqrt)

        lin_ratio = backbone_kwargs.get("linformer_kv_compression_ratio")
        if lin_ratio is None:
            n_tokens_enc = None
            n_tokens_dec = None
            n_tokens_mem = None
        else:
            n_tokens_enc = enc_tokens_len
            n_tokens_dec = dec_tokens_len
            n_tokens_mem = enc_tokens_len

        self.encoder = FTTransformerEncoder(
            d_out=None,
            n_tokens=n_tokens_enc,
            **backbone_kwargs,
        )

        self.decoder = FTTransformerDecoder(
            n_blocks=int(backbone_kwargs["n_blocks"]),
            d_block=int(backbone_kwargs["d_block"]),
            attention_n_heads=int(backbone_kwargs["attention_n_heads"]),
            attention_dropout=float(backbone_kwargs["attention_dropout"]),
            ffn_d_hidden=backbone_kwargs.get("ffn_d_hidden"),
            ffn_d_hidden_multiplier=backbone_kwargs.get("ffn_d_hidden_multiplier"),
            ffn_dropout=float(backbone_kwargs["ffn_dropout"]),
            ffn_activation=cast(
                _TransformerFFNActivation,
                backbone_kwargs.get("ffn_activation", "ReGLU"),
            ),
            residual_dropout=float(backbone_kwargs["residual_dropout"]),
            n_tokens_dec=n_tokens_dec,
            n_tokens_mem=n_tokens_mem,
            linformer_kv_compression_ratio=backbone_kwargs.get(
                "linformer_kv_compression_ratio"
            ),
            linformer_kv_compression_sharing=backbone_kwargs.get(
                "linformer_kv_compression_sharing"
            ),
        )

        self.recon_cont_head = nn.Linear(d_block, 1) if n_cont_features > 0 else None

        if latent_dim is None:
            latent_dim = d_block
        self.latent_dim = int(latent_dim)
        self.logits_from = logits_from

        self.z_mu_head = nn.Linear(d_block, self.latent_dim)
        self.z_logvar_head = nn.Linear(d_block, self.latent_dim)

        if d_out is None:
            self.cls_head = None
        else:
            d_out_int = int(d_out)
            self.cls_head = nn.Sequential(
                nn.Linear(self.latent_dim, d_out_int),
                # nn.Linear(self.latent_dim, self.latent_dim),
                # nn.ReLU(),
            )

        self.feat_gate = FeatGate(self.latent_dim)

    @classmethod
    def get_default_kwargs(cls, n_blocks: int = 3) -> Dict[str, Any]:
        if n_blocks < 0 or n_blocks > 6:
            raise ValueError(
                "Default configurations are available"
                " only for the following values of n_blocks: 1, 2, 3, 4, 5, 6."
                f" However, {n_blocks=}"
            )
        return {
            "n_blocks": n_blocks,
            "d_block": [96, 128, 192, 256, 320, 384][n_blocks - 1],
            "attention_n_heads": 8,
            "attention_dropout": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1],
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1],
            "residual_dropout": 0.0,
            "_is_default": True,
        }

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def encode_only(
        self,
        x_cont: Tensor | None,
        x_cat: Tensor | None,
        return_attn: bool = False,
    ):
        x_any = x_cat if x_cont is None else x_cont
        if x_any is None:
            raise ValueError("At least one of x_cont and x_cat must be provided.")

        feat_tokens: list[Tensor] = []

        if self.cont_embeddings is None:
            if x_cont is not None:
                raise ValueError("x_cont must be None")
        else:
            if x_cont is None:
                raise ValueError("x_cont must not be None")
            feat_tokens.append(self.cont_embeddings(x_cont))

        if self.cat_embeddings is None:
            if x_cat is not None:
                raise ValueError("x_cat must be None")
        else:
            if x_cat is None:
                raise ValueError("x_cat must not be None")
            feat_tokens.append(self.cat_embeddings(x_cat))

        if len(feat_tokens) == 0:
            raise ValueError("No feature tokens were built")

        x_feat_enc = torch.cat(feat_tokens, dim=1)  # (B, F, D)
        x_cls = self.cls_embedding(x_any.shape[:-1])  # (B, 1, D)
        x_enc = torch.cat([x_cls, x_feat_enc], dim=1)  # (B, 1+F, D)

        enc = self.encoder(x_enc, return_attn=return_attn)
        mem = enc["tokens"]
        h = enc["embedding"]
        if mem is None or h is None:
            raise ValueError("encoder outputs are None")

        z_mu = self.z_mu_head(h)
        z_logvar = self.z_logvar_head(h)

        return {
            "embedding": (
                z_mu
                if self.logits_from == "mu"
                else self._reparameterize(z_mu, z_logvar)
            ),
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "tokens_enc": mem,
            "attn_enc": enc["attn"],
        }

    def forward(
        self,
        x_cont: Optional[Tensor],
        x_cat: Optional[Tensor],
        return_attn: bool = False,
    ) -> Dict[str, Tensor | None]:
        x_any = x_cat if x_cont is None else x_cont
        if x_any is None:
            raise ValueError("At least one of x_cont and x_cat must be provided.")

        feat_tokens: List[Tensor] = []

        if self.cont_embeddings is None:
            if x_cont is not None:
                raise ValueError("x_cont must be None")
        else:
            if x_cont is None:
                raise ValueError("x_cont must not be None")
            feat_tokens.append(self.cont_embeddings(x_cont))

        if self.cat_embeddings is None:
            if x_cat is not None:
                raise ValueError("x_cat must be None")
        else:
            if x_cat is None:
                raise ValueError("x_cat must not be None")
            feat_tokens.append(self.cat_embeddings(x_cat))

        if len(feat_tokens) == 0:
            raise ValueError("No feature tokens were built")

        x_feat_enc = torch.cat(feat_tokens, dim=1)  # (B, F, D)
        x_cls = self.cls_embedding(x_any.shape[:-1])  # (B, 1, D)
        x_enc = torch.cat([x_cls, x_feat_enc], dim=1)  # (B, 1+F, D)

        enc = self.encoder(x_enc, return_attn=return_attn)
        mem = enc["tokens"]
        if mem is None:
            raise ValueError("encoder must return tokens as memory")

        h = enc["embedding"]
        if h is None:
            raise ValueError("encoder must return embedding")

        z_mu = self.z_mu_head(h)
        z_logvar = self.z_logvar_head(h)
        z = self._reparameterize(z_mu, z_logvar)

        if self.logits_from == "mu":
            cls_inp = z_mu
            embedding = z_mu
        else:
            cls_inp = z
            embedding = z

        logits = self.cls_head(cls_inp) if self.cls_head is not None else None

        B = int(x_enc.shape[0])
        x_dec_in = self.dec_queries.unsqueeze(0).expand(B, -1, -1)  # (B, F, D)

        dec = self.decoder(x_dec_in, cast(Tensor, mem), return_attn=return_attn)
        tokens_dec = dec["tokens"]

        recon = None
        if self.recon_cont_head is not None:
            s = 0
            e = self.n_cont_features
            cont_tokens = tokens_dec[:, s:e, :]
            recon = self.recon_cont_head(cont_tokens).squeeze(-1)

        return {
            "logits": logits,
            "embedding": embedding,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
            "tokens_enc": mem,
            "tokens_dec": tokens_dec,
            "recon": recon,
            "attn_enc": enc["attn"],
            "self_attn_dec": dec.get("self_attn"),
            "cross_attn_dec": dec.get("cross_attn"),
        }


class FeatGate(nn.Module):
    """
    query(mu_query) + retrieved(mu_retr) -> gated fusion
    fused = g * mu_query + (1-g) * mu_retr
    """

    def __init__(self, d: int, hidden: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden is None:
            hidden = max(64, d)

        self.fc1 = nn.Linear(d * 2, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, mu_query: Tensor, mu_retr: Tensor, return_gate: bool = False):
        x = torch.cat([mu_query, mu_retr], dim=1)
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        g = torch.sigmoid(self.fc2(h))
        fused = g * mu_query + (1.0 - g) * mu_retr

        if return_gate:
            return fused, g

        return fused
