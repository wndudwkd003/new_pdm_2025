# src/core/modules/embedder.py

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor


class Embedder(nn.Module):
    """
    built upon /opt/conda/lib/python3.11/site-packages/rtdl_revisiting_models.py
    """
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()

        self.n_features = n_features
        self.embed_dim = embed_dim

        self.enc_weight = Parameter(torch.empty(n_features, embed_dim))   # (F, D)
        self.enc_bias   = Parameter(torch.empty(n_features, embed_dim))   # (F, D)

        self.dec_weight = Parameter(torch.empty(n_features, embed_dim))   # (F, D)
        self.dec_bias   = Parameter(torch.empty(n_features))              # (F,)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embed_dim ** -0.5

        nn.init.uniform_(self.enc_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.enc_bias,   -d_rsqrt, d_rsqrt)

        nn.init.uniform_(self.dec_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.dec_bias,   -d_rsqrt, d_rsqrt)

    def encode(self, x: Tensor, bemv: Tensor):
        B, S, F = x.shape

        emb = x.unsqueeze(-1) * self.enc_weight    # (B, S, F, D)
        emb = emb + self.enc_bias               # (B, S, F, D)

        # bemv 임베딩 공간 확장
        bemv_exp = bemv.unsqueeze(-1)                        # (B, S, F, 1)
        bemv_emb = bemv_exp.expand(B, S, F, self.embed_dim)  # (B, S, F, D)

        return emb, bemv_emb

    def decode(self, x: Tensor) -> Tensor:
        B, S, F, D = x.shape

        x_proj = (x * self.dec_weight).sum(dim=-1)          # (B, S, F)
        x_rec = x_proj + self.dec_bias.view(1, 1, F)        # (B, S, F)

        return x_rec

    def forward(self, x: Tensor, bemv: Tensor):
        return self.encode(x, bemv)
