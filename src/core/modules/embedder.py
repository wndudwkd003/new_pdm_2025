# src/core/modules/embedder.py

import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor, bemv: torch.Tensor):
        B, S, F = x.shape

        x_exp = x.unsqueeze(-1)                                 # (B,S,F,1)
        emb: torch.Tensor = self.linear(x_exp)                  # (B,S,F,D)

        bemv_exp = bemv.unsqueeze(-1)                           # (B,S,F,1)
        bemv_emb = bemv_exp.expand(B, S, F, emb.size(-1))

        return emb, bemv_emb
