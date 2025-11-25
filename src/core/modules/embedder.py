# src/core/modules/embedder.py

import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.encoder = nn.Linear(1, embed_dim)  # embedding
        self.decoder = nn.Linear(embed_dim, 1)  # scalar



    def encode(self, x: torch.Tensor, bemv: torch.Tensor):
        B, S, F = x.shape

        x_exp = x.unsqueeze(-1)                                 # (B,S,F,1)
        emb: torch.Tensor = self.encoder(x_exp)                  # (B,S,F,D)

        bemv_exp = bemv.unsqueeze(-1)                           # (B,S,F,1)
        bemv_emb = bemv_exp.expand(B, S, F, emb.size(-1))

        return emb, bemv_emb


    def decode(self, x: torch.Tensor):
        x_rec = self.decoder(x).squeeze(-1)
        return x_rec


    def forward(self, x: torch.Tensor, bemv: torch.Tensor):
        return self.encode(x, bemv)
