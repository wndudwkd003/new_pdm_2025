import torch
import torch.nn as nn
from typing import Sequence

class MPIDModel(nn.Module):
    def __init__(self, out_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor):
        emb = self.mlp(h)
        return emb
