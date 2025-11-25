import torch
import torch.nn as nn
from typing import Sequence


class MPIEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int]
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)



        self.init_ln = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        bemv: torch.Tensor
    ) -> torch.Tensor:

        # B, S, F, D







        h = self.mlp(x)
        return h
