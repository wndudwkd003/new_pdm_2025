# src/core/models/MDBE_1.py

"""

decoder --> gru


"""

import torch
import torch.nn as nn


class HybridDoubleBranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        feature_hidden_dims: list[int],
        num_class: int,
        nhead: int,
        transformer_layers: int,
        decoder_hidden_dims: int,
        total_layer: int,
    ):

        super().__init__()

    def forward(
        self,
        x:      torch.Tensor,
        bemv:   torch.Tensor,
    ) ->        torch.Tensor:
        pass


