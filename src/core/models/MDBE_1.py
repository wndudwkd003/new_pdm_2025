# src/core/models/MDBE_1.py

"""

decoder --> gru


"""

import torch
import torch.nn as nn

from core.modules.decoders import MPIDModel
from src.core.modules.embedder import Embedder
from src.core.modules.encoders import MPIEModel
from src.core.modules.decoders import MPIDModel


class HybridDoubleBranchEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        feature_hidden_dims: list[int],
        num_class: int,
        nhead: int,
        transformer_layers: int,
        decoder_hidden_dim: int,
        total_layer: int,
        horizon: int,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.horizon = horizon
        self.num_class = num_class
        self.num_tokens = num_class + 1
        self.start_idx = num_class

        self.embedder = Embedder(embed_dim)

        self.mpie = MPIEModel(embed_dim, feature_hidden_dims)
        self.mpid = MPIDModel(embed_dim, feature_hidden_dims[::-1])

        mpie_out_dim = feature_hidden_dims[-1]
        self.latent_dim = mpie_out_dim * input_dim

        self.sequence_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=mpie_out_dim,
                    nhead=nhead,
                    dim_feedforward=mpie_out_dim * 4,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=transformer_layers,
            )
            for _ in range(input_dim)
        ])

        self.latent_to_decoder = nn.Linear(self.latent_dim, decoder_hidden_dim)

        self.class_embed = nn.Embedding(self.num_tokens, decoder_hidden_dim)

        self.decoder_gru = nn.GRU(
            decoder_hidden_dim,
            decoder_hidden_dim,
            total_layer,
            batch_first=True
        )

        self.decoder_out = nn.Linear(decoder_hidden_dim, num_class)


    def forward(
        self,
        x:      torch.Tensor,
        bemv:   torch.Tensor,
    ) ->        torch.Tensor:

        x_emb, bemv_emb = self.embedder(x, bemv)

        h = self.mpie(x_emb)

        hs = []

        for i in range(self.input_dim):
            xf = h[:, :, i, :]
            h = self.sequence_encoders[i](xf)
            hs.append(h)

        hs = torch.stack(hs, dim=2)

        recon = self.mpid(hs)
        recon = self.embedder.decode(recon)

        #
        B = x.size(0)
        latent = hs.mean(dim=1)
        latent = latent.reshape(B, -1)
        h = self.latent_to_decoder(latent).unsqueeze(0)

        logits_list = []

        y_prev = torch.full(
            (B,),
            self.start_idx,
            dtype=torch.long,
            device=x.device
        )


        for t in range(self.horizon):
            dec_in = self.class_embed(y_prev).unsqueeze(1)
            out, h = self.decoder_gru(dec_in, h)

            logits = self.decoder_out(out.squeeze(1))
            logits_list.append(logits)


            y_prev = torch.argmax(logits, dim=-1)

        preds = torch.stack(logits_list, dim=1)

        return {
            "recon": recon,
            "preds": preds,
            "latent": latent,
        }


















