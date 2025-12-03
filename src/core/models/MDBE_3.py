# src/core/models/MDBE_1.py

"""

decoder --> gru


"""

import torch
import torch.nn as nn
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

        self.embedder = Embedder(input_dim, embed_dim)

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

        self.init_bn = nn.BatchNorm1d(input_dim)


    def forward(
        self,
        x:      torch.Tensor,
        bemv:   torch.Tensor,
    ) ->        torch.Tensor:

        # 초기 배치 정규화
        B, S, F = x.shape
        x_flat = x.view(B * S, F)
        x_n_flat = self.init_bn(x_flat)
        x_n = x_n_flat.view(B, S, F)

        x_emb, bemv_emb = self.embedder(x_n, bemv)
        x_feat = self.mpie(x_emb, bemv_emb)
        hs = []

        for i in range(self.input_dim):
            xf = x_feat[:, :, i, :]
            h_i = self.sequence_encoders[i](xf)
            hs.append(h_i)

        hs = torch.stack(hs, dim=2)

        recon = self.mpid(hs)
        recon = self.embedder.decode(recon)

        #
        B = x.size(0)
        latent = hs.mean(dim=1)
        latent = latent.reshape(B, -1)

        h0 = self.latent_to_decoder(latent)
        h_dec = h0.unsqueeze(0).repeat(
            self.decoder_gru.num_layers, 1, 1
        )



        logits_list = []
        y_prev = torch.full(
            (B,),
            self.start_idx,
            dtype=torch.long,
            device=x.device
        )


        for t in range(self.horizon):
            dec_in = self.class_embed(y_prev).unsqueeze(1)
            out, h_dec = self.decoder_gru(dec_in, h_dec)
            logits = self.decoder_out(out.squeeze(1))
            logits_list.append(logits)
            y_prev = torch.argmax(logits, dim=-1)

        logits = torch.stack(logits_list, dim=1)

        return {
            "recon": recon,
            "logits": logits,
            "latent": latent,
        }


















