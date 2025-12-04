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

        self.embedder = Embedder(embed_dim)

        self.mpie = MPIEModel(embed_dim, feature_hidden_dims)
        self.mpid = MPIDModel(embed_dim, feature_hidden_dims[::-1])

        mpie_out_dim = feature_hidden_dims[-1]
        self.latent_dim = mpie_out_dim * input_dim
        self.hs_to_memory = nn.Linear(self.latent_dim, decoder_hidden_dim)

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


        # 트랜스포머 디코더
        self.class_embed = nn.Embedding(self.num_tokens, decoder_hidden_dim)
        # 디코더용 위치 임베딩 (최대 horizon+1 길이까지)
        self.pos_embed = nn.Embedding(horizon + 1, decoder_hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_hidden_dim,
            nhead=nhead,
            dim_feedforward=decoder_hidden_dim * 4,
            dropout=0.1,
            batch_first=True,  # (B, L, D)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=total_layer)
        self.decoder_out = nn.Linear(decoder_hidden_dim, num_class)

    def _generate_square_subsequent_mask(
        self,
        sz: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        길이 sz에 대해, 미래 토큰을 보지 못하게 하는 causal mask 생성
        - (sz, sz) 크기
        - 상삼각(대각 위)이 -inf, 나머지는 0
        """
        mask = torch.full((sz, sz), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask


    def forward(
        self,
        x:    torch.Tensor,  # (B, S, F_input) 정도라고 가정
        bemv: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # ===== 인코더 브랜치 =====
        x_emb, bemv_emb = self.embedder(x, bemv)   # 예: (B, S, F, E)
        x_feat = self.mpie(x_emb)                  # (B, S, F, D_enc)

        hs_list = []
        for i in range(self.input_dim):
            xf = x_feat[:, :, i, :]                # (B, S, D_enc)
            h_i = self.sequence_encoders[i](xf)    # (B, S, D_enc)
            hs_list.append(h_i)

        # hs: (B, S, F, D_enc)
        hs = torch.stack(hs_list, dim=2)

        # ===== 재구성 브랜치 =====
        recon = self.mpid(hs)
        recon = self.embedder.decode(recon)

        B, S, F, D_enc = hs.shape

        # ===== hs → latent / memory =====
        # 1) 시간별 통합 임베딩: (B, S, F*D_enc)
        hs_flat = hs.reshape(B, S, F * D_enc)

        # 디코더의 encoder memory로 사용할 시계열: (B, S, D_dec)
        memory = self.hs_to_memory(hs_flat)  # (B, S, decoder_hidden_dim)

        # latent은 XGBoost 등 downstream 용도로 global context로 사용
        # (시간축 평균) → (B, F*D_enc)
        latent = hs_flat.mean(dim=1)         # (B, latent_dim)


        # ===== Transformer Decoder (autoregressive) =====
        logits_list: list[torch.Tensor] = []

        # 시작 토큰: <START>
        y_seq = torch.full(
            (B, 1),
            self.start_idx,
            dtype=torch.long,
            device=x.device,
        )  # (B, 1)

        for t in range(self.horizon):
            L = y_seq.size(1)  # 현재 디코더 입력 시퀀스 길이

            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            tgt = self.class_embed(y_seq) + self.pos_embed(positions)  # (B, L, D_dec)

            tgt_mask = self._generate_square_subsequent_mask(L, x.device)  # (L, L)

            dec_out = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
            )  # (B, L, D_dec)

            step_logits = self.decoder_out(dec_out[:, -1, :])  # (B, num_class)
            logits_list.append(step_logits)

            next_token = torch.argmax(step_logits, dim=-1, keepdim=True)  # (B, 1)
            y_seq = torch.cat([y_seq, next_token], dim=1)  # (B, L+1)

        logits = torch.stack(logits_list, dim=1)  # (B, horizon, num_class)

        return {
            "recon": recon,     # (B, S, F, ?) 형태 (embedder 내부 정의에 따라)
            "logits": logits,   # (B, horizon, num_class)
            "latent": latent,   # (B, latent_dim = F*D_enc)
        }











