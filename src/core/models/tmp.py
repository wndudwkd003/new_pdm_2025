import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Utils
# =========================================================

def match_len_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    x: (B, C, L)
    """
    L = x.size(-1)
    if L > target_len:
        return x[..., :target_len]
    if L < target_len:
        return F.pad(x, (0, target_len - L))
    return x


# =========================================================
# 1D UNet blocks (Stable-Diffusion-like: Res + Attn between down/up)
# =========================================================

class ResConv1DBlock(nn.Module):
    """
    x: (B, C, L)
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        h = self.drop(h)
        return self.act(x + h)


class MaskedSelfAttention1D(nn.Module):
    """
    Self-attn along length L on (B, C, L).

    mode:
      - "causal":   삼각형 (미래 차단)  => encoder side
      - "anti":     역삼각형 (과거 차단) => decoder side
      - None:       full attention
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0, mode: str | None = None):
        super().__init__()
        self.mode = mode
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def _build_mask(self, L: int, device: torch.device) -> torch.Tensor | None:
        if self.mode is None:
            return None

        m = torch.full((L, L), float("-inf"), device=device)

        if self.mode == "causal":
            # upper triangular: i attends only <= i
            return torch.triu(m, diagonal=1)

        if self.mode == "anti":
            # lower triangular: i attends only >= i
            return torch.tril(m, diagonal=-1)

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C)
        h = x.transpose(1, 2)
        L = h.size(1)
        attn_mask = self._build_mask(L, h.device)

        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ff(h))

        return h.transpose(1, 2)  # (B, C, L)


class Downsample1D(nn.Module):
    """
    stride-2 conv downsample
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    nearest upsample + conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# =========================================================
# UNet Attention Encoder / Decoder
# =========================================================

class UNetAttnEncoder(nn.Module):
    """
    Input: x_missing(B,F), missing_mask(B,F)
    Tokenize each feature with [x_i, m_i] -> C0
    Then 1D UNet down path with (Res -> Attn(causal) -> Down).
    Bottleneck also Res/Attn(causal)/Res.

    Returns:
      z: (B, embed_dim)
      skips: list of (B, C_i, L_i)
      bottleneck: (B, C_last, L_last)
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        channels: tuple[int, ...],   # e.g. (128, 64, 64, 32)
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.channels = list(channels)

        self.token_embed = nn.Linear(2, self.channels[0])  # [x_i, m_i] -> C0

        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(len(self.channels) - 1):
            c = self.channels[i]
            c_next = self.channels[i + 1]
            self.down_res.append(ResConv1DBlock(c, dropout=dropout))
            self.down_attn.append(MaskedSelfAttention1D(c, num_heads=num_heads, dropout=dropout, mode="causal"))
            self.down.append(Downsample1D(c, c_next))

        c_mid = self.channels[-1]
        self.mid_res1 = ResConv1DBlock(c_mid, dropout=dropout)
        self.mid_attn = MaskedSelfAttention1D(c_mid, num_heads=num_heads, dropout=dropout, mode="causal")
        self.mid_res2 = ResConv1DBlock(c_mid, dropout=dropout)

        self.to_z = nn.Linear(c_mid, embed_dim)

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor):
        B, F = x.size()

        tok = torch.stack([x, missing_mask], dim=-1)     # (B,F,2)
        h = self.token_embed(tok).transpose(1, 2)        # (B,C0,F)

        skips = []
        for res, attn, down in zip(self.down_res, self.down_attn, self.down):
            h = res(h)
            h = attn(h)
            skips.append(h)
            h = down(h)

        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)

        pooled = h.mean(dim=-1)          # (B, C_mid)
        z = self.to_z(pooled)            # (B, embed_dim)

        return z, skips, h


class UNetAttnDecoder(nn.Module):
    """
    Decoder: start from bottleneck, add z-conditioning, then
      Up -> concat skip -> 1x1 merge -> Res -> Attn(anti)

    Output: x_hat (B,F) or x_delta added to x_skip.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        channels: tuple[int, ...],   # same as encoder channels
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.channels = list(channels)

        self.z_proj = nn.Linear(embed_dim, self.channels[-1])

        rev = list(reversed(self.channels))  # e.g. [32,64,64,128]

        self.ups = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()

        for i in range(len(rev) - 1):
            c = rev[i]
            c_next = rev[i + 1]

            self.ups.append(Upsample1D(c, c_next))

            # concat(h_up, skip) where both are c_next channels => 2*c_next -> c_next
            self.merges.append(nn.Conv1d(c_next + c_next, c_next, kernel_size=1))

            self.up_res.append(ResConv1DBlock(c_next, dropout=dropout))
            self.up_attn.append(MaskedSelfAttention1D(c_next, num_heads=num_heads, dropout=dropout, mode="anti"))

        self.out_conv = nn.Conv1d(self.channels[0], 1, kernel_size=1)

    def forward(
        self,
        z: torch.Tensor,
        skips: list[torch.Tensor],
        bottleneck: torch.Tensor,
        x_skip: torch.Tensor | None = None,
    ):
        h = bottleneck  # (B, C_mid, L_mid)
        h = h + self.z_proj(z).unsqueeze(-1)

        for up, merge, res, attn, s in zip(self.ups, self.merges, self.up_res, self.up_attn, reversed(skips)):
            h = up(h)
            h = match_len_1d(h, s.size(-1))
            h = torch.cat([h, s], dim=1)
            h = merge(h)
            h = res(h)
            h = attn(h)

        x_delta = self.out_conv(h).squeeze(1)  # (B, L_out)
        x_delta = match_len_1d(x_delta.unsqueeze(1), self.input_dim).squeeze(1)  # (B, F)

        if x_skip is not None:
            return x_skip + x_delta
        return x_delta


# =========================================================
# FeatGate: triangular transformer on interleaved (z_curr, z_ret) along 2D length
# =========================================================

class TriangularTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        ff_dim = hidden_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_out, attn_weights = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


class FeatGate(nn.Module):
    """
    z_curr(B,D), z_ret(B,D)
    -> interleave to length 2D tokens
    -> triangular attention
    -> pair-wise gate M(B,D)
    -> z_gate = M*z_curr + (1-M)*z_ret
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        ff_mult: int = 4,
        pair_proj_dim: int = 128,  # 128 확장 후 반반(att/feat)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.scalar_embed = nn.Linear(1, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * embed_dim, hidden_dim))

        self.blocks = nn.ModuleList([
            TriangularTransformerBlock(hidden_dim, num_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pair_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, pair_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

        att_dim = pair_proj_dim // 2
        self.att_to_gate = nn.Linear(att_dim, 1)

    def _tri_mask(self, L: int, device):
        m = torch.full((L, L), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, z_curr: torch.Tensor, z_ret: torch.Tensor):
        B, D = z_curr.size()
        L = 2 * D

        seq = torch.stack([z_curr, z_ret], dim=2).reshape(B, L)  # (B,2D)
        x = self.scalar_embed(seq.unsqueeze(-1))                 # (B,2D,H)
        x = x + self.pos_emb[:, :L, :]

        attn_mask = self._tri_mask(L, x.device)

        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x, attn_mask=attn_mask)
            attn_list.append(attn)

        curr_ctx = x[:, 0::2, :]  # (B,D,H)
        ret_ctx  = x[:, 1::2, :]  # (B,D,H)

        pair = torch.cat([curr_ctx, ret_ctx], dim=-1)    # (B,D,2H)
        pair_128 = self.pair_proj(pair)                  # (B,D,128)

        att_half = pair_128[:, :, : pair_128.size(-1)//2]        # (B,D,64)
        gate_logits = self.att_to_gate(att_half).squeeze(-1)     # (B,D)
        M = torch.sigmoid(gate_logits)

        z_gate = M * z_curr + (1.0 - M) * z_ret

        attn_mean = attn_list[-1].mean(dim=1)  # (B, 2D, 2D)
        return z_gate, M, attn_mean


# =========================================================
# Attention Classifier (FT-Transformer-like on latent dims)
# =========================================================

class AttnLabelPriorClassifier(nn.Module):
    """
    z_gate(B,D_embed)를 길이 D_embed 토큰으로 보고 Transformer Encoder 적용.
    [CLS] 풀링 -> logits.
    y_ret 있으면 prior(one-hot)*alpha + residual.
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        alpha: float = 1.0,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.scalar_embed = nn.Linear(1, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, 1 + embed_dim, token_dim))
        else:
            self.pos_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, num_classes)

    def forward(self, z_gate: torch.Tensor, y_ret: torch.Tensor | None = None) -> torch.Tensor:
        B, D = z_gate.size()

        x = self.scalar_embed(z_gate.unsqueeze(-1))  # (B,D,T)
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,T)
        h = torch.cat([cls, x], dim=1)               # (B,1+D,T)

        if self.pos_emb is not None:
            h = h + self.pos_emb[:, : (1 + D), :]

        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])
        res_logits = self.head(cls_out)

        if y_ret is None:
            return res_logits

        prior = F.one_hot(y_ret, num_classes=self.num_classes).float()
        return self.alpha * prior + res_logits


# =========================================================
# Attention RScore Head
# =========================================================

class AttnRScoreHead(nn.Module):
    """
    x_hat(B,F) feature tokens + z(B,D) token + [CLS] -> Transformer -> r in [0,1]
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        token_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.x_embed = nn.Linear(1, token_dim)
        self.z_embed = nn.Linear(embed_dim, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, 2 + input_dim, token_dim))
        else:
            self.pos_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1)

    def forward(self, x_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, F = x_hat.size()

        x_tok = self.x_embed(x_hat.unsqueeze(-1))     # (B,F,T)
        z_tok = self.z_embed(z).unsqueeze(1)          # (B,1,T)
        cls = self.cls_token.expand(B, -1, -1)        # (B,1,T)

        h = torch.cat([cls, z_tok, x_tok], dim=1)     # (B,2+F,T)
        if self.pos_emb is not None:
            h = h + self.pos_emb[:, : (2 + F), :]

        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])
        r = torch.sigmoid(self.head(cls_out)).squeeze(-1)
        return r


# =========================================================
# Final Model: RetrievalGatedAutoEncoder
# =========================================================

class RetrievalGatedAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        enc_hidden: tuple[int, ...],   # UNet channels
        r_hidden: tuple[int, ...],     # rscore token dim + layers
        gate_hidden: tuple[int, ...],  # gate hidden dim
        clf_hidden: tuple[int, ...],   # classifier token dim + layers
        dropout: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # UNet encoder/decoder
        self.encoder = UNetAttnEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            channels=enc_hidden,
            num_heads=4,
            dropout=dropout,
        )
        self.decoder = UNetAttnDecoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            channels=enc_hidden,
            num_heads=4,
            dropout=dropout,
        )

        # rscore head
        r_token_dim = r_hidden[0]
        r_layers = len(r_hidden)
        self.r_head = AttnRScoreHead(
            input_dim=input_dim,
            embed_dim=embed_dim,
            token_dim=r_token_dim,
            num_layers=r_layers,
            num_heads=4,
            dropout=dropout,
            use_pos_emb=True,
        )

        # classifier
        clf_token_dim = clf_hidden[0]
        clf_layers = len(clf_hidden)
        self.classifier = AttnLabelPriorClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            token_dim=clf_token_dim,
            num_layers=clf_layers,
            num_heads=4,
            dropout=dropout,
            alpha=1.0,
            use_pos_emb=True,
        )

        # feat gate
        gate_dim = gate_hidden[0]
        self.feat_gate = FeatGate(
            embed_dim=embed_dim,
            hidden_dim=gate_dim,
            num_layers=3,
            num_heads=4,
            dropout=dropout,
            ff_mult=4,
            pair_proj_dim=128,
        )

    # ---- Stage2 compatibility ----
    def encode(self, x_missing: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.encoder(x_missing, missing_mask)
        return z

    # (원하시면 adapter에서 더 쓰기 좋게)
    def reconstruct(self, x_missing: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        z, skips, bottleneck = self.encoder(x_missing, missing_mask)
        return self.decoder(z, skips=skips, bottleneck=bottleneck, x_skip=x_missing)

    def forward(
        self,
        x_missing: torch.Tensor,
        missing_mask: torch.Tensor,
        retrieved_embedding: torch.Tensor | None = None,
        retrieved_r_score: torch.Tensor | None = None,
        retrieved_label: torch.Tensor | None = None,
        use_gate: bool = False,
        return_logits: bool = True,
    ):
        z_curr, skips, bottleneck = self.encoder(x_missing, missing_mask)

        x_hat_from_curr = self.decoder(z_curr, skips=skips, bottleneck=bottleneck, x_skip=x_missing)
        r_from_curr = self.r_head(x_hat_from_curr, z_curr)

        z_used = z_curr
        gate_mask = None
        gate_attentions = None
        gate_M_loss = None

        if use_gate and (retrieved_embedding is not None):
            if retrieved_embedding.dim() == 3:
                z_ret = retrieved_embedding[:, 0, :]
            else:
                z_ret = retrieved_embedding

            z_gate, M, attn = self.feat_gate(z_curr, z_ret)
            z_used = z_gate
            gate_mask = M
            gate_attentions = attn

        logits = None
        if return_logits:
            logits = self.classifier(z_used, retrieved_label)

        return {
            "embedding_curr": z_curr,
            "recon_from_curr": x_hat_from_curr,
            "r_from_curr": r_from_curr,
            "embedding_used": z_used,
            "logits": logits,
            "gate_mask": gate_mask,
            "gate_M_loss": gate_M_loss,
            "gate_attentions": gate_attentions,
        }
