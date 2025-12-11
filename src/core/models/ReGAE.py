import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.modules.tabular_encoder import (
    FeatTransformer,
    AttentiveTransformer
)



# --------------------------------------------------------
# ResNet-style MLP
# --------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)


class ResMLP(nn.Module):
    """
    in_dim -> hidden_dim -> (ResBlock x L) -> out_dim
    hidden_dim 에서만 residual connection 이 순환하도록 설계했습니다.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(max(num_layers, 1))]
        )
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        return self.fc_out(h)




class MLP(nn.Module):
    """
    간단한 다층퍼셉트론 블록.
    hidden_dims: [h1, h2, ...]
    마지막 층에는 활성함수를 적용하지 않습니다.
    """
    def __init__(self, in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """
    입력 (x_missing, missing_mask) -> latent embedding
    - x_missing: 결측 위치에는 0 또는 간단한 대치값을 넣어둔 상태
    - missing_mask: 관측=1, 결측=0
    두 값을 concat 해서 인코더에 넣습니다.
    enc_hidden 을 그대로 피라미드 차원으로 사용합니다.
      예: enc_hidden=(256, 256, 128, 128, 64, 32)
        2*input_dim -> 256 -> 256 -> 128 -> 128 -> 64 -> 32 -> (optional) embed_dim
    """
    def __init__(self, input_dim: int, embed_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        in_dim = input_dim * 2  # 값 + 마스크
        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            hidden_dims = [128]

        layers = []
        prev_dim = in_dim
        # 피라미드 다운: 2*input_dim -> h1 -> h2 -> ... -> hL
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        # 마지막 hidden_dim 이 embed_dim 과 다르면 projection 추가
        if prev_dim != embed_dim:
            layers.append(nn.Linear(prev_dim, embed_dim))
            prev_dim = embed_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([x, missing_mask], dim=-1)
        z = self.net(x_in)
        return z



class Decoder(nn.Module):
    """
    latent embedding z -> 재구성된 입력 x_hat
    - 피라미드 업: embed_dim -> h1 -> h2 -> ... -> hL -> output_dim
    - 마지막에 x_skip 을 residual 로 더해 UNet-ish 효과만 유지
    """
    def __init__(self, embed_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            hidden_dims = [128]

        layers = []
        prev_dim = embed_dim
        # 역피라미드 업: embed_dim -> h1 -> h2 -> ... -> hL
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        # 마지막에서 output_dim 으로 투영
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        x_skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_delta = self.net(z)  # 복원해야 할 잔차
        if x_skip is not None:
            return x_skip + x_delta
        return x_delta



class RScoreHead(nn.Module):
    """
    재구성 샘플과 임베딩을 입력받아 R-score(0~1) 예측.
    """
    def __init__(self, input_dim: int, embed_dim: int, hidden_dims=(64,)):
        super().__init__()
        in_dim = input_dim + embed_dim
        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            hidden_dims = [64]
        hidden_dim = hidden_dims[0]
        num_layers = len(hidden_dims)

        self.mlp = ResMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
        )

    def forward(self, x_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_hat, z], dim=-1)
        r = self.mlp(h)
        r = torch.sigmoid(r)  # 0~1 범위로 제한
        return r.squeeze(-1)  # (B,)


class FeatGate(nn.Module):
    """
    TabNet 스타일 feature-selection + embedding gate 를 한 번에 수행하는 모듈.

    1) TabNet-like encoder:
       - x, bemv 를 받아 step별 feature representation(step_outputs),
         entropy 기반 sparsity loss(M_loss), attention mask(tab_mask) 리스트를 생성
    2) 마지막 step 출력(step_outputs[-1])과 (z_curr, z_ret, r-score) 를 concat 해서
       임베딩 차원 게이트 g \in (0,1)^d 생성

        z_gate = g * z_curr + (1 - g) * z_ret
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        n_d: int,
        n_a: int,
        n_shared: int,
        n_independent: int,
        n_steps: int,
        virtual_batch_size: int,
        momentum: float,
        mask_type: str = "sparsemax",
        bias: bool = True,
        epsilon: float = 1e-15,
        gamma: float = 1.5,
        gate_hidden: int = 128,
    ):
        super().__init__()

        # -----------------------------
        # TabNet-style encoder (기존 MPIEncoder 내용에서 msa 부분 제거)
        # -----------------------------
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.n_steps = n_steps
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.bias = bias
        self.epsilon = epsilon
        self.gamma = gamma

        # group_attention_matrix: feature grouping. 일단 I 로 설정
        group_attention_matrix = torch.eye(input_dim)
        self.register_buffer(
            "group_attention_matrix",
            group_attention_matrix.to(torch.float32),
        )
        self.attention_dim = self.group_attention_matrix.shape[0]
        self.feat_output_dim = self.n_d + self.n_a

        self.initial_bn = nn.BatchNorm1d(
            num_features=self.input_dim,
            momentum=self.momentum,
        )

        # shared fully connected layers
        self.shared_feat_transform = nn.ModuleList([
            nn.Linear(
                in_features=self.input_dim if i == 0 else self.feat_output_dim,
                out_features=2 * (self.feat_output_dim),
                bias=self.bias,
            )
            for i in range(self.n_shared)
        ])

        # 첫 split
        self.initial_splitter = FeatTransformer(
            input_dim=self.input_dim,
            output_dim=self.feat_output_dim,
            shared_layers=self.shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
        )

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for _ in range(self.n_steps):
            feat_transformer = FeatTransformer(
                input_dim=self.input_dim,
                output_dim=self.feat_output_dim,
                shared_layers=self.shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )

            attention = AttentiveTransformer(
                input_dim=self.n_a,
                group_dim=self.attention_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                mask_type=self.mask_type,
            )

            self.feat_transformers.append(feat_transformer)
            self.att_transformers.append(attention)

        self.activate = nn.LeakyReLU()

        # -----------------------------
        # embedding gate MLP (ResMLP 사용)
        # input: [z_curr, z_ret, gate_feat, r_curr, r_ret]
        # -----------------------------
        in_dim = embed_dim * 2 + n_d + 2  # z_curr, z_ret, gate_feat, r_curr/r_ret
        self.gate_mlp = ResMLP(
            in_dim=in_dim,
            hidden_dim=gate_hidden,
            out_dim=embed_dim,
            num_layers=2,
        )

    def _run_tabnet(
        self,
        x: torch.Tensor,    # (B, input_dim)
        bemv: torch.Tensor, # (B, input_dim), 현재는 사용하지 않지만 인터페이스 유지
    ):
        """
        순수 TabNet-style encoder:
        x, bemv -> step_outputs, M_loss, attention_masks(tab_mask 리스트)
        """
        # init prior
        prior = torch.ones(
            (x.shape[0], self.attention_dim),
            device=x.device,
            dtype=x.dtype,
        )

        M_loss = 0.0
        step_outputs = []
        attention_maps = []

        # init batch norm
        x_bn = self.initial_bn(x)

        feat_out = self.initial_splitter(x_bn)
        att = feat_out[:, self.n_d:]  # (B, n_a)

        for step in range(self.n_steps):
            # attentive transformer: feature-group mask
            tab_mask = self.att_transformers[step](prior, att)  # (B, attention_dim)

            # sparsity loss (TabNet 과 같은 entropy term)
            M_loss = M_loss + torch.mean(
                torch.sum(tab_mask * torch.log(tab_mask + self.epsilon), dim=1)
            )

            # update prior
            prior = prior * (self.gamma - tab_mask)

            # feature-level mask
            M_feature_level = tab_mask @ self.group_attention_matrix  # (B, input_dim)
            x_masked = M_feature_level * x_bn                         # (B, input_dim)

            # feature transformer
            feat_out = self.feat_transformers[step](x_masked)

            feature_part = feat_out[:, :self.n_d]   # (B, n_d)
            activated = self.activate(feature_part)
            step_outputs.append(activated)

            # attention map(여기서는 tab_mask 자체를 저장)
            attention_maps.append(tab_mask)

            # update attention input
            att = feat_out[:, self.n_d:]

        M_loss = M_loss / self.n_steps

        return step_outputs, M_loss, attention_maps

    def forward(
        self,
        x: torch.Tensor,             # (B, input_dim)
        bemv: torch.Tensor,          # (B, input_dim)
        z_curr: torch.Tensor,        # (B, embed_dim)
        z_ret: torch.Tensor,         # (B, embed_dim)
        r_curr: torch.Tensor | None, # (B,)
        r_ret: torch.Tensor | None,  # (B,),
    ):
        # 1) TabNet-style encoder 실행
        step_outputs, M_loss, attention_maps = self._run_tabnet(x, bemv)
        gate_feat = step_outputs[-1]          # (B, n_d)

        # 2) embedding gate 계산
        parts = [z_curr, z_ret, gate_feat]
        if (r_curr is not None) and (r_ret is not None):
            r_pair = torch.stack([r_curr, r_ret], dim=-1)  # (B, 2)
            parts.append(r_pair)

        h = torch.cat(parts, dim=-1)          # (B, embed_dim*2 + n_d + 2)
        g = torch.sigmoid(self.gate_mlp(h))   # (B, embed_dim)

        z_gate = g * z_curr + (1.0 - g) * z_ret

        return z_gate, g, M_loss, attention_maps


class LabelPriorClassifier(nn.Module):
    """
    - y_ret 가 주어지면: prior(one-hot) + residual(z_gate) 방식을 사용
    - y_ret 가 None 이면: 순수 residual(z_gate) 만 사용 (Stage 1 용)
    """
    def __init__(
        self,
        embed_dim: int | None = None,
        num_classes: int | None = None,
        hidden_dims=(128, 128),
        alpha: float = 1.0,
        input_dim: int | None = None,   # 호환용
        **kwargs,                       # 그 외 keyword 무시
    ):
        super().__init__()

        if embed_dim is None and input_dim is not None:
            embed_dim = input_dim

        if embed_dim is None or num_classes is None:
            raise ValueError(
                "LabelPriorClassifier: embed_dim 과 num_classes 는 반드시 지정해야 합니다."
            )

        hidden_dims = list(hidden_dims)
        if len(hidden_dims) == 0:
            hidden_dims = [128]
        hidden_dim = hidden_dims[0]
        num_layers = len(hidden_dims)

        self.mlp = ResMLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=num_classes,
            num_layers=num_layers,
        )
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(
        self,
        z_gate: torch.Tensor,
        y_ret: torch.Tensor | None = None,
    ) -> torch.Tensor:
        res_logits = self.mlp(z_gate)

        if y_ret is None:
            return res_logits

        prior = F.one_hot(y_ret, num_classes=self.num_classes).float()
        prior_logits = self.alpha * prior
        logits = prior_logits + res_logits
        return logits




class RetrievalGatedAutoEncoder(nn.Module):
    """
    전체 구조를 하나로 묶은 모듈.
    - Stage 1: use_gate=False, retrieved_embedding=None, retrieved_label=None
    - Stage 2: use_gate=True,  retrieved_embedding/label 제공
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_classes: int,
        enc_hidden=(128, 128),
        dec_hidden=(128, 128),
        r_hidden=(64,),
        gate_hidden=(128, 128),
        clf_hidden=(128, 128),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.encoder = Encoder(input_dim, embed_dim, enc_hidden)
        self.decoder = Decoder(embed_dim, input_dim, dec_hidden)
        self.r_head = RScoreHead(input_dim, embed_dim, r_hidden)

        # MPIEncoder 기반 gate 설정 (우선 embed_dim 을 n_d, n_a 로 사용)
        n_d = embed_dim
        n_a = embed_dim
        n_shared = 2
        n_independent = 2
        n_steps = 3
        virtual_batch_size = 128
        momentum = 0.02

        gate_hidden_dim = gate_hidden[0] if isinstance(gate_hidden, (list, tuple)) else gate_hidden

        self.gate = FeatGate(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_shared=n_shared,
            n_independent=n_independent,
            n_steps=n_steps,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            gate_hidden=gate_hidden_dim,
        )


        self.classifier = LabelPriorClassifier(embed_dim, num_classes, clf_hidden)

    def encode(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, missing_mask)

    def decode(self, z: torch.Tensor, x_skip: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, x_skip=x_skip)

    def forward(
        self,
        x_missing: torch.Tensor,             # (B, input_dim)
        missing_mask: torch.Tensor,          # (B, input_dim) 1=관측, 0=결측
        retrieved_embedding: torch.Tensor | None = None,  # (B, embed_dim) 또는 (B, K, embed_dim)
        retrieved_r_score: torch.Tensor | None = None,    # (B,) 또는 (B, K)
        retrieved_label: torch.Tensor | None = None,      # (B,) class index
        use_gate: bool = False,
    ):
        # -------------------------
        # Stage 공통: 기본 인코딩
        # -------------------------
        z_curr = self.encoder(x_missing, missing_mask)  # (B, embed_dim)

        # UNet-ish decoder: x_missing 을 skip 으로 사용
        x_hat_from_curr = self.decoder(z_curr, x_skip=x_missing)
        r_from_curr = self.r_head(x_hat_from_curr, z_curr)  # (B,)

        # 기본값: 게이트 미사용
        z_used = z_curr
        gate_mask = None
        gate_M_loss = None
        gate_attentions = None

        # -------------------------
        # Stage 2: retrieval + gate (classifier 전용)
        # -------------------------
        if use_gate and (retrieved_embedding is not None):
            # 단순화를 위해 top-1 retrieved embedding 이 이미 (B, embed_dim) 라고 가정
            if retrieved_embedding.dim() == 3:
                z_ret = retrieved_embedding[:, 0, :]
                if (retrieved_r_score is not None) and (retrieved_r_score.dim() == 2):
                    r_ret = retrieved_r_score[:, 0]
                else:
                    r_ret = None
            else:
                z_ret = retrieved_embedding
                r_ret = retrieved_r_score

            z_gate, gate_mask, gate_M_loss, gate_attentions = self.gate(
                x=x_missing,
                bemv=missing_mask,
                z_curr=z_curr,
                z_ret=z_ret,
                r_curr=r_from_curr,
                r_ret=r_ret,
            )

            # 이제부터 classifier 는 z_gate 사용
            z_used = z_gate

        # Stage1: retrieved_label 이 없으면 residual-only classifier
        # Stage2: retrieved_label 이 있으면 prior + residual classifier
        logits = self.classifier(z_used, retrieved_label)

        out = {
            "embedding_curr": z_curr,
            "recon_from_curr": x_hat_from_curr,
            "r_from_curr": r_from_curr,
            "embedding_used": z_used,     # gate 사용 여부와 관계 없이 최종 분류용 임베딩
            "logits": logits,
            "gate_mask": gate_mask,       # 임베딩 게이트 g
            "gate_M_loss": gate_M_loss,   # TabNet sparsity term
            "gate_attentions": gate_attentions,
        }
        return out
