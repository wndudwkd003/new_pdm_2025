# src/models/saint_adapter.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core.utils.losses import info_nce_loss
from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split


# -----------------------------
# SAINT model components
# -----------------------------


class _ContinuousFeatureEmbedder(nn.Module):
    """
    SAINT: 연속형 feature마다 (1 -> d_token) FC + ReLU를 독립적으로 적용.
    여기서는 per-feature affine로 동일 효과를 구현:
      e_{i,j} = ReLU(x_{i,j} * W_j + b_j),  W_j,b_j in R^{d_token}
    """

    def __init__(self, n_cont_features: int, d_token: int):
        super().__init__()
        self.n_cont_features = int(n_cont_features)
        self.d_token = int(d_token)

        self.weight = nn.Parameter(torch.empty(self.n_cont_features, self.d_token))
        self.bias = nn.Parameter(torch.empty(self.n_cont_features, self.d_token))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        # x_cont: (B, F)
        if x_cont.dim() != 2:
            raise ValueError(f"x_cont must be 2D (B,F). got: {tuple(x_cont.shape)}")

        x = x_cont.unsqueeze(-1)  # (B, F, 1)
        w = self.weight.unsqueeze(0)  # (1, F, D)
        b = self.bias.unsqueeze(0)  # (1, F, D)
        out = x * w + b  # (B, F, D)
        out = F.relu(out)
        return out


class _ProjectionHead(nn.Module):
    """
    SAINT: projection head = hidden 1-layer MLP + ReLU (여기서는 2층으로 구현)
    """

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class _FeatureReconMLP(nn.Module):
    """
    SAINT: feature별로 MLP_j (hidden 1-layer + ReLU)로 원 feature 복원 (연속형 scalar 출력)
    """

    def __init__(self, d_token: int):
        super().__init__()
        self.fc1 = nn.Linear(d_token, d_token)
        self.fc2 = nn.Linear(d_token, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, D)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t).squeeze(-1)  # (B,)
        return t


class _SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        num_heads: int,
        attn_dropout: float,
        ff_mult: int,
        ff_dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(attn_dropout)

        self.ln2 = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, ff_mult * d_token),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_mult * d_token, d_token),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class _IntersampleAttentionBlock(nn.Module):
    """
    SAINT Algorithm 1:
      x: (B, N, D)
      reshape -> (1, B, N*D) 로 만든 뒤 "sample 축(B)"에 self-attn 수행
      reshape back -> (B, N, D)
    """

    def __init__(
        self,
        n_tokens: int,
        d_token: int,
        num_heads: int,
        attn_dropout: float,
        ff_mult: int,
        ff_dropout: float,
    ):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.d_token = int(d_token)
        self.flat_dim = self.n_tokens * self.d_token

        if self.flat_dim % int(num_heads) != 0:
            raise ValueError(
                f"intersample attention requires (n_tokens*d_token) % num_heads == 0. "
                f"got flat_dim={self.flat_dim}, num_heads={int(num_heads)}"
            )

        self.ln_flat = nn.LayerNorm(self.flat_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.flat_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(attn_dropout)

        self.ln_tok = nn.LayerNorm(self.d_token)
        self.ff = nn.Sequential(
            nn.Linear(self.d_token, ff_mult * self.d_token),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_mult * self.d_token, self.d_token),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        b, n, d = x.shape
        if n != self.n_tokens or d != self.d_token:
            raise ValueError(
                f"shape mismatch. expected (B,{self.n_tokens},{self.d_token}), got {tuple(x.shape)}"
            )

        x_flat = x.reshape(b, self.flat_dim)  # (B, N*D)
        seq = x_flat.unsqueeze(0)  # (1, B, N*D)

        h = self.ln_flat(seq)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        seq = seq + self.drop1(attn_out)

        x = seq.squeeze(0).reshape(b, n, d)

        h2 = self.ln_tok(x)
        x = x + self.ff(h2)
        return x


class SAINT(nn.Module):
    def __init__(
        self,
        n_cont_features: int,
        d_token: int,
        num_heads: int,
        num_layers: int,
        d_out: int,
        use_self_attention: bool = True,
        use_intersample_attention: bool = True,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        ff_mult: int = 4,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.n_cont_features = int(n_cont_features)
        self.d_token = int(d_token)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.d_out = int(d_out)

        self.use_self_attention = bool(use_self_attention)
        self.use_intersample_attention = bool(use_intersample_attention)

        self.embedder = _ContinuousFeatureEmbedder(self.n_cont_features, self.d_token)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        n_tokens = self.n_cont_features + 1  # CLS + features

        blocks: list[nn.Module] = []
        for _ in range(self.num_layers):
            if self.use_self_attention:
                blocks.append(
                    _SelfAttentionBlock(
                        d_token=self.d_token,
                        num_heads=self.num_heads,
                        attn_dropout=attn_dropout,
                        ff_mult=ff_mult,
                        ff_dropout=ff_dropout,
                    )
                )
            if self.use_intersample_attention:
                blocks.append(
                    _IntersampleAttentionBlock(
                        n_tokens=n_tokens,
                        d_token=self.d_token,
                        num_heads=self.num_heads,
                        attn_dropout=attn_dropout,
                        ff_mult=ff_mult,
                        ff_dropout=ff_dropout,
                    )
                )
        self.encoder = nn.Sequential(*blocks)

        # supervised head (CLS -> logits)
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_token, self.d_token),
            nn.ReLU(),
            nn.Linear(self.d_token, self.d_out),
        )

        # projection heads for contrastive pre-training
        self.proj_head_1 = _ProjectionHead(self.d_token, int(proj_dim))
        self.proj_head_2 = _ProjectionHead(self.d_token, int(proj_dim))

        # denoising recon MLP per feature (continuous)
        self.recon_mlps = nn.ModuleList(
            [_FeatureReconMLP(self.d_token) for _ in range(self.n_cont_features)]
        )

    def embed(self, x_cont: torch.Tensor) -> torch.Tensor:
        # x_cont: (B, F)
        b, f = x_cont.shape
        if f != self.n_cont_features:
            raise ValueError(
                f"n_cont_features mismatch. expected {self.n_cont_features}, got {f}"
            )

        feat_tokens = self.embedder(x_cont)  # (B, F, D)
        cls = self.cls_token.expand(b, 1, self.d_token)  # (B,1,D)
        tokens = torch.cat([cls, feat_tokens], dim=1)  # (B, F+1, D)
        return tokens

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)
        ctx = self.encoder(tokens)
        return ctx

    def classify_from_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        cls = ctx[:, 0, :]  # (B, D)
        logits = self.cls_head(cls)  # (B, C)
        return logits

    def project_view1(self, ctx: torch.Tensor) -> torch.Tensor:
        cls = ctx[:, 0, :]
        z = self.proj_head_1(cls)
        return z

    def project_view2(self, ctx: torch.Tensor) -> torch.Tensor:
        cls = ctx[:, 0, :]
        z = self.proj_head_2(cls)
        return z

    def reconstruct_from_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: (B, F+1, D) -> use feature tokens only
        feat_ctx = ctx[:, 1:, :]  # (B, F, D)
        outs: list[torch.Tensor] = []
        for j in range(self.n_cont_features):
            xj = self.recon_mlps[j](feat_ctx[:, j, :])  # (B,)
            outs.append(xj.unsqueeze(1))
        x_recon = torch.cat(outs, dim=1)  # (B, F)
        return x_recon

    def forward(
        self,
        x_cont: torch.Tensor,
        need_logits: bool = True,
        need_recon: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        tokens = self.embed(x_cont)
        ctx = self.encode_tokens(tokens)

        logits = None
        recon = None

        if need_logits:
            logits = self.classify_from_ctx(ctx)

        if need_recon:
            recon = self.reconstruct_from_ctx(ctx)

        return {
            "ctx": ctx,
            "logits": logits,
            "recon": recon,
        }


# -----------------------------
# SAINT Adapter (Stage1 -> Stage2 고정)
# -----------------------------


class SAINTAdapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)

        self.model: SAINT | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        model_cfg = self.config.model

        # unified names (configs.py 기준)
        self.lambda_cls = float(model_cfg.lambda_cls)
        self.lambda_view = float(model_cfg.lambda_view)
        self.lambda_recon = float(model_cfg.lambda_recon)

        # SAINT paper default: tau=0.7, pcutmix=0.3, alpha=0.2
        if hasattr(self.config.train, "temperature"):
            self.temperature = float(self.config.train.temperature)
        else:
            self.temperature = 0.7

        if hasattr(model_cfg, "saint_p_cutmix"):
            self.p_cutmix = float(model_cfg.saint_p_cutmix)
        else:
            self.p_cutmix = 0.3

        if hasattr(model_cfg, "saint_mixup_alpha"):
            self.mixup_alpha = float(model_cfg.saint_mixup_alpha)
        else:
            self.mixup_alpha = 0.2

    # -------------------------
    # Fit (Stage1 -> Stage2)
    # -------------------------
    def fit(self, train_data: Datasets, valid_data: Datasets):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim = int(train_data.meta.input_dim)
        self.num_class = int(train_data.meta.num_class)

        self.model = self._get_model(self.input_dim, self.num_class)

        stage1_epochs = int(self.config.train.epochs)
        stage2_epochs = int(self.config.train.epochs)
        max_patience = int(self.config.train.early_stopping_rounds)

        # -------------------------
        # Stage1 logs
        # -------------------------
        train_total_1: list[float] = []
        valid_total_1: list[float] = []
        train_ce_1: list[float] = []
        valid_ce_1: list[float] = []
        train_info_1: list[float] = []
        valid_info_1: list[float] = []
        train_recon_1: list[float] = []
        valid_recon_1: list[float] = []

        # -------------------------
        # Stage2 logs
        # -------------------------
        train_total_2: list[float] = []
        valid_total_2: list[float] = []
        train_ce_2: list[float] = []
        valid_ce_2: list[float] = []
        train_info_2: list[float] = []
        valid_info_2: list[float] = []
        train_recon_2: list[float] = []
        valid_recon_2: list[float] = []

        # -------------------------
        # Stage 1: contrastive + denoising (clean x_originals 기반)
        # -------------------------
        opt1, sch1 = self._make_optimizer_scheduler(stage=1, num_epochs=stage1_epochs)

        best_valid_1 = None
        best_state_1 = None
        patience_1 = 0

        for epoch in range(stage1_epochs):
            tr = self.run_epoch(tr_loader, opt1, Split.TRAIN, stage=1)
            vl = self.run_epoch(vl_loader, None, Split.VALID, stage=1)

            lr = float(opt1.param_groups[0]["lr"])
            print(
                f"[SAINT Stage1 Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, info={tr['info']:.4f}, recon={tr['recon']:.4f} | "
                f"Valid: total={vl['total']:.4f}, info={vl['info']:.4f}, recon={vl['recon']:.4f} | "
                f"LR: {lr:.6f}"
            )

            train_total_1.append(tr["total"])
            valid_total_1.append(vl["total"])
            train_ce_1.append(tr["ce"])
            valid_ce_1.append(vl["ce"])
            train_info_1.append(tr["info"])
            valid_info_1.append(vl["info"])
            train_recon_1.append(tr["recon"])
            valid_recon_1.append(vl["recon"])

            sch1.step()

            if best_valid_1 is None or vl["total"] < best_valid_1:
                best_valid_1 = vl["total"]
                patience_1 = 0
                best_state_1 = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience_1 += 1
                if patience_1 >= max_patience:
                    print(f"[SAINT] Stage1 Early stopping at epoch {epoch + 1}")
                    break

        if best_state_1 is not None:
            self.model.load_state_dict(best_state_1)
            self.model.to(self.device)

        # -------------------------
        # Stage 2: supervised finetuning (missing x 기반)
        # -------------------------
        opt2, sch2 = self._make_optimizer_scheduler(stage=2, num_epochs=stage2_epochs)

        best_valid_2 = None
        best_state_2 = None
        patience_2 = 0

        for epoch in range(stage2_epochs):
            tr2 = self.run_epoch(tr_loader, opt2, Split.TRAIN, stage=2)
            vl2 = self.run_epoch(vl_loader, None, Split.VALID, stage=2)

            lr2 = float(opt2.param_groups[0]["lr"])
            print(
                f"[SAINT Stage2 Epoch {epoch + 1}] "
                f"Train: total={tr2['total']:.4f}, ce={tr2['ce']:.4f} | "
                f"Valid: total={vl2['total']:.4f}, ce={vl2['ce']:.4f} | "
                f"LR: {lr2:.6f}"
            )

            train_total_2.append(tr2["total"])
            valid_total_2.append(vl2["total"])
            train_ce_2.append(tr2["ce"])
            valid_ce_2.append(vl2["ce"])
            train_info_2.append(tr2["info"])
            valid_info_2.append(vl2["info"])
            train_recon_2.append(tr2["recon"])
            valid_recon_2.append(vl2["recon"])

            sch2.step()

            if best_valid_2 is None or vl2["total"] < best_valid_2:
                best_valid_2 = vl2["total"]
                patience_2 = 0
                best_state_2 = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience_2 += 1
                if patience_2 >= max_patience:
                    print(f"[SAINT] Stage2 Early stopping at epoch {epoch + 1}")
                    break

        if best_state_2 is not None:
            self.model.load_state_dict(best_state_2)
            self.model.to(self.device)

        # -------------------------
        # Final metrics (Stage2 기준)
        # -------------------------
        _, tr_preds, tr_labels, _, _ = self.predict(
            tr_loader, split=Split.TRAIN, stage=2
        )
        _, vl_preds, vl_labels, _, _ = self.predict(
            vl_loader, split=Split.VALID, stage=2
        )

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        metric_name = "total_loss"
        tasks = [
            {Split.TRAIN.value: train_total_1, Split.VALID.value: valid_total_1},
            {Split.TRAIN.value: train_total_2, Split.VALID.value: valid_total_2},
        ]

        components = {
            "stage1": {
                "train": {
                    "ce": train_ce_1,
                    "info": train_info_1,
                    "recon": train_recon_1,
                    "total": train_total_1,
                },
                "valid": {
                    "ce": valid_ce_1,
                    "info": valid_info_1,
                    "recon": valid_recon_1,
                    "total": valid_total_1,
                },
            },
            "stage2": {
                "train": {
                    "ce": train_ce_2,
                    "info": train_info_2,
                    "recon": train_recon_2,
                    "total": train_total_2,
                },
                "valid": {
                    "ce": valid_ce_2,
                    "info": valid_info_2,
                    "recon": valid_recon_2,
                    "total": valid_total_2,
                },
            },
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": metric_name,
                "tasks": tasks,
                "components": components,
            },
        }
        return results

    # -------------------------
    # Train loop (stage=1/2 통합)
    # -------------------------
    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
        stage: int = 1,
    ):
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(f"SAINT_stage{stage}", split)

        total_sum = 0.0
        ce_sum = 0.0
        info_sum = 0.0
        recon_sum = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x_missing, y, x_clean, _, _, _ = self._prepare_batch(batch)

            if stage == 1:
                # Stage1은 clean(x_originals) 사용
                xb = x_clean
                b, f = xb.shape

                with torch.set_grad_enabled(is_train):
                    # view1 (clean)
                    tokens_clean = self.model.embed(xb)
                    ctx_clean = self.model.encode_tokens(tokens_clean)
                    z_clean = self.model.project_view1(ctx_clean)

                    # CutMix on input
                    perm_a = torch.randperm(b, device=xb.device)
                    x_a = xb[perm_a]

                    m = (torch.rand(b, f, device=xb.device) < self.p_cutmix).to(
                        xb.dtype
                    )
                    x_cut = xb * m + x_a * (1.0 - m)

                    tokens_cut = self.model.embed(x_cut)

                    # MixUp on token embeddings
                    perm_b = torch.randperm(b, device=xb.device)
                    tokens_cut_b = tokens_cut[perm_b]

                    alpha = self.mixup_alpha
                    tokens_mix = alpha * tokens_cut + (1.0 - alpha) * tokens_cut_b

                    ctx_aug = self.model.encode_tokens(tokens_mix)
                    z_aug = self.model.project_view2(ctx_aug)

                    loss_info = info_nce_loss(z_clean, z_aug, self.temperature)

                    x_recon = self.model.reconstruct_from_ctx(ctx_aug)
                    loss_recon = F.mse_loss(x_recon, xb)

                    loss_ce = torch.zeros((), device=self.device)
                    loss_total = (
                        self.lambda_view * loss_info + self.lambda_recon * loss_recon
                    )

                    if is_train:
                        if optimizer is None:
                            raise ValueError("TRAIN split인데 optimizer가 None 입니다.")
                        optimizer.zero_grad()
                        loss_total.backward()
                        optimizer.step()

            elif stage == 2:
                # Stage2는 missing(x)로 CE finetune
                with torch.set_grad_enabled(is_train):
                    out = self.model(
                        x_cont=x_missing, need_logits=True, need_recon=False
                    )
                    logits = out["logits"]
                    if logits is None:
                        raise ValueError("logits is None")

                    loss_ce = F.cross_entropy(logits, y)
                    loss_info = torch.zeros((), device=self.device)
                    loss_recon = torch.zeros((), device=self.device)
                    loss_total = self.lambda_cls * loss_ce

                    if is_train:
                        if optimizer is None:
                            raise ValueError("TRAIN split인데 optimizer가 None 입니다.")
                        optimizer.zero_grad()
                        loss_total.backward()
                        optimizer.step()
            else:
                raise ValueError(f"unknown stage: {stage}")

            num_batches += 1
            total_sum += float(loss_total.item())
            ce_sum += float(loss_ce.item())
            info_sum += float(loss_info.item())
            recon_sum += float(loss_recon.item())

        denom = max(1, num_batches)
        return {
            "total": total_sum / denom,
            "ce": ce_sum / denom,
            "info": info_sum / denom,
            "recon": recon_sum / denom,
        }

    # -------------------------
    # Test / Predict (stage=2 사용 권장)
    # -------------------------
    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, split=Split.TEST, stage=2
        )

        metrics_overall = compute_classification_metrics(labels_all, preds_all)

        patterns = test_data.config.data.missing_patterns
        ratios = test_data.ratios

        metrics_by_ratio: dict[str, dict[float, dict]] = {}
        for p_i, pattern in enumerate(patterns):
            p_val = pattern.value
            metrics_by_ratio[p_val] = {}

            for r_i, ratio in enumerate(ratios):
                mask = (pattern_idx_all == p_i) & (ratio_idx_all == r_i)
                if np.any(mask):
                    y_sub = labels_all[mask]
                    y_hat_sub = preds_all[mask]
                    m = compute_classification_metrics(y_sub, y_hat_sub)
                    metrics_by_ratio[p_val][ratio] = m

        return {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

    @torch.no_grad()
    def predict(self, loader: DataLoader, split: Split = Split.TEST, stage: int = 2):
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        desc = self.get_desc("SAINT", split)

        for batch in tqdm(loader, desc=desc):
            x_missing, y, _, _, pattern_idx, ratio_idx = self._prepare_batch(batch)

            if stage == 1:
                # stage1은 분류 성능 평가 대상이 아니므로 logits 경로를 명시적으로 열어두되,
                # 실제로는 stage2 평가를 권장합니다.
                out = self.model(x_cont=x_missing, need_logits=True, need_recon=False)
                logits = out["logits"]
                if logits is None:
                    raise ValueError("logits is None")
            elif stage == 2:
                out = self.model(x_cont=x_missing, need_logits=True, need_recon=False)
                logits = out["logits"]
                if logits is None:
                    raise ValueError("logits is None")
            else:
                raise ValueError(f"unknown stage: {stage}")

            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            num_batches += 1

            all_preds.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_pattern_idx.append(pattern_idx.detach().cpu())
            all_ratio_idx.append(ratio_idx.detach().cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    # -------------------------
    # Utilities
    # -------------------------
    def _prepare_batch(self, batch: dict):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        x_ori = batch["x_originals"].to(self.device)
        bemv = batch["bemv"].to(self.device)
        pattern_idx = batch["pattern_idx"].to(self.device)
        ratio_idx = batch["ratio_idx"].to(self.device)
        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    def _get_model(self, input_dim: int, num_class: int | None) -> SAINT:
        if num_class is None:
            raise ValueError("num_class is None")

        model_cfg = self.config.model

        if hasattr(model_cfg, "saint_d_token"):
            d_token = int(model_cfg.saint_d_token)
        else:
            d_token = 192

        if hasattr(model_cfg, "saint_num_heads"):
            n_heads = int(model_cfg.saint_num_heads)
        else:
            n_heads = 8

        if hasattr(model_cfg, "saint_num_layers"):
            n_layers = int(model_cfg.saint_num_layers)
        else:
            n_layers = 3

        if hasattr(model_cfg, "saint_attn_dropout"):
            attn_dropout = float(model_cfg.saint_attn_dropout)
        else:
            attn_dropout = 0.1

        if hasattr(model_cfg, "saint_ff_dropout"):
            ff_dropout = float(model_cfg.saint_ff_dropout)
        else:
            ff_dropout = 0.1

        if hasattr(model_cfg, "saint_ff_mult"):
            ff_mult = int(model_cfg.saint_ff_mult)
        else:
            ff_mult = 4

        if hasattr(model_cfg, "saint_proj_dim"):
            proj_dim = int(model_cfg.saint_proj_dim)
        else:
            proj_dim = 128

        if hasattr(model_cfg, "saint_use_self_attention"):
            use_self = bool(model_cfg.saint_use_self_attention)
        else:
            use_self = True

        if hasattr(model_cfg, "saint_use_intersample_attention"):
            use_intersample = bool(model_cfg.saint_use_intersample_attention)
        else:
            use_intersample = True

        model = SAINT(
            n_cont_features=int(input_dim),
            d_token=d_token,
            num_heads=n_heads,
            num_layers=n_layers,
            d_out=int(num_class),
            use_self_attention=use_self,
            use_intersample_attention=use_intersample,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            ff_mult=ff_mult,
            proj_dim=proj_dim,
        ).to(self.device)

        return model

    def _make_optimizer_scheduler(self, stage: int, num_epochs: int):
        if self.model is None:
            raise ValueError("model is None")

        if stage == 1:
            lr = float(self.config.train.lr_stage_1)
            lr_min = float(self.config.train.lr_min_stage_1)
        elif stage == 2:
            lr = float(self.config.train.lr_stage_2)
            lr_min = float(self.config.train.lr_min_stage_2)
        else:
            raise ValueError(f"unknown stage: {stage}")

        # SAINT: AdamW + weight_decay (default 0.01)
        weight_decay = 0.01
        if hasattr(self.config.train, "weight_decay"):
            weight_decay = float(self.config.train.weight_decay)

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, int(num_epochs)),
            eta_min=lr_min,
        )
        return opt, sch

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: Path):
        if self.model is None:
            raise ValueError(
                "저장할 모델이 없습니다. fit() 이후에 save()를 호출하십시오."
            )

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "input_dim": int(self.input_dim) if self.input_dim is not None else None,
            "num_class": int(self.num_class) if self.num_class is not None else None,
            "model_path": str(model_path),
        }
        self.save_meta(save_dir, meta)
        return model_path

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        if meta["input_dim"] is None or meta["num_class"] is None:
            raise ValueError("meta에 input_dim/num_class가 없습니다.")

        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])

        model_path = Path(meta["model_path"])
        self.model = self._get_model(self.input_dim, self.num_class)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        return True
