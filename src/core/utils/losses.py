from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Standard (symmetric) InfoNCE for paired views.

    Args:
        z1: [B, D]
        z2: [B, D]
        temperature: temperature (tau)
        normalize: if True, L2-normalize embeddings before similarity

    Returns:
        scalar loss
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must be 2D tensors [B, D]")
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("z1 and z2 must have the same batch size")
    if z1.shape[1] != z2.shape[1]:
        raise ValueError("z1 and z2 must have the same embedding dim")

    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    B = z1.shape[0]
    logits = (z1 @ z2.t()) / float(temperature)  # [B, B]
    labels = torch.arange(B, device=logits.device)

    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, exp(logvar)) || N(0, I) )
    mu/logvar: [B, D] (또는 [*, D]) 를 가정하고 마지막 차원을 latent dim으로 처리합니다.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu/logvar shape mismatch: {tuple(mu.shape)} vs {tuple(logvar.shape)}"
        )
    if mu.ndim < 2:
        raise ValueError("mu/logvar must be at least 2D like [B, D]")

    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    kl = kl.sum(dim=-1).mean()  # scalar
    return kl


class HierarchicalMissingContrastiveLoss(nn.Module):
    """
    최종 로스(요구사항 반영):

      L_view = L_inst + lambda_cls * L_cls

    L_inst (강함):
      - positive: 같은 base의 모든 결측 버전들 + (원본 1개)
      - negative:
          * classification: 라벨이 다른 샘플들만 (diff-label only)
          * regression: 다른 base는 모두 negative

    L_cls (약함):
      - classification only
      - base(원본) 임베딩들끼리만 사용
      - positive: 같은 라벨(다른 base)
      - negative: 다른 라벨
    """

    def __init__(self, *, tau: float = 0.07, lambda_cls: float = 0.2) -> None:
        super().__init__()
        self.tau = float(tau)
        self.lambda_cls = float(lambda_cls)

    def _mpnce(
        self, logits: torch.Tensor, pos_mask: torch.Tensor, den_mask: torch.Tensor
    ) -> torch.Tensor:
        neg_inf = torch.finfo(logits.dtype).min

        den = torch.logsumexp(logits.masked_fill(~den_mask, neg_inf), dim=1)
        num = torch.logsumexp(logits.masked_fill(~pos_mask, neg_inf), dim=1)

        loss = -(num - den)
        valid = pos_mask.any(dim=1)

        if not bool(valid.any().item()):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        return loss[valid].mean()

    def forward(
        self,
        *,
        z_missing: torch.Tensor,  # [N, D]
        z_base: torch.Tensor,  # [B, D]
        y_base: torch.Tensor | None,  # [B] (classification) or None (regression)
        views_per_base: int,
    ) -> torch.Tensor:
        if z_missing.ndim != 2 or z_base.ndim != 2:
            raise ValueError("z_missing, z_base must be 2D tensors")

        N, D = z_missing.shape
        B = z_base.shape[0]

        if z_base.shape[1] != D:
            raise ValueError("z_missing and z_base embedding dim mismatch")
        if N != B * views_per_base:
            raise ValueError(f"N must be B*V, but got N={N}, B={B}, V={views_per_base}")

        if y_base is not None:
            if y_base.ndim != 1:
                raise ValueError("y_base must be 1D tensor")
            if y_base.shape[0] != B:
                raise ValueError("y_base length must equal B")

        z_m = F.normalize(z_missing, dim=1)
        z_b = F.normalize(z_base, dim=1)

        # -------------------------
        # (1) instance alignment: [N + B, D]
        # -------------------------
        z_all = torch.cat([z_m, z_b], dim=0)  # [M, D]
        M = z_all.shape[0]

        base_ids_m = torch.arange(B, device=z_all.device).repeat_interleave(
            views_per_base
        )  # [N]
        base_ids_b = torch.arange(B, device=z_all.device)  # [B]
        base_all = torch.cat([base_ids_m, base_ids_b], dim=0)  # [M]

        logits_all = (z_all @ z_all.t()) / self.tau  # [M, M]
        diag = torch.eye(M, dtype=torch.bool, device=z_all.device)

        same_base = base_all[:, None] == base_all[None, :]
        pos_inst = same_base & (~diag)

        if y_base is None:
            # regression: negatives = different base
            neg_inst = (~same_base) & (~diag)
        else:
            y_m = y_base.repeat_interleave(views_per_base)  # [N]
            y_all = torch.cat([y_m, y_base], dim=0)  # [M]
            same_cls = y_all[:, None] == y_all[None, :]
            neg_inst = (~same_cls) & (~diag)

        den_inst = pos_inst | neg_inst
        loss_inst = self._mpnce(logits_all, pos_inst, den_inst)

        # -------------------------
        # (2) class alignment: base only (classification only)
        # -------------------------
        if y_base is None or self.lambda_cls == 0.0:
            loss_cls = torch.zeros((), device=z_all.device, dtype=z_all.dtype)
        else:
            logits_b = (z_b @ z_b.t()) / self.tau  # [B, B]
            diag_b = torch.eye(B, dtype=torch.bool, device=z_b.device)

            same_cls_b = y_base[:, None] == y_base[None, :]
            pos_cls = same_cls_b & (~diag_b)
            neg_diff_b = (~same_cls_b) & (~diag_b)

            den_cls = pos_cls | neg_diff_b
            loss_cls = self._mpnce(logits_b, pos_cls, den_cls)

        return loss_inst + self.lambda_cls * loss_cls


class ReGVAEFinalStage1Loss(nn.Module):
    def __init__(
        self,
        *,
        tau: float = 0.07,
        lambda_cls: float = 0.2,
        w_contrast: float = 1.0,
        w_mu: float = 1.0,
        w_prior: float = 0.1,
        w_recon: float = 1.0,
        mu_align: str = "mse",
    ) -> None:
        super().__init__()
        self.contrast = HierarchicalMissingContrastiveLoss(
            tau=tau, lambda_cls=lambda_cls
        )

        self.w_contrast = float(w_contrast)
        self.w_mu = float(w_mu)
        self.w_prior = float(w_prior)
        self.w_recon = float(w_recon)

        if mu_align not in ("mse", "cosine"):
            raise ValueError(f"mu_align must be 'mse' or 'cosine', got: {mu_align}")
        self.mu_align = mu_align

    def _mu_align_loss(
        self, mu_missing: torch.Tensor, mu_clean_rep: torch.Tensor
    ) -> torch.Tensor:
        if self.mu_align == "mse":
            return F.mse_loss(mu_missing, mu_clean_rep)
        a = F.normalize(mu_missing, dim=-1)
        b = F.normalize(mu_clean_rep, dim=-1)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    def forward(
        self,
        *,
        mu_clean: torch.Tensor,  # [B, D]
        logvar_clean: torch.Tensor,  # [B, D]
        mu_missing: torch.Tensor,  # [N, D]
        logvar_missing: torch.Tensor,  # [N, D]
        x_clean: torch.Tensor,  # [B, F]
        recon_clean: torch.Tensor,  # [B, F]
        recon_missing: torch.Tensor,  # [N, F]
        y_base: torch.Tensor | None,  # [B] or None(regression)
        views_per_base: int,
    ) -> dict[str, torch.Tensor]:
        if mu_missing.ndim != 2 or mu_clean.ndim != 2:
            raise ValueError("mu_missing/mu_clean must be [*, D]")

        if y_base is not None:
            if mu_clean.shape[0] != y_base.shape[0]:
                raise ValueError(
                    "B mismatch: mu_clean.shape[0] must equal y_base.shape[0]"
                )

        if mu_missing.shape[0] != mu_clean.shape[0] * int(views_per_base):
            raise ValueError("N must be B*views_per_base")

        B = mu_clean.shape[0]
        V = int(views_per_base)

        loss_contrast = self.contrast(
            z_missing=mu_missing,
            z_base=mu_clean,
            y_base=y_base,  # regression이면 None
            views_per_base=V,
        )

        mu_clean_rep = mu_clean.repeat_interleave(V, dim=0)
        loss_mu = self._mu_align_loss(mu_missing, mu_clean_rep)

        loss_prior_clean = kl_standard_normal(mu_clean, logvar_clean)
        loss_prior_missing = kl_standard_normal(mu_missing, logvar_missing)
        loss_prior = 0.5 * (loss_prior_clean + loss_prior_missing)

        x_clean_rep = x_clean.repeat_interleave(V, dim=0)
        loss_recon_clean = F.mse_loss(recon_clean, x_clean)
        loss_recon_missing = F.mse_loss(recon_missing, x_clean_rep)
        loss_recon = 0.5 * (loss_recon_clean + loss_recon_missing)

        total = (
            self.w_contrast * loss_contrast
            + self.w_mu * loss_mu
            + self.w_prior * loss_prior
            + self.w_recon * loss_recon
        )

        return {
            "total": total,
            "contrast": loss_contrast,
            "mu_align": loss_mu,
            "prior": loss_prior,
            "prior_clean": loss_prior_clean,
            "prior_missing": loss_prior_missing,
            "recon": loss_recon,
            "recon_clean": loss_recon_clean,
            "recon_missing": loss_recon_missing,
        }
