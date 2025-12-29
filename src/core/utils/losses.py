from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    z_clean: torch.Tensor, z_noisy: torch.Tensor, temperature: float = 0.1
):
    z_clean = F.normalize(z_clean, dim=-1)
    z_noisy = F.normalize(z_noisy, dim=-1)

    logits = z_clean @ z_noisy.T / temperature  # (B, B)
    B = z_clean.size(0)
    labels = torch.arange(B, device=z_clean.device)

    loss_clean_to_noisy = F.cross_entropy(logits, labels)
    loss_noisy_to_clean = F.cross_entropy(logits.T, labels)

    loss = 0.5 * (loss_clean_to_noisy + loss_noisy_to_clean)
    return loss


class HierarchicalMissingContrastiveLoss(nn.Module):
    """
    최종 로스(요구사항 반영):

      L_view = L_inst + lambda_cls * L_cls

    L_inst (강함):
      - positive: 같은 base의 모든 결측 버전들 + (원본 1개)
      - negative: 라벨이 다른 샘플들만 (diff-label only)

    L_cls (약함):
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
        z_missing: torch.Tensor,  # [N, D]  (결측 시나리오 임베딩들)
        z_base: torch.Tensor,  # [B, D]  (원본 임베딩, base당 1개)
        y_base: torch.Tensor,  # [B]
        views_per_base: int,  # V = P * R
    ) -> torch.Tensor:
        if z_missing.ndim != 2 or z_base.ndim != 2:
            raise ValueError("z_missing, z_base must be 2D tensors")
        if y_base.ndim != 1:
            raise ValueError("y_base must be 1D tensor")

        N, D = z_missing.shape
        B = z_base.shape[0]

        if z_base.shape[1] != D:
            raise ValueError("z_missing and z_base embedding dim mismatch")
        if N != B * views_per_base:
            raise ValueError(f"N must be B*V, but got N={N}, B={B}, V={views_per_base}")

        z_m = F.normalize(z_missing, dim=1)
        z_b = F.normalize(z_base, dim=1)

        # -------------------------
        # (1) 인스턴스 정렬: [N + B, D]
        # missing끼리도, missing-원본도 같은 base면 positive
        # -------------------------
        z_all = torch.cat([z_m, z_b], dim=0)  # [M, D]
        M = z_all.shape[0]

        base_ids_m = torch.arange(B, device=z_all.device).repeat_interleave(
            views_per_base
        )  # [N]
        base_ids_b = torch.arange(B, device=z_all.device)  # [B]
        base_all = torch.cat([base_ids_m, base_ids_b], dim=0)  # [M]

        y_m = y_base.repeat_interleave(views_per_base)  # [N]
        y_all = torch.cat([y_m, y_base], dim=0)  # [M]

        logits_all = (z_all @ z_all.t()) / self.tau  # [M, M]

        diag = torch.eye(M, dtype=torch.bool, device=z_all.device)
        same_base = base_all[:, None] == base_all[None, :]
        same_cls = y_all[:, None] == y_all[None, :]

        pos_inst = same_base & (~diag)
        neg_diff = (~same_cls) & (~diag)

        den_inst = pos_inst | neg_diff
        loss_inst = self._mpnce(logits_all, pos_inst, den_inst)

        # -------------------------
        # (2) 클래스 정렬: base 임베딩(B개)만 사용
        # -------------------------
        logits_b = (z_b @ z_b.t()) / self.tau  # [B, B]
        diag_b = torch.eye(B, dtype=torch.bool, device=z_b.device)

        same_cls_b = y_base[:, None] == y_base[None, :]
        pos_cls = same_cls_b & (~diag_b)
        neg_diff_b = (~same_cls_b) & (~diag_b)

        den_cls = pos_cls | neg_diff_b
        loss_cls = self._mpnce(logits_b, pos_cls, den_cls)

        return loss_inst + self.lambda_cls * loss_cls


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu, exp(logvar)) || N(0, I) )
    return: scalar (batch mean)
    """
    if mu.ndim != 2 or logvar.ndim != 2:
        raise ValueError("mu and logvar must be 2D tensors [B, D]")
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar shape mismatch")

    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    kl = 0.5 * (torch.exp(logvar) + mu * mu - 1.0 - logvar).sum(dim=1)
    return kl.mean()


def kl_gaussian(
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(mu_p, exp(logvar_p)) || N(mu_q, exp(logvar_q)) )
    return: scalar (batch mean)
    """
    if mu_p.ndim != 2 or logvar_p.ndim != 2:
        raise ValueError("mu_p and logvar_p must be 2D tensors [B, D]")
    if mu_q.ndim != 2 or logvar_q.ndim != 2:
        raise ValueError("mu_q and logvar_q must be 2D tensors [B, D]")
    if mu_p.shape != logvar_p.shape or mu_q.shape != logvar_q.shape:
        raise ValueError("mu/logvar shape mismatch")
    if mu_p.shape != mu_q.shape:
        raise ValueError("p and q shape mismatch")

    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)

    # 0.5 * sum( log(var_q/var_p) + (var_p + (mu_p-mu_q)^2)/var_q - 1 )
    kl = 0.5 * ((logvar_q - logvar_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0).sum(
        dim=1
    )
    return kl.mean()


def symmetric_kl_gaussian(
    mu_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mu_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    """
    0.5 * ( KL(a||b) + KL(b||a) )
    return: scalar
    """
    kl_ab = kl_gaussian(mu_a, logvar_a, mu_b, logvar_b)
    kl_ba = kl_gaussian(mu_b, logvar_b, mu_a, logvar_a)
    return 0.5 * (kl_ab + kl_ba)


class ReGVAEViewKLLoss(nn.Module):
    """
    ReGVAE용 view loss:
      - view 정렬: symmetric KL(q_missing || q_clean)
      - prior 정규화: KL(q(z|x)||N(0,I)) (missing/clean 둘 다 평균)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        *,
        mu_clean: torch.Tensor,
        logvar_clean: torch.Tensor,
        mu_missing: torch.Tensor,
        logvar_missing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_view = symmetric_kl_gaussian(
            mu_missing, logvar_missing, mu_clean, logvar_clean
        )
        loss_kl_clean = kl_standard_normal(mu_clean, logvar_clean)
        loss_kl_missing = kl_standard_normal(mu_missing, logvar_missing)
        loss_kl = 0.5 * (loss_kl_clean + loss_kl_missing)

        return {
            "view": loss_view,
            "kl": loss_kl,
            "kl_clean": loss_kl_clean,
            "kl_missing": loss_kl_missing,
        }


class ReGVAEFinalStage1Loss(nn.Module):
    """
    추천 1 구성(contrastive + mu-only align + prior KL + recon)을 그대로 묶은 최종 loss.

    total =
        w_contrast * L_contrast
      + w_mu       * L_mu_align
      + w_prior    * L_prior_kl
      + w_recon    * L_recon

    - L_contrast:
        HierarchicalMissingContrastiveLoss (multi-positive; same base all views + clean)
    - L_mu_align:
        mu_missing(view별) <-> mu_clean(base) 정렬 (MSE 또는 cosine)
    - L_prior_kl:
        0.5*( KL(q(z|clean)||N(0,I)) + KL(q(z|missing)||N(0,I)) )
    - L_recon:
        0.5*( MSE(recon_clean, x_clean) + MSE(recon_missing, x_clean_rep) )
        (missing view가 여러 개면 x_clean을 repeat_interleave 해서 맞춤)
    """

    def __init__(
        self,
        *,
        tau: float = 0.07,
        lambda_cls: float = 0.2,
        w_contrast: float = 1.0,
        w_mu: float = 1.0,
        w_prior: float = 0.1,
        w_recon: float = 1.0,
        mu_align: str = "mse",  # "mse" | "cosine"
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
        self,
        mu_missing: torch.Tensor,  # [N, D]
        mu_clean_rep: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        if self.mu_align == "mse":
            return F.mse_loss(mu_missing, mu_clean_rep)
        # cosine distance
        a = F.normalize(mu_missing, dim=-1)
        b = F.normalize(mu_clean_rep, dim=-1)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    def forward(
        self,
        *,
        # posterior params
        mu_clean: torch.Tensor,  # [B, D]
        logvar_clean: torch.Tensor,  # [B, D]
        mu_missing: torch.Tensor,  # [N, D] where N = B*V
        logvar_missing: torch.Tensor,  # [N, D]
        # recon targets/preds
        x_clean: torch.Tensor,  # [B, F]
        recon_clean: torch.Tensor,  # [B, F]
        recon_missing: torch.Tensor,  # [N, F]
        # contrastive meta
        y_base: torch.Tensor,  # [B]
        views_per_base: int,  # V
    ) -> dict[str, torch.Tensor]:
        if mu_missing.ndim != 2 or mu_clean.ndim != 2:
            raise ValueError("mu_missing/mu_clean must be [*, D]")
        if mu_clean.shape[0] != y_base.shape[0]:
            raise ValueError("B mismatch: mu_clean.shape[0] must equal y_base.shape[0]")
        if mu_missing.shape[0] != mu_clean.shape[0] * int(views_per_base):
            raise ValueError("N must be B*views_per_base")

        B = mu_clean.shape[0]
        V = int(views_per_base)
        N = mu_missing.shape[0]

        # ----- (1) contrastive (multi-positive; same base all views + clean)
        loss_contrast = self.contrast(
            z_missing=mu_missing,
            z_base=mu_clean,
            y_base=y_base,
            views_per_base=V,
        )

        # ----- (2) mu-only alignment (each missing view -> its base clean)
        mu_clean_rep = mu_clean.repeat_interleave(V, dim=0)  # [N, D]
        loss_mu = self._mu_align_loss(mu_missing, mu_clean_rep)

        # ----- (3) prior KL (clean + missing)
        loss_prior_clean = kl_standard_normal(mu_clean, logvar_clean)
        loss_prior_missing = kl_standard_normal(mu_missing, logvar_missing)
        loss_prior = 0.5 * (loss_prior_clean + loss_prior_missing)

        # ----- (4) recon (clean + missing->x_clean)
        x_clean_rep = x_clean.repeat_interleave(V, dim=0)  # [N, F]
        loss_recon_clean = F.mse_loss(recon_clean, x_clean)
        loss_recon_missing = F.mse_loss(recon_missing, x_clean_rep)
        loss_recon = 0.5 * (loss_recon_clean + loss_recon_missing)

        # ----- total
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
