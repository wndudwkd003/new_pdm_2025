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
