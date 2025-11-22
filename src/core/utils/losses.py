import torch
import torch.nn.functional as F


def info_nce_loss(z_clean: torch.Tensor, z_noisy: torch.Tensor, temperature: float = 0.1):
    z_clean = F.normalize(z_clean, dim=-1)
    z_noisy = F.normalize(z_noisy, dim=-1)

    logits = z_clean @ z_noisy.T / temperature  # (B, B)
    B = z_clean.size(0)
    labels = torch.arange(B, device=z_clean.device)

    loss_clean_to_noisy = F.cross_entropy(logits, labels)
    loss_noisy_to_clean = F.cross_entropy(logits.T, labels)

    loss = 0.5 * (loss_clean_to_noisy + loss_noisy_to_clean)
    return loss
