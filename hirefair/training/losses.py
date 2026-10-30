"""Task and adversary losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def task_loss(
    logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | float | None = None
) -> torch.Tensor:
    """Binary cross-entropy with logits, optionally reweighted for imbalance."""
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)


def adversary_loss(adv_logits: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the gender labels the adversary receives."""
    if adv_logits.shape[0] == 0:
        return torch.zeros((), device=adv_logits.device)
    return F.cross_entropy(adv_logits, gender.long())
