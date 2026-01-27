"""RL-free multi-objective DPO loss utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PreferenceBatch:
    """Container for multi-objective preference log-probabilities."""

    logp_chosen: torch.Tensor
    logp_rejected: torch.Tensor


def compute_modpo_loss(
    batch: PreferenceBatch,
    *,
    weights: Optional[torch.Tensor] = None,
    beta: float = 0.1,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compute a multi-objective DPO loss.

    If logp tensors are 2D, the second dimension is interpreted as objectives.
    """
    logp_chosen = batch.logp_chosen
    logp_rejected = batch.logp_rejected
    logits = logp_chosen - logp_rejected

    if logits.dim() == 1:
        loss = -F.logsigmoid(beta * logits)
    else:
        if weights is None:
            weights = torch.full(
                (logits.size(1),),
                1.0 / logits.size(1),
                device=logits.device,
            )
        weights = weights.to(logits.device)
        weighted = logits * weights.unsqueeze(0)
        loss = -F.logsigmoid(beta * weighted.sum(dim=1))

    if reduce == "mean":
        return loss.mean()
    if reduce == "sum":
        return loss.sum()
    return loss
