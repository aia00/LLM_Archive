"""Uncertainty-aware reward aggregation to downweight noisy objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch

from .base import RewardAggregator, normalize_weights


@dataclass
class UncertaintyWeightAggregator(RewardAggregator):
    """Weight rewards inversely to their running variance."""

    base_weights: torch.Tensor
    ema_alpha: float = 0.05
    eps: float = 1e-6

    name: str = "uncertainty"

    def __post_init__(self) -> None:
        if not isinstance(self.base_weights, torch.Tensor):
            self.base_weights = torch.tensor(self.base_weights, dtype=torch.float32)
        self.base_weights = normalize_weights(self.base_weights)
        self._ema_var = torch.ones_like(self.base_weights)

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        weights = self._current_weights().to(rewards_per_func.device)
        return (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)

    def update(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
        training: bool = True,
    ) -> Dict[str, float]:
        if not training or rewards_per_func.numel() == 0:
            return {}
        batch_var = rewards_per_func.var(dim=0, unbiased=False).detach().cpu()
        self._ema_var = (
            (1.0 - self.ema_alpha) * self._ema_var + self.ema_alpha * batch_var
        )
        metrics = {"uncertainty/mean_var": float(self._ema_var.mean().item())}
        return metrics

    def _current_weights(self) -> torch.Tensor:
        inv = 1.0 / (self._ema_var + self.eps)
        weights = self.base_weights * inv
        return normalize_weights(weights)

    def state_dict(self) -> Dict:
        return {"ema_var": self._ema_var.tolist()}

    def load_state_dict(self, state: Mapping) -> None:
        if "ema_var" in state:
            self._ema_var = torch.tensor(state["ema_var"], dtype=torch.float32)
