"""Distributionally robust (minimax) weight updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .base import WeightScheduler, normalize_weights


@dataclass
class AdversarialWeightScheduler(WeightScheduler):
    """Adversarial weights via exponentiated-gradient on mean rewards.

    Lower-performing objectives get higher weights.
    """

    eta: float = 0.5
    min_weight: float = 0.0

    def update(
        self,
        weights: torch.Tensor,
        rewards_per_func: torch.Tensor,
    ) -> torch.Tensor:
        if rewards_per_func.numel() == 0:
            return weights
        mean_rewards = rewards_per_func.mean(dim=0)
        scaled = weights * torch.exp(-self.eta * mean_rewards)
        if self.min_weight > 0:
            scaled = torch.clamp(scaled, min=self.min_weight)
        return normalize_weights(scaled)

    def state_dict(self) -> dict:
        return {"eta": float(self.eta), "min_weight": float(self.min_weight)}

    def load_state_dict(self, state: Mapping) -> None:
        if "eta" in state:
            self.eta = float(state["eta"])
        if "min_weight" in state:
            self.min_weight = float(state["min_weight"])
