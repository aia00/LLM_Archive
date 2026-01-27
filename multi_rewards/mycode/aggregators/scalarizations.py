"""Nonlinear scalarizations for multi-objective rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from .base import RewardAggregator, normalize_weights


@dataclass
class ChebyshevScalarizer(RewardAggregator):
    """Chebyshev scalarization: max min_i alpha_i (r_i - z_i)."""

    weights: torch.Tensor
    reference: Optional[Sequence[float]] = None

    name: str = "chebyshev"

    def __post_init__(self) -> None:
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.tensor(self.weights, dtype=torch.float32)
        self.weights = normalize_weights(self.weights)
        if self.reference is not None:
            self.reference = [float(x) for x in self.reference]
            if len(self.reference) != self.weights.numel():
                raise ValueError("reference length must match number of rewards")

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        weights = self.weights.to(rewards_per_func.device)
        if self.reference is None:
            ref = torch.zeros_like(rewards_per_func)
        else:
            ref = torch.tensor(self.reference, device=rewards_per_func.device)
            ref = ref.unsqueeze(0).expand_as(rewards_per_func)
        scaled = weights.unsqueeze(0) * (rewards_per_func - ref)
        return torch.min(scaled, dim=1).values


@dataclass
class NashScalarizer(RewardAggregator):
    """Nash social welfare: sum log(eps + r_i - b_i)."""

    epsilon: float = 1e-3
    baseline: Optional[Sequence[float]] = None

    name: str = "nash"

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        if self.baseline is None:
            adjusted = rewards_per_func
        else:
            base = torch.tensor(self.baseline, device=rewards_per_func.device)
            adjusted = rewards_per_func - base.unsqueeze(0)
        adjusted = torch.clamp(adjusted, min=self.epsilon)
        return torch.log(adjusted).sum(dim=1)


@dataclass
class CVaRScalarizer(RewardAggregator):
    """Risk-sensitive scalarization focusing on the lower tail."""

    weights: torch.Tensor
    alpha: float = 0.2

    name: str = "cvar"

    def __post_init__(self) -> None:
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.tensor(self.weights, dtype=torch.float32)
        self.weights = normalize_weights(self.weights)

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        weights = self.weights.to(rewards_per_func.device)
        base_reward = (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)
        if base_reward.numel() == 0:
            return base_reward
        alpha = float(max(1e-4, min(1.0, self.alpha)))
        # Focus on the lower tail: keep rewards at or below the alpha-quantile.
        q = torch.quantile(base_reward.detach(), alpha)
        mask = base_reward <= q
        scale = 1.0 / max(alpha, 1e-4)
        return base_reward * mask.to(base_reward.dtype) * scale
