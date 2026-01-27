"""Reward aggregation interfaces for multi-objective RLHF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch


def normalize_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    total = weights.sum().clamp(min=eps)
    return weights / total


class RewardAggregator:
    """Base class for aggregating per-objective rewards into a scalar reward."""

    name: str = "base"

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
        training: bool = True,
    ) -> Dict[str, float]:
        return {}

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Mapping) -> None:
        _ = state


@dataclass
class WeightScheduler:
    """Optional stateful scheduler for reward weights."""

    def update(
        self,
        weights: torch.Tensor,
        rewards_per_func: torch.Tensor,
    ) -> torch.Tensor:
        return weights

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Mapping) -> None:
        _ = state
