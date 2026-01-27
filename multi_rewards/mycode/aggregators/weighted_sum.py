"""Weighted-sum reward aggregation with optional scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from .base import RewardAggregator, WeightScheduler, normalize_weights


@dataclass
class WeightedSumAggregator(RewardAggregator):
    """Aggregate rewards with a (possibly dynamic) weight vector."""

    weights: torch.Tensor
    scheduler: Optional[WeightScheduler] = None

    name: str = "weighted_sum"

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
        return (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)

    def update(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
        training: bool = True,
    ) -> dict:
        if self.scheduler is None or not training:
            return {}
        with torch.no_grad():
            updated = self.scheduler.update(
                self.weights.to(rewards_per_func.device), rewards_per_func
            )
        self.weights = normalize_weights(updated.detach().cpu())
        return {"weights/norm": float(self.weights.sum().item())}

    def state_dict(self) -> dict:
        state = {"weights": self.weights.tolist()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Mapping) -> None:
        weights = state.get("weights")
        if weights is not None:
            self.weights = normalize_weights(torch.tensor(weights, dtype=torch.float32))
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
