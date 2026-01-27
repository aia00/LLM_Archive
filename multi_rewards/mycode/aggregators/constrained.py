"""Constrained RLHF aggregation via Lagrangian multipliers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import torch

from .base import RewardAggregator


def _direction_to_sign(direction: str) -> int:
    if direction.lower() in {"max", ">=", "ge"}:
        return -1  # reward should be >= tau
    if direction.lower() in {"min", "<=", "le"}:
        return 1  # reward should be <= tau
    raise ValueError(f"Unknown constraint direction: {direction}")


@dataclass
class ConstrainedRewardAggregator(RewardAggregator):
    """Lagrangian constrained aggregation: r_primary - sum(lambda_j * violation_j)."""

    primary_index: int
    constraint_indices: Sequence[int]
    thresholds: Sequence[float]
    directions: Sequence[str]
    lambda_lr: float = 0.05
    lambda_max: float = 10.0
    use_ema: bool = True
    ema_alpha: float = 0.05
    constraint_names: Optional[Sequence[str]] = None

    name: str = "constrained"

    def __post_init__(self) -> None:
        if len(self.constraint_indices) != len(self.thresholds):
            raise ValueError("constraint_indices and thresholds must be same length")
        if len(self.constraint_indices) != len(self.directions):
            raise ValueError("constraint_indices and directions must be same length")
        self._lambda = torch.zeros(len(self.constraint_indices), dtype=torch.float32)
        self._ema_violation = torch.zeros(len(self.constraint_indices), dtype=torch.float32)

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        device = rewards_per_func.device
        primary = rewards_per_func[:, self.primary_index]
        lambdas = self._lambda.to(device)
        penalties = []
        for idx, tau, direction in zip(
            self.constraint_indices, self.thresholds, self.directions
        ):
            reward = rewards_per_func[:, idx]
            sign = _direction_to_sign(direction)
            violation = sign * (reward - float(tau))
            penalties.append(torch.relu(violation))
        if not penalties:
            return primary
        penalty_stack = torch.stack(penalties, dim=1)
        scalar = primary - (penalty_stack * lambdas.unsqueeze(0)).sum(dim=1)
        return scalar

    def update(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
        training: bool = True,
    ) -> Dict[str, float]:
        if not training:
            return {}
        device = rewards_per_func.device
        violations = []
        for idx, tau, direction in zip(
            self.constraint_indices, self.thresholds, self.directions
        ):
            reward = rewards_per_func[:, idx]
            sign = _direction_to_sign(direction)
            violation = sign * (reward - float(tau))
            violations.append(torch.relu(violation).mean())
        if not violations:
            return {}
        violation_vec = torch.stack(violations).detach().cpu()
        if self.use_ema:
            self._ema_violation = (
                (1.0 - self.ema_alpha) * self._ema_violation
                + self.ema_alpha * violation_vec
            )
            update_signal = self._ema_violation
        else:
            update_signal = violation_vec
        self._lambda = torch.clamp(
            self._lambda + self.lambda_lr * update_signal,
            min=0.0,
            max=self.lambda_max,
        )
        metrics: Dict[str, float] = {}
        names = self.constraint_names or [f"c{i}" for i in range(len(violations))]
        for name, val, lam in zip(names, update_signal, self._lambda):
            metrics[f"constraint/{name}/violation"] = float(val.item())
            metrics[f"constraint/{name}/lambda"] = float(lam.item())
        return metrics

    def state_dict(self) -> Dict:
        return {
            "lambda": self._lambda.tolist(),
            "ema_violation": self._ema_violation.tolist(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if "lambda" in state:
            self._lambda = torch.tensor(state["lambda"], dtype=torch.float32)
        if "ema_violation" in state:
            self._ema_violation = torch.tensor(state["ema_violation"], dtype=torch.float32)
