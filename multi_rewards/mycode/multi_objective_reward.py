"""Reward weighting helpers and hypervolume-based meta-reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, List
import math

from .pareto import ParetoBuffer


DEFAULT_REWARD_NAMES: Tuple[str, ...] = (
    "reward_acc",
    "reward_conc",
    "reward_clar",
    "reward_answer_found",
    "reward_boxed",
    "reward_answer_len",
)

WEIGHT_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "accuracy": (0.5, 0.25, 0.25),
    "balanced": (0.334, 0.333, 0.333),
    "efficiency": (0.25, 0.375, 0.375),
}


def normalize_weights(weights: Sequence[float]) -> Tuple[float, ...]:
    total = float(sum(weights))
    if total <= 0:
        n = len(weights)
        if n == 0:
            return ()
        return tuple(1.0 / n for _ in weights)
    return tuple(float(w) / total for w in weights)


def _parse_reward_names(reward_names: Optional[Sequence[str]]) -> List[str]:
    if reward_names is None:
        return list(DEFAULT_REWARD_NAMES)
    if isinstance(reward_names, str):
        items = [item.strip() for item in reward_names.split(",") if item.strip()]
        return items or list(DEFAULT_REWARD_NAMES)
    return [str(item).strip() for item in reward_names if str(item).strip()]


def resolve_weights(
    preset: Optional[str],
    override: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    return resolve_weights_with_names(preset, override, DEFAULT_REWARD_NAMES)


def resolve_weights_with_names(
    preset: Optional[str],
    override: Optional[Sequence[float]] = None,
    reward_names: Optional[Sequence[str]] = None,
) -> Tuple[float, ...]:
    names = _parse_reward_names(reward_names)
    n = len(names)
    if override is not None:
        if len(override) != n:
            raise ValueError(f"Expected {n} weights, got {len(override)}")
        return normalize_weights(override)
    if n == 3 and preset in WEIGHT_PRESETS:
        return normalize_weights(WEIGHT_PRESETS[preset])

    if preset is None or preset == "balanced":
        return normalize_weights([1.0] * n)

    if preset == "accuracy":
        weights = [1.0] * n
        if "reward_acc" in names and n > 1:
            idx = names.index("reward_acc")
            weights = [0.5 / (n - 1)] * n
            weights[idx] = 0.5
        return normalize_weights(weights)

    if preset == "efficiency":
        weights = [1.0] * n
        if "reward_conc" in names:
            weights[names.index("reward_conc")] = 1.5
        if "reward_clar" in names:
            weights[names.index("reward_clar")] = 1.5
        if "reward_acc" in names:
            weights[names.index("reward_acc")] = 1.0
        return normalize_weights(weights)

    raise ValueError(f"Unknown weight preset: {preset}")


@dataclass
class ParetoMetaReward:
    """Compute hypervolume-based meta scale for Method B."""

    buffer: ParetoBuffer
    base: float = 0.5
    scale: float = 1.5

    def update(self, point: Iterable[float]) -> float:
        """Update the Pareto buffer and return the meta scale r_pareto."""
        delta_hv = self.buffer.hypervolume_with(point) - self.buffer.hypervolume()
        self.buffer.add(point)
        return self.base + self.scale * math.tanh(delta_hv)

    def update_with_delta(self, point: Iterable[float]) -> tuple[float, float]:
        """Update the buffer and return (meta_scale, delta_hv)."""
        delta_hv = self.buffer.hypervolume_with(point) - self.buffer.hypervolume()
        self.buffer.add(point)
        meta_scale = self.base + self.scale * math.tanh(delta_hv)
        return meta_scale, delta_hv
