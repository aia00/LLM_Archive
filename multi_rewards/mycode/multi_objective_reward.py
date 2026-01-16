"""Reward weighting helpers and hypervolume-based meta-reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import math

from .pareto import ParetoBuffer


WEIGHT_PRESETS: Dict[str, Tuple[float, float, float]] = {
    "accuracy": (0.5, 0.25, 0.25),
    "balanced": (0.334, 0.333, 0.333),
    "efficiency": (0.25, 0.375, 0.375),
}


def normalize_weights(weights: Sequence[float]) -> Tuple[float, float, float]:
    total = float(sum(weights))
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(float(w) / total for w in weights)


def resolve_weights(
    preset: Optional[str],
    override: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    if override is not None:
        return normalize_weights(override)
    if preset is None:
        return normalize_weights(WEIGHT_PRESETS["balanced"])
    if preset not in WEIGHT_PRESETS:
        raise ValueError(f"Unknown weight preset: {preset}")
    return normalize_weights(WEIGHT_PRESETS[preset])


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
