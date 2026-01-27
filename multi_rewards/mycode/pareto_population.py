"""Utilities for Pareto-set learning and checkpoint selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import random

from .pareto import pareto_front


@dataclass
class ParetoCheckpoint:
    """Metadata for a checkpoint evaluated on multiple objectives."""

    path: str
    metrics: Sequence[float]
    weights: Sequence[float] | None = None


@dataclass
class ParetoArchive:
    """Track and filter Pareto-optimal checkpoints."""

    entries: List[ParetoCheckpoint]

    def __init__(self) -> None:
        self.entries = []

    def add(self, entry: ParetoCheckpoint) -> None:
        self.entries.append(entry)

    def pareto_optimal(self) -> List[ParetoCheckpoint]:
        points = [tuple(float(x) for x in e.metrics) for e in self.entries]
        front = set(pareto_front(points))
        return [e for e in self.entries if tuple(float(x) for x in e.metrics) in front]


def sample_weight_vectors(
    num_vectors: int,
    dim: int,
    alpha: float = 1.0,
    seed: int = 0,
) -> List[List[float]]:
    """Sample Dirichlet weight vectors for Pareto population training."""
    rng = random.Random(seed)
    weights = []
    for _ in range(num_vectors):
        gamma = [rng.gammavariate(alpha, 1.0) for _ in range(dim)]
        total = sum(gamma) or 1.0
        weights.append([g / total for g in gamma])
    return weights
