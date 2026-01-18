"""Pareto front tracking and hypervolume utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import random


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if a dominates b (>= in all dims and > in at least one)."""
    ge_all = all(x >= y for x, y in zip(a, b))
    gt_any = any(x > y for x, y in zip(a, b))
    return ge_all and gt_any


def pareto_front(points: Iterable[Sequence[float]]) -> List[Tuple[float, ...]]:
    """Filter to non-dominated points."""
    front: List[Tuple[float, ...]] = []
    for p in points:
        p_t = tuple(float(x) for x in p)
        dominated = False
        to_remove: List[Tuple[float, ...]] = []
        for q in front:
            if dominates(q, p_t):
                dominated = True
                break
            if dominates(p_t, q):
                to_remove.append(q)
        if dominated:
            continue
        if to_remove:
            front = [q for q in front if q not in to_remove]
        front.append(p_t)
    return front


def hypervolume_2d(points: List[Tuple[float, float]]) -> float:
    """Compute 2D hypervolume for points in [0,1]^2 with reference at origin."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[0], reverse=True)
    hv = 0.0
    max_y = 0.0
    prev_x = 0.0
    for x, y in pts:
        hv += (prev_x - x) * max_y
        if y > max_y:
            max_y = y
        prev_x = x
    hv += prev_x * max_y
    return hv


def hypervolume_3d(points: List[Tuple[float, float, float]]) -> float:
    """Compute 3D hypervolume for points in [0,1]^3 with reference at origin."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[0], reverse=True)
    hv = 0.0
    current: List[Tuple[float, float, float]] = []
    for idx, point in enumerate(pts):
        current.append(point)
        next_x = pts[idx + 1][0] if idx + 1 < len(pts) else 0.0
        yz_points = [(p[1], p[2]) for p in current]
        hv += (point[0] - next_x) * hypervolume_2d(yz_points)
    return hv


def hypervolume_mc(
    points: List[Tuple[float, ...]], samples: int = 2048, seed: int = 0
) -> float:
    """Approximate hypervolume for >=4D by Monte Carlo sampling in [0,1]^d."""
    if not points:
        return 0.0
    dim = len(points[0])
    rng = random.Random(seed)
    dominated = 0
    for _ in range(samples):
        sample = tuple(rng.random() for _ in range(dim))
        for p in points:
            if all(s <= v for s, v in zip(sample, p)):
                dominated += 1
                break
    return dominated / float(samples)


def hypervolume(points: List[Tuple[float, ...]], samples: int = 2048) -> float:
    """Dispatch hypervolume computation by dimensionality."""
    if not points:
        return 0.0
    dim = len(points[0])
    if dim == 2:
        return hypervolume_2d(points)  # type: ignore[arg-type]
    if dim == 3:
        return hypervolume_3d(points)  # type: ignore[arg-type]
    return hypervolume_mc(points, samples=samples)


@dataclass
class ParetoBuffer:
    """Maintain a Pareto front with a bounded buffer size."""

    max_size: int = 128
    hv_samples: int = 2048

    def __post_init__(self):
        self._front: List[Tuple[float, ...]] = []
        self._dim: Optional[int] = None

    @property
    def front(self) -> List[Tuple[float, ...]]:
        return list(self._front)

    def add(self, point: Sequence[float]) -> bool:
        """Add a point to the Pareto front if non-dominated."""
        candidate = tuple(float(x) for x in point)
        if self._dim is None:
            self._dim = len(candidate)
        elif len(candidate) != self._dim:
            raise ValueError(f"Expected point dimension {self._dim}, got {len(candidate)}")
        updated = False
        new_front: List[Tuple[float, ...]] = []
        for existing in self._front:
            if dominates(existing, candidate):
                return False
            if not dominates(candidate, existing):
                new_front.append(existing)
            else:
                updated = True
        new_front.append(candidate)
        self._front = new_front
        if updated:
            self._prune_if_needed()
        elif len(self._front) > self.max_size:
            self._prune_if_needed()
        return True

    def update(self, points: Iterable[Sequence[float]]) -> None:
        """Add a batch of points to the Pareto front."""
        for p in points:
            self.add(p)

    def hypervolume(self) -> float:
        return hypervolume(self._front, samples=self.hv_samples)

    def hypervolume_with(self, point: Sequence[float]) -> float:
        front = pareto_front(self._front + [tuple(float(x) for x in point)])
        return hypervolume(front, samples=self.hv_samples)

    def _prune_if_needed(self) -> None:
        if len(self._front) <= self.max_size:
            return
        # Remove points with smallest hypervolume contribution
        while len(self._front) > self.max_size:
            base_hv = hypervolume(self._front, samples=self.hv_samples)
            contributions = []
            for idx, _ in enumerate(self._front):
                reduced = [p for i, p in enumerate(self._front) if i != idx]
                hv = hypervolume(reduced, samples=self.hv_samples)
                contributions.append((base_hv - hv, idx))
            contributions.sort(key=lambda x: x[0])
            _, drop_idx = contributions[0]
            self._front.pop(drop_idx)
