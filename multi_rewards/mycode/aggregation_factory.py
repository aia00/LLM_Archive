"""Factory for reward aggregators used by multi-objective trainers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .aggregators import (
    AdversarialWeightScheduler,
    CVaRScalarizer,
    ChebyshevScalarizer,
    ConstrainedRewardAggregator,
    ContextualWeightAggregator,
    NashScalarizer,
    UncertaintyWeightAggregator,
    WeightedSumAggregator,
)


def _parse_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_floats(value: Optional[str]) -> List[float]:
    items = _parse_csv(value)
    return [float(x) for x in items]


def _parse_weights(value: Optional[str]) -> Optional[List[float]]:
    vals = _parse_floats(value)
    return vals if vals else None


def _resolve_index(name_or_idx: str, reward_names: Sequence[str]) -> int:
    if name_or_idx.isdigit():
        idx = int(name_or_idx)
        if idx < 0 or idx >= len(reward_names):
            raise ValueError(f"Reward index out of range: {name_or_idx}")
        return idx
    if name_or_idx not in reward_names:
        raise ValueError(f"Unknown reward name: {name_or_idx}")
    return reward_names.index(name_or_idx)


def build_reward_aggregator(method: str, reward_names: Sequence[str], base_weights: Sequence[float], args):
    method = method.lower()
    base_weights_tensor = torch.tensor(base_weights, dtype=torch.float32)

    if method == "minimax":
        scheduler = AdversarialWeightScheduler(
            eta=getattr(args, "minimax_eta", 0.5),
            min_weight=getattr(args, "minimax_min_weight", 0.0),
        )
        return WeightedSumAggregator(base_weights_tensor, scheduler=scheduler)

    if method == "chebyshev":
        reference = _parse_floats(getattr(args, "chebyshev_reference", None))
        reference = reference if reference else None
        return ChebyshevScalarizer(base_weights_tensor, reference=reference)

    if method == "nash":
        baseline = _parse_floats(getattr(args, "nash_baseline", None))
        baseline = baseline if baseline else None
        return NashScalarizer(
            epsilon=getattr(args, "nash_epsilon", 1e-3),
            baseline=baseline,
        )

    if method == "cvar":
        return CVaRScalarizer(
            base_weights_tensor,
            alpha=getattr(args, "cvar_alpha", 0.2),
        )

    if method == "contextual":
        return ContextualWeightAggregator(
            base_weights_tensor,
            short_length=getattr(args, "context_short_length", 120),
            long_length=getattr(args, "context_long_length", 300),
            short_weights=_parse_weights(getattr(args, "context_short_weights", None)),
            long_weights=_parse_weights(getattr(args, "context_long_weights", None)),
            clarity_weights=_parse_weights(getattr(args, "context_clarity_weights", None)),
            conciseness_weights=_parse_weights(getattr(args, "context_conciseness_weights", None)),
        )

    if method == "uncertainty":
        return UncertaintyWeightAggregator(
            base_weights_tensor,
            ema_alpha=getattr(args, "uncertainty_ema_alpha", 0.05),
            eps=getattr(args, "uncertainty_eps", 1e-6),
        )

    if method == "constrained":
        primary = getattr(args, "primary_reward", "accuracy")
        primary_idx = _resolve_index(primary, reward_names)

        constraint_names = _parse_csv(getattr(args, "constraint_rewards", None))
        if not constraint_names:
            constraint_indices = [i for i in range(len(reward_names)) if i != primary_idx]
        else:
            constraint_indices = [_resolve_index(name, reward_names) for name in constraint_names]

        thresholds = _parse_floats(getattr(args, "constraint_thresholds", None))
        if thresholds and len(thresholds) != len(constraint_indices):
            raise ValueError("constraint_thresholds must match constraint_rewards length")
        if not thresholds:
            thresholds = [0.0 for _ in constraint_indices]

        directions = _parse_csv(getattr(args, "constraint_directions", None))
        if directions and len(directions) != len(constraint_indices):
            raise ValueError("constraint_directions must match constraint_rewards length")
        if not directions:
            directions = ["max" for _ in constraint_indices]

        constraint_names_final = [reward_names[i] for i in constraint_indices]

        return ConstrainedRewardAggregator(
            primary_index=primary_idx,
            constraint_indices=constraint_indices,
            thresholds=thresholds,
            directions=directions,
            lambda_lr=getattr(args, "constraint_lambda_lr", 0.05),
            lambda_max=getattr(args, "constraint_lambda_max", 10.0),
            use_ema=not getattr(args, "constraint_disable_ema", False),
            ema_alpha=getattr(args, "constraint_ema_alpha", 0.05),
            constraint_names=constraint_names_final,
        )

    # Default: weighted sum
    return WeightedSumAggregator(base_weights_tensor)
