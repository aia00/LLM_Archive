"""Multi-objective reward function for VERL."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from mycode.reward_components import AccuracyReward, ConcisenessReward, ClarityReward
from mycode.multi_objective_reward import resolve_weights


_META_SCALE = 1.0

_accuracy = AccuracyReward()
_conciseness = ConcisenessReward(tokenizer=None)
_clarity = ClarityReward()


def set_meta_scale(value: float) -> None:
    global _META_SCALE
    _META_SCALE = float(value)


def get_meta_scale() -> float:
    return float(_META_SCALE)


def _compute_components(
    solution_strs: List[str],
    ground_truths: List[str],
    update_running_avg: bool = True,
) -> tuple[List[float], List[float], List[float]]:
    acc = _accuracy(solution_strs, ground_truths)
    conc = _conciseness(solution_strs, update=update_running_avg)
    clar = _clarity(solution_strs)
    return acc, conc, clar


def _compute_scalar(
    acc: List[float],
    conc: List[float],
    clar: List[float],
    weights: Sequence[float],
    meta_scale: Optional[float] = None,
) -> List[float]:
    w_acc, w_conc, w_clar = weights
    scale = _META_SCALE if meta_scale is None else float(meta_scale)
    return [
        scale * (w_acc * a + w_conc * c + w_clar * l)
        for a, c, l in zip(acc, conc, clar)
    ]


def compute_score(
    data_sources: Optional[List[str]] = None,
    solution_strs: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    extra_infos: Optional[List[dict]] = None,
    *,
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
    multi_objective_mode: str = "static",
    weight_preset: str = "balanced",
    weights: Optional[Iterable[float]] = None,
    meta_scale: Optional[float] = None,
    update_running_avg: bool = True,
    **_kwargs,
) -> Union[List[dict], dict]:
    """Compute scalar reward and component rewards for VERL's reward managers."""
    single_mode = solution_strs is None and ground_truths is None and data_sources is None

    if solution_strs is None:
        solution_strs = [solution_str or ""]
    if ground_truths is None:
        ground_truths = [ground_truth or ""]
    if data_sources is None:
        data_sources = [data_source or "default"] * len(solution_strs)
    if extra_infos is None:
        extra_infos = [extra_info or {}] * len(solution_strs)

    if weights is not None:
        weights = resolve_weights(None, weights)
    else:
        weights = resolve_weights(weight_preset, None)

    acc, conc, clar = _compute_components(
        solution_strs, ground_truths, update_running_avg=update_running_avg
    )

    mode = multi_objective_mode.lower()
    if mode == "hypervolume":
        scalar = _compute_scalar(acc, conc, clar, weights, meta_scale=meta_scale)
    elif mode == "static" or mode == "dynamic":
        scalar = _compute_scalar(acc, conc, clar, weights, meta_scale=1.0)
    else:
        scalar = _compute_scalar(acc, conc, clar, weights, meta_scale=1.0)

    results = []
    for a, c, l, s in zip(acc, conc, clar, scalar):
        results.append(
            {
                "score": float(s),
                "reward_acc": float(a),
                "reward_conc": float(c),
                "reward_clar": float(l),
            }
        )
    if single_mode:
        return results[0]
    return results
