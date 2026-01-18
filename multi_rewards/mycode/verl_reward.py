"""Multi-objective reward function for VERL."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union
import re

from mycode.reward_components import (
    AccuracyReward,
    ConcisenessReward,
    ClarityReward,
    AnswerLengthReward,
    extract_final_answer,
    normalize_final_answer,
    last_boxed_only_string,
)
from mycode.multi_objective_reward import (
    DEFAULT_REWARD_NAMES,
    resolve_weights_with_names,
)


_META_SCALE = 1.0

_accuracy = AccuracyReward()
_conciseness = ConcisenessReward(tokenizer=None)
_clarity = ClarityReward()
_answer_len = AnswerLengthReward(tokenizer=None, answer_pattern=_accuracy.answer_pattern)


def set_meta_scale(value: float) -> None:
    global _META_SCALE
    _META_SCALE = float(value)


def get_meta_scale() -> float:
    return float(_META_SCALE)


def _parse_reward_names(reward_names: Optional[Sequence[str]]) -> List[str]:
    if reward_names is None:
        return list(DEFAULT_REWARD_NAMES)
    if isinstance(reward_names, str):
        items = [item.strip() for item in reward_names.split(",") if item.strip()]
        return items or list(DEFAULT_REWARD_NAMES)
    return [str(item).strip() for item in reward_names if str(item).strip()]


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
    reward_names: Optional[Sequence[str]] = None,
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

    reward_names = _parse_reward_names(reward_names or _kwargs.get("reward_names"))

    acc = _accuracy(solution_strs, ground_truths)
    conc = _conciseness(solution_strs, update=update_running_avg)
    clar = _clarity(solution_strs)
    answer_len = _answer_len(solution_strs, update=update_running_avg)
    answer_found = []
    boxed = []
    for completion in solution_strs:
        has_boxed = last_boxed_only_string(completion) is not None
        has_answer = bool(re.findall(_accuracy.answer_pattern, completion))
        has_final = bool(re.findall(r"(?i)final\s*answer\s*[:=]", completion))
        has_answer_is = bool(re.findall(r"(?i)answer\s*is\s*\S", completion))
        answer_found.append(1.0 if (has_boxed or has_answer or has_final or has_answer_is) else 0.0)
        boxed.append(1.0 if has_boxed else 0.0)

    components = {
        "reward_acc": acc,
        "reward_conc": conc,
        "reward_clar": clar,
        "reward_answer_len": answer_len,
        "reward_answer_found": answer_found,
        "reward_boxed": boxed,
    }

    if weights is not None:
        weights = resolve_weights_with_names(None, weights, reward_names)
    else:
        weights = resolve_weights_with_names(weight_preset, None, reward_names)

    mode = multi_objective_mode.lower()
    scale = _META_SCALE if meta_scale is None else float(meta_scale)
    if mode == "hypervolume":
        scalar_scale = scale
    else:
        scalar_scale = 1.0

    scalar = []
    for idx in range(len(solution_strs)):
        total = 0.0
        for name, weight in zip(reward_names, weights):
            total += float(weight) * float(components.get(name, [0.0])[idx])
        scalar.append(scalar_scale * total)

    results = []
    for idx, (completion, gt, s) in enumerate(zip(solution_strs, ground_truths, scalar)):
        extracted = extract_final_answer(completion, _accuracy.answer_pattern)
        pred_norm = normalize_final_answer(extracted)
        gt_norm = normalize_final_answer(gt)
        clarity_markers = _clarity.count_markers(completion)
        completion_len = len(completion)
        completion_tokens = len(completion.split())
        pred_len = len(pred_norm)
        gt_len = len(gt_norm)
        token_len = _conciseness.token_length(completion)
        results.append(
            {
                "score": float(s),
                "reward_acc": float(components["reward_acc"][idx]),
                "reward_conc": float(components["reward_conc"][idx]),
                "reward_clar": float(components["reward_clar"][idx]),
                "reward_answer_len": float(components["reward_answer_len"][idx]),
                "reward_answer_found": float(components["reward_answer_found"][idx]),
                "reward_boxed": float(components["reward_boxed"][idx]),
                "answer_found": float(components["reward_answer_found"][idx]),
                "clarity_markers": float(clarity_markers),
                "completion_len": float(completion_len),
                "completion_tokens": float(completion_tokens),
                "pred_len": float(pred_len),
                "gt_len": float(gt_len),
                "token_len": float(token_len),
                "pred": pred_norm,
            }
        )
    if single_mode:
        return results[0]
    return results
