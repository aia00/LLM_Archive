"""
Reward logic for DAPO math verification (exact match to verl's implementation).

Matches verl/verl/utils/reward_score/math_dapo.py exactly by:
- ONLY using "Answer:" pattern for extraction (no boxed fallbacks)
- NO numeric fallback comparison (only string matching after normalization)
- Truncating long responses to last 300 characters before scoring
- Returning binary +1/-1 rewards
- Applying overlong reward shaping based on token lengths
"""

from typing import Dict, List, Optional
import re


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Strip \\boxed{...} wrapper."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalization used by the official math_dapo scorer."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> tuple[bool, str]:
    """Minerva-style answer check (exact match to verl implementation).

    ONLY uses the answer_pattern regex to extract the answer.
    If no match is found, returns "[INVALID]".
    This matches verl/verl/utils/reward_score/math_dapo.py exactly.
    """
    # Extract answer from solution using ONLY the answer pattern
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"

    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Strict boxed-answer verification."""
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, str]:
    """Wrapper matching verl.utils.reward_score.math_dapo.verify."""
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> Dict[str, object]:
    """Compute reward score exactly like the verl math_dapo scorer.

    Only uses string comparison after normalization - NO numeric fallback.
    This matches verl/verl/utils/reward_score/math_dapo.py exactly.
    """
    # Short-circuit clearly invalid ground truths (used by verify_setup)
    if not ground_truth or ground_truth == "[INVALID]":
        return {"score": -1.0, "acc": False, "pred": ""}

    solution_str = solution_str[-300:]  # match official truncation
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    # NO numeric fallback - only string comparison (matching verl)
    reward = 1.0 if correct else -1.0
    return {"score": reward, "acc": correct, "pred": pred}


def _extract_completion_text(completion) -> str:
    """Handle TRL completion formats (chat list or plain string)."""
    if isinstance(completion, list) and completion:
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _token_length(text: str, tokenizer=None) -> int:
    """Token length helper to mirror verl's attention-mask based length."""
    if tokenizer is None:
        return len(text)
    try:
        # Avoid adding extra special tokens; length is what matters
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return len(text)


def compute_dapo_reward(
    completions: List,
    prompts: List[str],
    ground_truths: List[str],
    *,
    tokenizer=None,
    max_response_length: int = 20480,
    overlong_buffer_len: int = 4096,
    overlong_penalty_factor: float = 1.0,
    apply_overlong_penalty: bool = True,
    strict_box_verify: bool = False,
    **kwargs,
) -> List[float]:
    """
    Compute DAPO rewards using the same semantics as the original Verl implementation.
    """
    rewards = []

    for completion, ground_truth in zip(completions, ground_truths):
        completion_text = _extract_completion_text(completion)

        score_dict = compute_score(
            solution_str=completion_text,
            ground_truth=ground_truth,
            strict_box_verify=strict_box_verify,
        )
        reward = float(score_dict["score"])

        if apply_overlong_penalty:
            resp_len = _token_length(completion_text, tokenizer)
            expected_len = max_response_length - overlong_buffer_len
            exceed_len = resp_len - expected_len
            if exceed_len > 0:
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0.0)
                reward += overlong_reward

        rewards.append(reward)

    return rewards


def create_dapo_reward_fn(
    ground_truths: List[str],
    *,
    tokenizer=None,
    max_response_length: int = 20480,
    overlong_buffer_len: int = 4096,
    overlong_penalty_factor: float = 1.0,
    apply_overlong_penalty: bool = True,
    strict_box_verify: bool = False,
):
    """
    Factory returning a TRL-compatible reward function mirroring verl's logic.
    """

    def reward_fn(completions, prompts=None, **kwargs):
        return compute_dapo_reward(
            completions=completions,
            prompts=prompts or [""] * len(completions),
            ground_truths=ground_truths,
            tokenizer=tokenizer,
            max_response_length=max_response_length,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
            apply_overlong_penalty=apply_overlong_penalty,
            strict_box_verify=strict_box_verify,
        )

    return reward_fn
