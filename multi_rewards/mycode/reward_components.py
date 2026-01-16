"""Reward components for multi-objective math reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re


# -----------------------------------------------------------------------------
# Accuracy reward (DAPO-style string match)
# -----------------------------------------------------------------------------

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\\ ", ""),
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
    "\\text{\n}s",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\\mathrm{th}",
    r"^\\circ",
    r"^{\\circ}",
    r"\\;",
    r",\\!",
    "{,}",
    '"',
    "\\dots",
]


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
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


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
    """Minerva-style answer check using a fixed answer regex."""
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"

    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        boxed = last_boxed_only_string(gt)
        if boxed:
            gt = normalize_final_answer(remove_boxed(boxed))
        else:
            gt = normalize_final_answer(gt)
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def compute_math_score(solution_str: str, ground_truth: str) -> dict:
    """Compute reward score for math verification using string match."""
    if not ground_truth or ground_truth == "[INVALID]":
        return {"score": 0.0, "acc": False, "pred": ""}

    solution_str = solution_str[-300:]
    correct, pred = is_correct_minerva(solution_str, ground_truth)

    reward = 1.0 if correct else 0.0
    return {"score": reward, "acc": correct, "pred": pred}


@dataclass
class AccuracyReward:
    """Binary accuracy reward using string match."""

    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"

    def __call__(self, completions: List[str], ground_truths: List[str]) -> List[float]:
        rewards: List[float] = []
        for completion, gt in zip(completions, ground_truths):
            score = compute_math_score(completion, gt)
            rewards.append(float(score["score"]))
        return rewards


# -----------------------------------------------------------------------------
# Conciseness reward
# -----------------------------------------------------------------------------


@dataclass
class ConcisenessReward:
    """Reward for shorter-than-average completions."""

    tokenizer: Optional[object] = None
    use_ema: bool = True
    ema_alpha: float = 0.05

    def __post_init__(self):
        self._avg_len: Optional[float] = None
        self._count: int = 0

    def _token_length(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        try:
            return len(self.tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            return len(text)

    def __call__(self, completions: List[str], update: bool = True) -> List[float]:
        lengths = [self._token_length(text) for text in completions]
        mean_len = sum(lengths) / max(1, len(lengths))

        if self._avg_len is None:
            self._avg_len = mean_len

        rewards = [1.0 if length < self._avg_len else 0.0 for length in lengths]

        if update:
            if self.use_ema:
                self._avg_len = (
                    (1.0 - self.ema_alpha) * self._avg_len
                    + self.ema_alpha * mean_len
                )
            else:
                self._count += len(lengths)
                self._avg_len = (
                    (self._avg_len * (self._count - len(lengths)) + sum(lengths))
                    / self._count
                )

        return rewards


# -----------------------------------------------------------------------------
# Clarity reward
# -----------------------------------------------------------------------------


@dataclass
class ClarityReward:
    """Reward for explicit step-by-step markers."""

    patterns: Optional[List[str]] = None

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = [
                r"(?i)step\s*by\s*step",
                r"(?i)step\s*\d+",
                r"(?i)first\b",
                r"(?i)second\b",
                r"(?i)third\b",
                r"(?m)^\s*\d+\.",
                r"(?m)^\s*\(\d+\)",
            ]
        self._compiled = [re.compile(p) for p in self.patterns]

    def __call__(self, completions: List[str]) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            matched = any(regex.search(completion) for regex in self._compiled)
            rewards.append(1.0 if matched else 0.0)
        return rewards


def build_reward_functions(tokenizer: Optional[object] = None):
    """Create reward functions for accuracy, conciseness, and clarity."""
    accuracy = AccuracyReward()
    conciseness = ConcisenessReward(tokenizer=tokenizer)
    clarity = ClarityReward()

    def accuracy_fn(completions, prompts=None, **kwargs):
        ground_truths = kwargs.get("ground_truth") or kwargs.get("ground_truths")
        if ground_truths is None:
            ground_truths = [""] * len(completions)
        return accuracy(completions, ground_truths)

    def conciseness_fn(completions, prompts=None, **kwargs):
        update = kwargs.get("update", True)
        return conciseness(completions, update=update)

    def clarity_fn(completions, prompts=None, **kwargs):
        return clarity(completions)

    accuracy_fn.__name__ = "accuracy"
    conciseness_fn.__name__ = "conciseness"
    clarity_fn.__name__ = "clarity"

    return [accuracy_fn, conciseness_fn, clarity_fn], {
        "accuracy": accuracy,
        "conciseness": conciseness,
        "clarity": clarity,
    }
