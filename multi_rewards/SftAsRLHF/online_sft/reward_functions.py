# online_sft/reward_functions.py
# Reward function interfaces and implementations for Online SFT training
# Supports both neural reward models and rule-based verification

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# =============================================================================
# Abstract Reward Interface
# =============================================================================


class RewardFunction(ABC):
    """Abstract base class for reward functions."""

    @abstractmethod
    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of generated completion strings
            prompts: Optional list of prompts
            **kwargs: Additional arguments (e.g., ground_truths for math)

        Returns:
            List of scalar rewards
        """
        pass


# =============================================================================
# Neural Reward Model
# =============================================================================


class NeuralRewardModel(RewardFunction):
    """
    Reward function using a neural reward model (AutoModelForSequenceClassification).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        use_sigmoid: Optional[bool] = None,
        scale: float = 1.0,
        shift: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Reward model (AutoModelForSequenceClassification)
            tokenizer: Tokenizer for the reward model
            use_sigmoid: Whether to apply sigmoid. If None, auto-detect.
            scale: Scale factor for rewards
            shift: Shift for rewards
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_sigmoid = use_sigmoid
        self.scale = scale
        self.shift = shift
        self.device = device or next(model.parameters()).device

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set model to eval mode
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """Compute rewards using the neural reward model."""
        toks = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Handle OOV padding tokens
        try:
            attn = toks.get("attention_mask")
            if attn is not None and "input_ids" in toks:
                input_ids = toks["input_ids"]
                pad_positions = attn == 0
                if pad_positions.any():
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None:
                        input_ids[pad_positions] = eos_id
                        toks["input_ids"] = input_ids
        except Exception:
            pass

        out = self.model(**toks)
        logits = out.logits if hasattr(out, "logits") else out[0]

        # Determine sigmoid usage
        use_sigmoid = self.use_sigmoid
        num_labels = getattr(getattr(self.model, "config", None), "num_labels", None)

        if use_sigmoid is None:
            if (num_labels == 2) or (logits.dim() > 1 and logits.size(-1) == 2):
                use_sigmoid = True
            else:
                use_sigmoid = False

        if use_sigmoid:
            if logits.dim() > 1 and logits.size(-1) == 2:
                r = torch.softmax(logits, dim=-1)[..., 1]
            else:
                r = torch.sigmoid(logits)
        else:
            r = logits

        # Reduce to per-example scalar
        while r.dim() > 1:
            if r.size(-1) == 1:
                r = r.squeeze(-1)
            else:
                r = r.mean(dim=-1)

        r = r * self.scale + self.shift
        return r.cpu().tolist()


# =============================================================================
# Math Verification Reward (adapted from DAPO)
# =============================================================================


# Normalization constants from DAPO
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
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
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
) -> tuple:
    """
    Minerva-style answer check (exact match to verl implementation).

    Only uses the answer_pattern regex to extract the answer.
    If no match is found, returns "[INVALID]".
    """
    # Extract answer from solution using ONLY the answer pattern
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


def compute_math_score(
    solution_str: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Compute reward score for math verification.

    Only uses string comparison after normalization - NO numeric fallback.
    """
    if not ground_truth or ground_truth == "[INVALID]":
        return {"score": -1.0, "acc": False, "pred": ""}

    solution_str = solution_str[-300:]  # Match official truncation
    correct, pred = is_correct_minerva(solution_str, ground_truth)

    reward = 1.0 if correct else -1.0
    return {"score": reward, "acc": correct, "pred": pred}


class MathVerificationReward(RewardFunction):
    """
    Rule-based reward function for math verification.
    Uses answer extraction and string comparison (matching DAPO/verl implementation).
    """

    def __init__(
        self,
        reward_lookup: Optional[Dict[str, str]] = None,
        tokenizer: Optional[Any] = None,
        max_response_length: int = 20480,
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
        apply_overlong_penalty: bool = True,
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
    ):
        """
        Args:
            reward_lookup: Optional dict mapping prompts to ground truths
            tokenizer: Optional tokenizer for computing token lengths (for overlong penalty)
            max_response_length: Maximum response length for overlong penalty
            overlong_buffer_len: Buffer length for overlong penalty calculation
            overlong_penalty_factor: Factor for overlong penalty
            apply_overlong_penalty: Whether to apply overlong penalty
            correct_reward: Reward for correct answer
            incorrect_reward: Reward for incorrect answer
        """
        self.reward_lookup = reward_lookup or {}
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.overlong_buffer_len = overlong_buffer_len
        self.overlong_penalty_factor = overlong_penalty_factor
        self.apply_overlong_penalty = apply_overlong_penalty
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward

    def _token_length(self, text: str) -> int:
        """Compute token length of text."""
        if self.tokenizer is None:
            return len(text)
        try:
            return len(self.tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:
            return len(text)

    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute math verification rewards.

        Args:
            completions: List of generated completions
            prompts: Optional list of prompts (used for reward_lookup)
            ground_truths: Optional list of ground truth answers (takes precedence)
            **kwargs: Additional arguments

        Returns:
            List of rewards
        """
        rewards = []
        prompts = prompts or [""] * len(completions)

        for i, completion in enumerate(completions):
            # Get ground truth
            gt = None
            if ground_truths is not None and i < len(ground_truths):
                gt = ground_truths[i]
            elif prompts[i] in self.reward_lookup:
                gt = self.reward_lookup[prompts[i]]

            if gt is None or gt == "":
                # No ground truth available
                rewards.append(0.0)
                continue

            # Compute base reward
            score_dict = compute_math_score(completion, gt)
            reward = float(score_dict["score"])

            # Apply correct/incorrect reward mapping
            if reward > 0:
                reward = self.correct_reward
            else:
                reward = self.incorrect_reward

            # Apply overlong penalty if enabled
            if self.apply_overlong_penalty:
                resp_len = self._token_length(completion)
                expected_len = self.max_response_length - self.overlong_buffer_len
                exceed_len = resp_len - expected_len
                if exceed_len > 0:
                    overlong_penalty = min(
                        -exceed_len / self.overlong_buffer_len * self.overlong_penalty_factor,
                        0.0,
                    )
                    reward += overlong_penalty

            rewards.append(reward)

        return rewards


# =============================================================================
# Factory Function
# =============================================================================


def create_reward_function(
    reward_type: str,
    **kwargs,
) -> RewardFunction:
    """
    Factory function to create reward functions.

    Args:
        reward_type: Type of reward function ("neural", "math")
        **kwargs: Arguments for the specific reward function

    Returns:
        RewardFunction instance
    """
    if reward_type == "neural":
        required = ["model", "tokenizer"]
        for r in required:
            if r not in kwargs:
                raise ValueError(f"Neural reward requires '{r}' argument")
        return NeuralRewardModel(**kwargs)
    elif reward_type == "math":
        return MathVerificationReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# =============================================================================
# Test
# =============================================================================


if __name__ == "__main__":
    # Test math verification
    math_reward = MathVerificationReward()

    test_cases = [
        ("The solution is straightforward. Answer: 42", "42", True),
        ("Working through the problem... Answer: 100", "100", True),
        ("The answer is 50", "50", False),  # Wrong format
        ("Answer: 3.14159", "3.14159", True),
        ("Answer: \\frac{1}{2}", "1/2", False),  # Normalization test
    ]

    for completion, gt, expected in test_cases:
        rewards = math_reward(
            completions=[completion],
            ground_truths=[gt],
        )
        score_dict = compute_math_score(completion, gt)
        print(f"Completion: {completion[:50]}...")
        print(f"Ground truth: {gt}")
        print(f"Reward: {rewards[0]}, Correct: {score_dict['acc']}, Pred: {score_dict['pred']}")
        print()
