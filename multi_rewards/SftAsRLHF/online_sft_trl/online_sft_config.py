# online_sft_trl/online_sft_config.py
"""Configuration class for Online SFT Trainer extending GRPOConfig."""

from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig


@dataclass
class OnlineSFTConfig(GRPOConfig):
    """
    Configuration class for the OnlineSFTTrainer.

    Extends GRPOConfig with Online SFT-specific parameters for the novel
    "Online SFT as RLHF" algorithm.

    The Online SFT loss is:
        L = - [ w * s + 0.5 * stop_grad(w) * s^2 ].mean()

    where:
        w = (r - beta * (log_pi - log_ref)) / (sign(log_pi) * clamp(|log_pi|, min=eps))
        s = log_pi (optionally normalized by length and clamped)

    Parameters:
        denom_eps (`float`, *optional*, defaults to `1e-8`):
            Epsilon for weight denominator to avoid division by zero.
        clip_weight_value (`float` or `None`, *optional*, defaults to `1.0`):
            Clip weight w to [-clip_weight_value, +clip_weight_value]. Set to None to disable.
        clip_logprob_min (`float` or `None`, *optional*, defaults to `-10.0`):
            Clamp log probabilities from below. Helps stabilize training when log probs
            become very negative. Set to None to disable.
        normalize_logprob_by_length (`bool`, *optional*, defaults to `True`):
            Whether to normalize sequence log probability by response length before
            applying the loss. Helps with length bias.
        correct_reward (`float`, *optional*, defaults to `1.0`):
            Reward value for correct answers (used in math verification reward).
        incorrect_reward (`float`, *optional*, defaults to `-1.0`):
            Reward value for incorrect answers (used in math verification reward).
        filter_groups_enable (`bool`, *optional*, defaults to `True`):
            Enable DAPO-style group filtering: keep only groups with reward variance > 0.
            This ensures training signal has both positive and negative examples.
        filter_groups_min_std (`float`, *optional*, defaults to `0.0`):
            Minimum reward std to keep a group. Groups with std <= this value are filtered.
        max_filter_retries (`int`, *optional*, defaults to `10`):
            Maximum number of retries when all groups are filtered out.
    """

    # Online SFT specific parameters
    denom_eps: float = field(
        default=1e-8,
        metadata={
            "help": "Epsilon for weight denominator to avoid division by zero."
        },
    )
    clip_weight_value: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Clip weight w to [-clip, +clip]. Set to None to disable clipping."
        },
    )
    clip_logprob_min: Optional[float] = field(
        default=-10.0,
        metadata={
            "help": "Clamp log probabilities from below. Set to None to disable."
        },
    )
    normalize_logprob_by_length: bool = field(
        default=False,
        metadata={
            "help": "Normalize sequence log prob by response length before loss computation."
        },
    )

    # Reward settings (for math verification)
    correct_reward: float = field(
        default=1.0,
        metadata={"help": "Reward for correct answers."},
    )
    incorrect_reward: float = field(
        default=-1.0,
        metadata={"help": "Reward for incorrect answers."},
    )

    # Group filtering (DAPO-style dynamic sampling)
    filter_groups_enable: bool = field(
        default=True,
        metadata={
            "help": "Enable group filtering: keep only groups with reward variance > 0."
        },
    )
    filter_groups_metric: str = field(
        default="acc",
        metadata={
            "help": "Metric to use for group filtering: 'acc' (accuracy) or 'reward'."
        },
    )
    filter_groups_min_std: float = field(
        default=0.0,
        metadata={
            "help": "Minimum reward std to keep a group (> 0 means must have variance)."
        },
    )
    max_filter_retries: int = field(
        default=10,
        metadata={"help": "Maximum retries when all groups are filtered out."},
    )

    def __post_init__(self):
        # Call parent's __post_init__
        super().__post_init__()

        # Validate Online SFT specific parameters
        if self.denom_eps <= 0:
            raise ValueError(
                f"denom_eps must be positive, got {self.denom_eps}"
            )

        if self.clip_weight_value is not None and self.clip_weight_value <= 0:
            raise ValueError(
                f"clip_weight_value must be positive or None, got {self.clip_weight_value}"
            )

        if self.max_filter_retries < 0:
            raise ValueError(
                f"max_filter_retries must be non-negative, got {self.max_filter_retries}"
            )
