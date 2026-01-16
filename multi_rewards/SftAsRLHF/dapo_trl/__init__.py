"""
DAPO Training with TRL

This package implements DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
using Hugging Face's TRL library.

Key Features:
- Token-level policy gradient normalization
- Clip-higher for improved diversity
- Overlong reward shaping
- Rule-based math answer verification

Usage:
    from dapo_trl import prepare_dapo_dataset, create_dapo_reward_fn
    from dapo_trl.train_dapo import main

    # Or run the training script directly:
    # python -m dapo_trl.train_dapo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct
"""

__version__ = "0.1.0"

from .dataset_utils import (
    load_dapo_math_dataset,
    prepare_dapo_dataset,
    preprocess_dataset_for_grpo,
    create_reward_lookup,
)

from .reward_function import (
    compute_score,
    compute_dapo_reward,
    create_dapo_reward_fn,
)
from .dynamic_grpo_trainer import DynamicGRPOTrainer

__all__ = [
    # Dataset utilities
    "load_dapo_math_dataset",
    "prepare_dapo_dataset",
    "preprocess_dataset_for_grpo",
    "create_reward_lookup",
    # Reward functions
    "compute_score",
    "compute_dapo_reward",
    "create_dapo_reward_fn",
    # Trainer
    "DynamicGRPOTrainer",
]
