# online_sft - Modular Online SFT Training Package
# Implements "Online SFT = RLHF" algorithm from the paper

from online_sft.trainer import OnlineSFTTrainer, OnlineSFTTrainerConfig
from online_sft.dataset_utils import (
    prepare_tldr_dataset,
    prepare_math_dataset,
    prepare_dataset,
    collate_fn,
    collate_fn_with_ground_truth,
)
from online_sft.reward_functions import (
    RewardFunction,
    NeuralRewardModel,
    MathVerificationReward,
    create_reward_function,
)
from online_sft.scheduler_utils import create_scheduler

# Distributed training (requires accelerate)
try:
    from online_sft.distributed_trainer import (
        DistributedOnlineSFTTrainer,
        DistributedOnlineSFTConfig,
        create_accelerator,
    )
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False
    DistributedOnlineSFTTrainer = None
    DistributedOnlineSFTConfig = None
    create_accelerator = None

__all__ = [
    # Trainer
    "OnlineSFTTrainer",
    "OnlineSFTTrainerConfig",
    # Distributed Trainer
    "DistributedOnlineSFTTrainer",
    "DistributedOnlineSFTConfig",
    "create_accelerator",
    "HAS_DISTRIBUTED",
    # Dataset utilities
    "prepare_tldr_dataset",
    "prepare_math_dataset",
    "prepare_dataset",
    "collate_fn",
    "collate_fn_with_ground_truth",
    # Reward functions
    "RewardFunction",
    "NeuralRewardModel",
    "MathVerificationReward",
    "create_reward_function",
    # Scheduler
    "create_scheduler",
]
