# Online SFT Package

A modular implementation of "Online SFT = RLHF" algorithm supporting multiple tasks.

## Overview

This package implements the Online SFT training algorithm from the paper, which shows that RLHF equals weighted online SFT when using the appropriate weight function and gradient estimator.

## Package Structure

```
online_sft/
├── __init__.py               # Package initialization and exports
├── trainer.py                # OnlineSFTTrainer class (single GPU)
├── distributed_trainer.py    # DistributedOnlineSFTTrainer (FSDP + vLLM)
├── dataset_utils.py          # Dataset loading utilities (TLDR, Math, etc.)
├── reward_functions.py       # Reward function interfaces and implementations
├── scheduler_utils.py        # Learning rate scheduler utilities
├── train.py                  # Single-GPU training script
├── train_distributed.py      # Distributed training script (FSDP)
├── run_tldr.sh               # Run script for TLDR task
├── run_math.sh               # Run script for Math task (single GPU)
├── run_math_distributed.sh   # Run script for Math task (multi-GPU FSDP)
└── README.md                 # This file
```

## Supported Tasks

### 1. TLDR Summarization (Original)

Uses a neural reward model to score summaries.

```bash
# Run with default settings
./online_sft/run_tldr.sh

# Or with custom settings
python -m online_sft.train \
    --task_type tldr \
    --dataset_name "trl-internal-testing/tldr-preference-sft-trl-style" \
    --sft_model_path "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr" \
    --reward_model_path "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr" \
    --total_episodes 30000 \
    --per_device_train_batch_size 16
```

### 2. Math Reasoning (Adapted from DAPO)

Uses rule-based math verification reward (exact match after normalization).

```bash
# Run with default settings
./online_sft/run_math.sh

# Or with custom settings
python -m online_sft.train \
    --task_type math \
    --math_dataset_name "BytedTsinghua-SIA/DAPO-Math-17k" \
    --sft_model_path "Qwen/Qwen2.5-Math-7B" \
    --total_episodes 50000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

## Key Components

### OnlineSFTTrainer

The core trainer implementing the weighted SFT loss:

```python
from online_sft import OnlineSFTTrainer, OnlineSFTTrainerConfig

cfg = OnlineSFTTrainerConfig(
    beta=0.1,                    # KL penalty coefficient
    max_new_tokens=256,          # Max response length
    temperature=1.0,
    normalize_logprob_by_length=True,
)

trainer = OnlineSFTTrainer(
    policy=policy_model,
    ref_policy=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,   # Or use reward_fn for rule-based
    cfg=cfg,
)
```

### Reward Functions

Two types of reward functions are supported:

1. **Neural Reward Model**: For tasks like TLDR
   ```python
   from online_sft import NeuralRewardModel

   reward = NeuralRewardModel(model, tokenizer)
   ```

2. **Math Verification**: For math reasoning tasks
   ```python
   from online_sft import MathVerificationReward

   reward = MathVerificationReward(
       reward_lookup={"prompt": "ground_truth"},
       correct_reward=1.0,
       incorrect_reward=-1.0,
   )
   ```

### Dataset Utilities

```python
from online_sft import prepare_dataset

# For TLDR
train_ds, val_ds, _ = prepare_dataset("tldr", tokenizer)

# For Math
train_ds, val_ds, reward_lookup = prepare_dataset("math", tokenizer)
```

## Configuration

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `task_type` | `tldr` | Task type: tldr, math |
| `total_episodes` | 10000 | Total training episodes |
| `per_device_train_batch_size` | 4 | Batch size per device |
| `gradient_accumulation_steps` | 8 | Gradient accumulation |
| `learning_rate` | 3e-6 | Learning rate |
| `beta` | 0.1 | KL penalty coefficient |
| `response_length` | 256 | Max response tokens |

### Math-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `correct_reward` | 1.0 | Reward for correct answer |
| `incorrect_reward` | -1.0 | Reward for incorrect answer |
| `apply_overlong_penalty` | True | Penalize long responses |
| `overlong_penalty_factor` | 1.0 | Penalty factor |

## Algorithm

The loss function implements the unbiased gradient estimator from Proposition 3:

```
L_hat = - [ w * s + 0.5 * stop_grad(w) * s^2 ]

where:
  s = log pi_theta(y|x)  (optionally normalized by length)
  w = [r(x,y) - beta * KL] / log pi_theta(y|x)
```

This provides the same gradient as the RLHF objective without PPO.

## Migration from Original Code

If you were using the original `train_online_sft.py`, migrate as follows:

```bash
# Old:
python train_online_sft.py --dataset_name "..." --total_episodes 30000

# New:
python -m online_sft.train --task_type tldr --dataset_name "..." --total_episodes 30000
```

The original files (`train_online_sft.py`, `online_sft_trainer.py`) are kept for backward compatibility.

## Distributed Training (FSDP + vLLM)

For efficient multi-GPU training, use the distributed trainer with Accelerate FSDP:

### Quick Start

```bash
# Run with FSDP across 4 GPUs
./online_sft/run_math_distributed.sh

# Enable vLLM for faster generation (optional)
./online_sft/run_math_distributed.sh --use_vllm
```

### Manual Launch

```bash
accelerate launch \
    --num_processes 4 \
    --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_sharding_strategy FULL_SHARD \
    --mixed_precision bf16 \
    -m online_sft.train_distributed \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --math_dataset_name "BytedTsinghua-SIA/DAPO-Math-17k" \
    --total_episodes 50000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

### Key Features

1. **FSDP (Fully Sharded Data Parallel)**
   - Shards model, optimizer, and gradients across GPUs
   - Enables training larger models with limited GPU memory
   - Automatic wrapping with `TRANSFORMER_BASED_WRAP`

2. **vLLM Support** (optional)
   - High-throughput generation using vLLM
   - Enable with `--use_vllm` flag
   - Configure memory with `--vllm_gpu_memory_utilization`

3. **Mixed Precision**
   - BF16 training by default for efficiency
   - Configurable via `--dtype`

### Distributed Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_vllm` | False | Enable vLLM for generation |
| `--vllm_gpu_memory_utilization` | 0.3 | GPU memory fraction for vLLM |
| `--vllm_tensor_parallel_size` | 1 | Tensor parallel size for vLLM |

### Example Configuration (4 GPUs)

```bash
# Effective batch size = 4 GPUs × 1 batch × 8 accumulation = 32
accelerate launch --num_processes 4 --use_fsdp ... \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

### Python API

```python
from online_sft import (
    DistributedOnlineSFTTrainer,
    DistributedOnlineSFTConfig,
)
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=8,
    mixed_precision="bf16",
)

cfg = DistributedOnlineSFTConfig(
    beta=0.1,
    max_new_tokens=3072,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.3,
)

trainer = DistributedOnlineSFTTrainer(
    policy=policy,
    ref_policy=ref_policy,
    tokenizer=tokenizer,
    accelerator=accelerator,
    reward_fn=reward_fn,
    cfg=cfg,
)
```
