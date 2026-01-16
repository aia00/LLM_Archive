# DAPO Training with TRL

This directory contains the implementation of DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) using Hugging Face's TRL library. This implementation reproduces the key features of the official DAPO algorithm from the [DAPO paper](https://dapo-sia.github.io/).

## Overview

DAPO is an improved version of GRPO (Group Relative Policy Optimization) that introduces four key enhancements:

1. **Clip-Higher**: Expanded upper clip range (0.28 vs 0.2) for better diversity
2. **Dynamic Sampling**: Filters prompts with perfect/zero accuracy to maintain consistent gradients
3. **Token-level Policy Gradient**: Uses token-level normalization for better handling of long chain-of-thought reasoning
4. **Overlong Reward Shaping**: Penalizes responses exceeding expected length to reduce noise

## Directory Structure

```
dapo_trl/
├── README.md                  # This file
├── train_dapo.py             # Main training script
├── dataset_utils.py          # Dataset loading and preprocessing
├── reward_function.py        # Math answer verification reward function
├── logging_utils.py          # Logging and monitoring utilities
├── run_dapo.sh              # Launch script with DAPO hyperparameters
└── test_dapo.sh             # Quick test script
```

## Installation

1. Activate the `llm` conda environment (which has TRL installed):
```bash
conda activate llm
```

2. Install any additional dependencies if needed:
```bash
pip install datasets transformers accelerate
```

## Usage

### Quick Test

Run a quick test with a small model and limited dataset:

```bash
cd dapo_trl
./test_dapo.sh
```

This will:
- Use Qwen2.5-0.5B-Instruct model
- Train on 10 samples with 2 generations per prompt
- Run for 2 steps to verify the implementation

### Full Training

For full training with default DAPO hyperparameters:

```bash
cd dapo_trl
./run_dapo.sh
```

### Custom Configuration

You can customize training by setting environment variables:

```bash
# Use a different model
export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# Adjust batch size
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=16

# Change number of generations
export NUM_GENERATIONS=16

# Run training
./run_dapo.sh
```

### Advanced Usage

For more control, run the training script directly:

```bash
python train_dapo.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset_name "BytedTsinghua-SIA/DAPO-Math-17k" \
    --output_dir "./outputs/my_experiment" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-6 \
    --num_generations 16 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type dapo \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_prompt_length 2048 \
    --max_completion_length 20480 \
    --overlong_buffer_len 4096 \
    --overlong_penalty_factor 1.0 \
    --apply_overlong_penalty \
    --bf16 \
    --report_to tensorboard
```

## Key Hyperparameters

### DAPO Specific
- `--loss_type`: Loss type (`dapo`, `grpo`, or `dr_grpo`). Use `dapo` for token-level normalization.
- `--epsilon`: Lower clip ratio (default: 0.2)
- `--epsilon_high`: Upper clip ratio (default: 0.28) - DAPO's "clip-higher" feature
- `--num_generations`: Number of generations per prompt (default: 16)

### Training
- `--learning_rate`: Learning rate (default: 1e-6)
- `--per_device_train_batch_size`: Batch size per device (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 32)
- `--max_grad_norm`: Max gradient norm for clipping (default: 1.0)

### Generation
- `--temperature`: Sampling temperature (default: 1.0)
- `--top_p`: Top-p sampling parameter (default: 1.0)
- `--max_prompt_length`: Maximum prompt length (default: 2048)
- `--max_completion_length`: Maximum completion length (default: 20480)

### Overlong Penalty
- `--apply_overlong_penalty`: Enable overlong reward shaping
- `--overlong_buffer_len`: Buffer length for overlong penalty (default: 4096)
- `--overlong_penalty_factor`: Penalty factor for overlong responses (default: 1.0)

## Dataset

The implementation uses the [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset, which contains:
- 17k+ math problems
- Step-by-step solution format
- Ground truth answers for verification
- Competition-level mathematics

The reward function performs rule-based math answer verification by:
1. Extracting the final answer from the solution
2. Normalizing both predicted and ground truth answers
3. Comparing for correctness (binary +1/-1 reward, matching original DAPO)
4. Applying overlong penalty if enabled

**Note**: The base reward is **+1.0 for correct** and **-1.0 for incorrect** answers, matching the original DAPO implementation. This provides a stronger learning signal compared to 0/1 rewards.

## Logging and Monitoring

### TensorBoard
By default, training logs are saved for TensorBoard:
```bash
tensorboard --logdir outputs/DAPO-TRL/
```

### Weights & Biases
To use W&B, set the report_to parameter:
```bash
export REPORT_TO="wandb"
./run_dapo.sh
```

### Metrics
The following metrics are logged during training:
- `reward/mean`: Average reward
- `reward/std`: Standard deviation of rewards
- `loss`: Training loss
- `learning_rate`: Current learning rate
- `clip_fraction`: Fraction of policy ratios that were clipped
- `entropy`: Policy entropy

## Differences from Official Implementation

While this implementation reproduces the core DAPO algorithm, there are some differences from the official verl-based implementation:

1. **Infrastructure**: Uses TRL's single-machine training instead of Ray-based distributed training
2. **Dynamic Sampling**: Group filtering (drop zero-std groups) is implemented with optional resampling retries (`--filter_groups_enable` + `--max_filter_gen_batches`). Defaults match the original (disabled).
3. **Ref / Entropy Options**: KL-to-ref and entropy bonus are exposed (`--beta`, `--entropy_coeff`) but default to 0 to match the original setup.
4. **vLLM Integration**: The official implementation uses vLLM for generation. TRL supports vLLM but this implementation uses standard HuggingFace generation.
4. **Dynamic Sampling**: Group filtering (drop prompts whose rewards have zero std across generations) is implemented via `--filter_groups_enable`. It avoids degenerate batches but does not currently resample new prompts to backfill dropped groups.

The key algorithmic features are preserved:
- ✅ Clip-Higher (epsilon_high parameter)
- ✅ Token-level normalization (loss_type="dapo")
- ✅ Overlong reward shaping
- ⚠️ Dynamic sampling (requires custom implementation)

## Troubleshooting

### Out of Memory
If you encounter OOM errors:
- Reduce `per_device_train_batch_size`
- Reduce `num_generations`
- Reduce `max_completion_length`
- Use gradient checkpointing (enabled by default)

### Slow Training
- Enable mixed precision with `--bf16`
- Use gradient accumulation to maintain effective batch size
- Consider using a smaller model for initial testing

### Poor Reward Scores
- Check that ground truth answers are being correctly extracted
- Verify the answer extraction patterns in `reward_function.py`
- Monitor the reward/mean metric during training

## Citation

If you use this implementation, please cite the original DAPO paper:

```bibtex
@article{dapo2024,
  title={DAPO: Efficient Reinforcement Learning for Large Language Models},
  author={ByteDance Research},
  year={2024}
}
```

## License

This implementation follows the Apache 2.0 license, consistent with both TRL and the original DAPO implementation.
