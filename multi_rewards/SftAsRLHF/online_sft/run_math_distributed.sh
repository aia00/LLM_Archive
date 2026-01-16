#!/bin/bash
# Run script for Online SFT on Math task with FSDP and vLLM
# Uses Accelerate for distributed training across multiple GPUs
#
# Usage:
#   ./run_math_distributed.sh
#   ./run_math_distributed.sh --use_vllm  # Enable vLLM for faster generation

set -e

# ============================================
# Hyperparameters
# ============================================

MODEL_PATH="Qwen/Qwen2.5-Math-7B"
DATASET_NAME="BytedTsinghua-SIA/DAPO-Math-17k"
EVAL_DATASET_NAME="BytedTsinghua-SIA/AIME-2024"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/online-sft-math-distributed}"

# Training settings
TOTAL_EPISODES=50000
LEARNING_RATE=1e-6
WARMUP_RATIO=0.1
MIN_LEARNING_RATE=1e-8
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

# Batch settings (optimized for 4 GPUs)
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
# Effective batch: 4 GPUs × 1 × 8 = 32 samples per update

# Generation settings
RESPONSE_LENGTH=3072
TEMPERATURE=1.0
TOP_P=1.0
NUM_RETURN_SEQUENCES=8

# Group filtering (DAPO-style)
# Keep only groups with reward variance > 0 (at least one correct, one incorrect)
FILTER_GROUPS_ENABLE=True
FILTER_GROUPS_MIN_STD=0.0
MAX_FILTER_RETRIES=5

# Online SFT specific
BETA=0.1
CLIP_LOGPROB_MIN=-10
NORMALIZE_BY_LENGTH=True

# Reward settings
CORRECT_REWARD=1.0
INCORRECT_REWARD=-1.0
APPLY_OVERLONG_PENALTY=False

# Dataset settings
MAX_PROMPT_LENGTH=1024
VAL_FRACTION=0.1

# Logging
LOGGING_STEPS=1
EVAL_STEPS=100
SAVE_STEPS=500000000000000
WANDB_PROJECT="online-sft-distributed"
REPORT_TO="wandb"

SEED=42

# ============================================
# vLLM Colocate Mode Settings
# ============================================
# vLLM colocate mode uses distributed_executor_backend="external_launcher"
# so each rank creates its own vLLM engine on its assigned GPU.
# This allows parallel generation across all GPUs.
#
# tensor_parallel_size options:
#   - 1: Each GPU has its own vLLM (recommended, each generates independently)
#   - N: N GPUs share one vLLM via tensor parallelism (world_size must be divisible by N)
#
# Requires: pip install vllm>=0.4.0 (external_launcher support)
USE_VLLM=false
VLLM_GPU_MEMORY_UTILIZATION=0.2  # GPU memory fraction for vLLM per GPU
VLLM_TENSOR_PARALLEL_SIZE=1      # 1 = each GPU has its own vLLM instance
VLLM_SYNC_INTERVAL=1             # Sync weights every N steps (1 = every step)

# Parse command line for --use_vllm flag
for arg in "$@"; do
    if [[ "$arg" == "--use_vllm" ]]; then
        USE_VLLM=true
    fi
done

# ============================================
# FSDP Configuration
# ============================================
NUM_PROCESSES=4  # Number of GPUs

echo "============================================"
echo "Online SFT Distributed Training"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Num GPUs: $NUM_PROCESSES"
echo "Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Use vLLM (colocate): $USE_VLLM"
if [[ "$USE_VLLM" == "true" ]]; then
    echo "  - GPU memory utilization: $VLLM_GPU_MEMORY_UTILIZATION"
    echo "  - Tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
    echo "  - Weight sync interval: $VLLM_SYNC_INTERVAL"
fi
echo "============================================"

# Build vLLM args
VLLM_ARGS=""
if [[ "$USE_VLLM" == "true" ]]; then
    VLLM_ARGS="--use_vllm --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE --vllm_sync_interval $VLLM_SYNC_INTERVAL"
    echo "[vLLM] Colocate mode enabled"
fi

# Launch with FSDP across GPUs
# FSDP shards model/optimizer/gradients across all GPUs for memory efficiency
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_offload_params false \
    --mixed_precision bf16 \
    -m online_sft.train_distributed \
    --model_name_or_path "$MODEL_PATH" \
    --math_dataset_name "$DATASET_NAME" \
    --eval_dataset_name "$EVAL_DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --total_episodes $TOTAL_EPISODES \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --min_learning_rate $MIN_LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --grad_clip_norm $MAX_GRAD_NORM \
    --beta $BETA \
    --response_length $RESPONSE_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --filter_groups_enable $FILTER_GROUPS_ENABLE \
    --filter_groups_min_std $FILTER_GROUPS_MIN_STD \
    --max_filter_retries $MAX_FILTER_RETRIES \
    --clip_logprob_min $CLIP_LOGPROB_MIN \
    --normalize_logprob_by_length $NORMALIZE_BY_LENGTH \
    --correct_reward $CORRECT_REWARD \
    --incorrect_reward $INCORRECT_REWARD \
    --apply_overlong_penalty $APPLY_OVERLONG_PENALTY \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --val_fraction $VAL_FRACTION \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --report_to "$REPORT_TO" \
    --seed $SEED \
    $VLLM_ARGS \
    "$@"
