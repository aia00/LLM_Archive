#!/bin/bash
# Run script for Online SFT using TRL's GRPO framework
# Supports FSDP distributed training across multiple GPUs
#
# Usage:
#   ./run_online_sft.sh
#   ./run_online_sft.sh --use_vllm  # Enable vLLM for faster generation

set -e

# ============================================
# Hyperparameters
# ============================================

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}"
DATASET_NAME="${DATASET_NAME:-BytedTsinghua-SIA/DAPO-Math-17k}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-BytedTsinghua-SIA/AIME-2024}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/online-sft-trl}"

# Training settings
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1  # -1 for full epoch
LEARNING_RATE=1e-6
WARMUP_STEPS=10
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

# Batch settings (optimized for 4 GPUs)
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
# Effective batch: 4 GPUs × 1 × 32 = 128 samples per update

# Generation settings
MAX_COMPLETION_LENGTH=3072
MAX_PROMPT_LENGTH=1024
TEMPERATURE=1.0
TOP_P=1.0
NUM_GENERATIONS=8

# Online SFT specific settings
BETA=0.1                     # KL coefficient
DENOM_EPS=1e-8               # Epsilon for weight denominator
CLIP_WEIGHT_VALUE=1.0        # Clip weight to [-1, +1]
CLIP_LOGPROB_MIN=-400.0       # Clamp log probs from below
NORMALIZE_BY_LENGTH=false   # Normalize log prob by length

# Group filtering (DAPO-style)
FILTER_GROUPS_ENABLE=true    # Filter groups with no reward variance
FILTER_GROUPS_METRIC="acc"   # Use accuracy-based filtering (more reliable than reward)
FILTER_GROUPS_MIN_STD=0.0    # Minimum std to keep a group
MAX_FILTER_RETRIES=5         # Max retries before falling back (0 = infinite)

# Reward settings
APPLY_OVERLONG_PENALTY=false
OVERLONG_BUFFER_LEN=4096
OVERLONG_PENALTY_FACTOR=1.0

# Dataset settings
VAL_FRACTION=0.1
VAL_SEED=42
MAX_VAL_SAMPLES=48  # Limit eval dataset to this many samples (for AIME-2024, full set is 30, so 3 is ~10%)

# Logging
LOGGING_STEPS=1
EVAL_STEPS=50
SAVE_STEPS=5000000000000
WANDB_PROJECT="OnlineSFT-TRL"
REPORT_TO="wandb"

SEED=42

# ============================================
# vLLM Colocate Mode Settings
# ============================================
USE_VLLM=true
VLLM_MODE="colocate"
VLLM_GPU_MEMORY_UTILIZATION=0.15
VLLM_TENSOR_PARALLEL_SIZE=1

# Parse command line for --use_vllm flag
for arg in "$@"; do
    if [[ "$arg" == "--use_vllm" ]]; then
        USE_VLLM=true
    fi
done

# ============================================
# FSDP Configuration
# ============================================
NUM_PROCESSES=${NUM_PROCESSES:-4}  # Number of GPUs

echo "============================================"
echo "Online SFT Training with TRL (GRPO Framework)"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Num GPUs: $NUM_PROCESSES"
echo "Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Num generations per prompt: $NUM_GENERATIONS"
echo "Beta (KL coefficient): $BETA"
echo "Use vLLM: $USE_VLLM"
if [[ "$USE_VLLM" == "true" ]]; then
    echo "  - Mode: $VLLM_MODE"
    echo "  - GPU memory utilization: $VLLM_GPU_MEMORY_UTILIZATION"
    echo "  - Tensor parallel size: $VLLM_TENSOR_PARALLEL_SIZE"
fi
echo "============================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build vLLM args
VLLM_ARGS=""
if [[ "$USE_VLLM" == "true" ]]; then
    VLLM_ARGS="--use_vllm --vllm_mode $VLLM_MODE --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION"
    echo "[vLLM] $VLLM_MODE mode enabled"
fi

# Set environment variables for vLLM
if [[ "$USE_VLLM" == "true" ]]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_V1=1
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
    "$SCRIPT_DIR/train_online_sft.py" \
    --model_name_or_path "$MODEL_PATH" \
    --device_map none \
    --dataset_name "$DATASET_NAME" \
    --eval_dataset_name "$EVAL_DATASET_NAME" \
    --max_val_samples $MAX_VAL_SAMPLES \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --beta $BETA \
    --denom_eps $DENOM_EPS \
    --clip_weight_value $CLIP_WEIGHT_VALUE \
    --clip_logprob_min $CLIP_LOGPROB_MIN \
    --normalize_logprob_by_length $NORMALIZE_BY_LENGTH \
    --num_generations $NUM_GENERATIONS \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --filter_groups_enable $FILTER_GROUPS_ENABLE \
    --filter_groups_metric $FILTER_GROUPS_METRIC \
    --filter_groups_min_std $FILTER_GROUPS_MIN_STD \
    --max_filter_retries $MAX_FILTER_RETRIES \
    --apply_overlong_penalty $APPLY_OVERLONG_PENALTY \
    --overlong_buffer_len $OVERLONG_BUFFER_LEN \
    --overlong_penalty_factor $OVERLONG_PENALTY_FACTOR \
    --val_fraction $VAL_FRACTION \
    --val_seed $VAL_SEED \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --report_to "$REPORT_TO" \
    --bf16 \
    --seed $SEED \
    $VLLM_ARGS