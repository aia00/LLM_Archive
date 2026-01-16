#!/usr/bin/env bash

# Hyperparameters
MODEL_PATH="Qwen/Qwen2.5-Math-7B"
DATASET_NAME="BytedTsinghua-SIA/DAPO-Math-17k"
EVAL_DATASET_NAME="BytedTsinghua-SIA/AIME-2024"
OUTPUT_DIR="./outputs/DAPO-TRL/Qwen2.5-7b"

NUM_EPOCHS=1
MAX_STEPS=-1
LEARNING_RATE=1e-6
LR_WARMUP_STEPS=10
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

PER_DEVICE_BATCH_SIZE=1  # 1 prompt per forward pass
GRADIENT_ACCUMULATION_STEPS=8  # 4 processes × 1 × 8 × 8 = 256 samples per update
# Generate 3× more prompts than needed for filtering (verl-style oversampling)
# 3 prompts × 8 generations = 24 samples per generation batch
GENERATION_BATCH_SIZE=24  # 3× oversampling: allows filtering to find mixed-reward groups

LOSS_TYPE="dapo"
NUM_GENERATIONS=8  # 8 generations per prompt
EVAL_NUM_GENERATIONS=8
EPSILON=0.2
EPSILON_HIGH=0.28
CLIP_RATIO_C=10.0
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TEMPERATURE=1.0
EVAL_TOP_P=0.7

MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=3072  # Reduced from 3072 to reduce memory
EVAL_MAX_COMPLETION_LENGTH=3072  # Reduced from 3072
EVAL_SUBSET_SIZE=100

APPLY_OVERLONG_PENALTY=false
OVERLONG_BUFFER_LEN=2048
OVERLONG_PENALTY_FACTOR=1.0

# Dynamic re-generation: If filtering drops too many prompts (those with zero reward std),
# the system will sample new prompts and generate more completions to fill the batch.
# This matches verl's behavior of ensuring full batches after filtering.
FILTER_GROUPS_ENABLE=true
FILTER_GROUPS_METRIC="acc"  # Filter based on accuracy std (can also use "reward")
MAX_FILTER_GEN_BATCHES=10  # Max attempts to refill batch (0 = unlimited)

LOGGING_STEPS=1
SAVE_STEPS=100000000000000000
EVAL_STEPS=50
REPORT_TO="wandb"

BF16=true
SEED=42

# vLLM (set to true to enable)
USE_VLLM=true
VLLM_HOST="127.0.0.1"
VLLM_PORT="8010" # use non-default to avoid collision

# Optional sample limits (leave empty to use all)
MAX_TRAIN_SAMPLES=
MAX_VAL_SAMPLES=

# Export vLLM server env if enabled (used by train_dapo.py defaults)
if [[ "$USE_VLLM" == "true" ]]; then
  export TRL_VLLM_HOST="$VLLM_HOST"
  export TRL_VLLM_PORT="$VLLM_PORT"
fi

# Launch with FSDP across 4 GPUs (one process per GPU)
# FSDP shards model/optimizer/gradients across all 4 GPUs for memory efficiency
accelerate launch \
    --num_processes 4 \
    --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_offload_params false \
    "$(dirname "$0")/train_dapo.py" \
    --device_map none \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --eval_dataset_name "$EVAL_DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --generation_batch_size $GENERATION_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --num_generations $NUM_GENERATIONS \
    --eval_num_generations $EVAL_NUM_GENERATIONS \
    --epsilon $EPSILON \
    --epsilon_high $EPSILON_HIGH \
    --clip_ratio_c $CLIP_RATIO_C \
    --loss_type $LOSS_TYPE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --eval_temperature $EVAL_TEMPERATURE \
    --eval_top_p $EVAL_TOP_P \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --eval_max_completion_length $EVAL_MAX_COMPLETION_LENGTH \
    --overlong_buffer_len $OVERLONG_BUFFER_LEN \
    --overlong_penalty_factor $OVERLONG_PENALTY_FACTOR \
    ${FILTER_GROUPS_ENABLE:+--filter_groups_enable} \
    --filter_groups_metric $FILTER_GROUPS_METRIC \
    --max_filter_gen_batches $MAX_FILTER_GEN_BATCHES \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --val_fraction 0.00003 \
    --report_to $REPORT_TO \
    --wandb_project "DAPO-TRL" \
    --seed $SEED \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 1 \
    ${USE_VLLM:+--use_vllm} \
    --apply_overlong_penalty $APPLY_OVERLONG_PENALTY \
    ${BF16:+--bf16} \
    ${MAX_TRAIN_SAMPLES:+--max_train_samples $MAX_TRAIN_SAMPLES} \
    ${MAX_VAL_SAMPLES:+--max_val_samples $MAX_VAL_SAMPLES} \
    ${EVAL_SUBSET_SIZE:+--eval_subset_size $EVAL_SUBSET_SIZE}
