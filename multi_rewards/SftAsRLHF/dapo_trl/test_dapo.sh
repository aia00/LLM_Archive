#!/bin/bash
# Quick test script for DAPO implementation
# Tests with a small model and limited dataset

set -e

echo "============================================================================"
echo "DAPO Quick Test"
echo "============================================================================"
echo "This script will test the DAPO implementation with:"
echo "  - Small model: Qwen/Qwen2.5-0.5B-Instruct"
echo "  - Limited dataset: 10 training samples"
echo "  - 2 generations per prompt"
echo "  - 2 training steps"
echo "============================================================================"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm

cd "$(dirname "$0")"

# Run with minimal settings for quick testing
export PROJECT_NAME="DAPO-Test"
export EXP_NAME="quick-test-$(date +%Y%m%d-%H%M%S)"
export MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
export MAX_TRAIN_SAMPLES="10"
export NUM_GENERATIONS="2"
export MAX_STEPS="2"
export PER_DEVICE_BATCH_SIZE="1"
export GRADIENT_ACCUMULATION_STEPS="1"
export MAX_COMPLETION_LENGTH="512"
export SAVE_STEPS="1"
export LOGGING_STEPS="1"

echo ""
echo "Starting test training..."
echo ""

python train_dapo.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "BytedTsinghua-SIA/DAPO-Math-17k" \
    --output_dir "./test_output/${EXP_NAME}" \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --num_train_epochs 1 \
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate 1e-6 \
    --num_generations ${NUM_GENERATIONS} \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --loss_type dapo \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_prompt_length 512 \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --overlong_buffer_len 128 \
    --apply_overlong_penalty \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --report_to none \
    --bf16

echo ""
echo "============================================================================"
echo "Test completed!"
echo "============================================================================"
