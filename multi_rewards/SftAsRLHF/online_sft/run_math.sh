#!/bin/bash
# Run script for Online SFT on Math task (adapted from DAPO settings)
# Uses rule-based math verification reward instead of neural reward model
# Includes group filtering: keeps only groups with reward variance > 0

set -e

# Default values
OUTPUT_DIR=${OUTPUT_DIR:-"output/online-sft-math"}
WANDB_PROJECT=${WANDB_PROJECT:-"online-sft-rlhf"}

python -m online_sft.train \
    --task_type math \
    --math_dataset_name "BytedTsinghua-SIA/DAPO-Math-17k" \
    --eval_dataset_name "BytedTsinghua-SIA/AIME-2024" \
    --model_name_or_path "Qwen/Qwen2.5-Math-7B" \
    --sft_model_path "Qwen/Qwen2.5-Math-7B" \
    --reward_model_path "" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --max_prompt_length 1024 \
    --val_fraction 0.1 \
    --total_episodes 50000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.1 \
    --min_learning_rate 1e-8 \
    --beta 0.1 \
    --response_length 3072 \
    --temperature 1.0 \
    --top_p 1.0 \
    --num_return_sequences 8 \
    --do_sample True \
    --filter_groups_enable True \
    --filter_groups_min_std 0.0 \
    --max_filter_retries 10 \
    --clip_logprob_min -10 \
    --correct_reward 1.0 \
    --incorrect_reward -1.0 \
    --apply_overlong_penalty False \
    --max_response_length 3072 \
    --logging_steps 2 \
    --eval_steps 100 \
    --save_steps 500000000000 \
    --seed 42 \
    "$@"
