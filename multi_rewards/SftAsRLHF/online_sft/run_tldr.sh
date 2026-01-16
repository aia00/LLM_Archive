#!/bin/bash
# Run script for Online SFT on TLDR task
# This mirrors the original run_online_sft.sh

set -e

# Default values
OUTPUT_DIR=${OUTPUT_DIR:-"output/online-sft-tldr"}
WANDB_PROJECT=${WANDB_PROJECT:-"online-sft-rlhf"}

python -m online_sft.train \
    --task_type tldr \
    --dataset_name "trl-internal-testing/tldr-preference-sft-trl-style" \
    --dataset_train_split train \
    --dataset_test_split validation \
    --model_name_or_path "EleutherAI/pythia-1b-deduped" \
    --sft_model_path "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr" \
    --reward_model_path "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --total_episodes 30000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.1 \
    --min_learning_rate 1e-7 \
    --beta 0.1 \
    --response_length 128 \
    --temperature 1.0 \
    --top_p 0.9 \
    --num_return_sequences 4 \
    --do_sample True \
    --normalize_logprob_by_length True \
    --clip_logprob_min -10 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --seed 42 \
    "$@"
