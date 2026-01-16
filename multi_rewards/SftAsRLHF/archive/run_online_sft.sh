#!/bin/bash
# run_online_sft.sh
# Example launcher script for OnlineSFT training

# Single GPU example (matching PPO script)
# python train_online_sft.py \
#     --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
#     --dataset_test_split validation \
#     --learning_rate 3e-6 \
#     --output_dir output/pythia-1b-tldr-online-sft \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 64 \
#     --total_episodes 30000 \
#     --model_name_or_path EleutherAI/pythia-1b-deduped \
#     --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
#     --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
#     --response_length 53 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_steps 500 \
#     --logging_steps 10 \
#     --beta 0.1 \
#     --temperature 1.0 \
#     --top_p 0.9 \
#     --use_wandb true \
#     --wandb_project online-sft-rlhf

# For distributed training with DeepSpeed (uncomment to use):
# accelerate launch --config_file deepspeed_zero2.yaml \
#     train_online_sft.py \
#     --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
#     --dataset_test_split validation \
#     --output_dir output/pythia-1b-tldr-online-sft \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 16 \
#     --gradient_accumulation_steps 4 \
#     --total_episodes 100000 \
#     --model_name_or_path EleutherAI/pythia-1b-deduped \
#     --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
#     --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
#     --response_length 53 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --save_steps 500 \
#     --logging_steps 10 \
#     --beta 0.1


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_online_sft.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --warmup_ratio 0.1 \
    --min_learning_rate 1e-7 \
    --output_dir output/pythia-1b-tldr-online-sft \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --response_length 128 \
    --num_return_sequences 4 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 99999 \
    --logging_steps 5 \
    --beta 0.1 \
    --temperature 1.0 \
    --top_p 0.9 \
    --use_wandb true \
    --wandb_project online-sft-rlhf