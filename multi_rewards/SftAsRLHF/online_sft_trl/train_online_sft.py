"""
Main training script for Online SFT using TRL's GRPO framework.
Implements the "Online SFT = RLHF" algorithm.
"""

import argparse
import os
import sys
from typing import Optional

import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dapo_trl.dataset_utils import create_reward_lookup, prepare_dapo_dataset
from dapo_trl.reward_function import compute_dapo_reward
from online_sft_trl import OnlineSFTConfig, OnlineSFTTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Online SFT Training with TRL")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        help="Device map for model loading ('none' for FSDP, 'auto' for model parallelism)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="Dataset name from HuggingFace",
    )
    parser.add_argument(
        "--train_split", type=str, default="train", help="Training split name"
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="Validation split name",
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="BytedTsinghua-SIA/AIME-2024",
        help="Dataset name for evaluation",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="Eval split name",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction to hold out for validation",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="Random seed for validation split",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./online_sft_output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=10, help="Number of warmup steps"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )

    # Online SFT specific hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL coefficient (beta parameter in Online SFT)",
    )
    parser.add_argument(
        "--denom_eps",
        type=float,
        default=1e-8,
        help="Epsilon for weight denominator",
    )
    parser.add_argument(
        "--clip_weight_value",
        type=float,
        default=1.0,
        help="Clip weight w to [-clip, +clip]. Set 0 to disable.",
    )
    parser.add_argument(
        "--clip_logprob_min",
        type=float,
        default=-10.0,
        help="Clamp log probabilities from below. Set large negative to disable.",
    )
    parser.add_argument(
        "--normalize_logprob_by_length",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Normalize log prob by response length",
    )

    # Generation settings
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help="Maximum prompt length",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=8192,
        help="Maximum completion length",
    )

    # Group filtering (DAPO-style)
    parser.add_argument(
        "--filter_groups_enable",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Enable group filtering",
    )
    parser.add_argument(
        "--filter_groups_metric",
        type=str,
        default="acc",
        choices=["acc", "reward"],
        help="Metric to use for group filtering: 'acc' (accuracy) or 'reward'",
    )
    parser.add_argument(
        "--filter_groups_min_std",
        type=float,
        default=0.0,
        help="Minimum reward std to keep a group",
    )
    parser.add_argument(
        "--max_filter_retries",
        type=int,
        default=10,
        help="Maximum retries when filtering finds no valid groups. Set 0 for infinite retries.",
    )

    # Reward settings
    parser.add_argument(
        "--apply_overlong_penalty",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Apply overlong reward shaping",
    )
    parser.add_argument(
        "--overlong_buffer_len",
        type=int,
        default=4096,
        help="Buffer length for overlong penalty",
    )
    parser.add_argument(
        "--overlong_penalty_factor",
        type=float,
        default=1.0,
        help="Penalty factor for overlong responses",
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Log every X steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluate every X steps"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting tool (tensorboard, wandb, none)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="OnlineSFT-TRL",
        help="W&B project name",
    )

    # Other arguments
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=False,
        help="Use vLLM for generation",
    )
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=["server", "colocate"],
        help="vLLM integration mode",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for vLLM",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def create_reward_function_with_lookup(
    reward_lookup, args, tokenizer: Optional[AutoTokenizer] = None
):
    """Create a reward function that uses the reward lookup to get ground truths."""

    def reward_fn(completions, prompts=None, **kwargs):
        gt_list = kwargs.get("ground_truth")

        batch_ground_truths = []
        for i in range(len(completions)):
            if gt_list is not None and i < len(gt_list):
                batch_ground_truths.append(gt_list[i])
                continue

            prompt_key = prompts[i] if prompts else ""
            if isinstance(prompt_key, str):
                batch_ground_truths.append(reward_lookup.get(prompt_key, ""))
            else:
                batch_ground_truths.append("")

        return compute_dapo_reward(
            completions=completions,
            prompts=prompts or [""] * len(completions),
            ground_truths=batch_ground_truths,
            tokenizer=tokenizer,
            max_response_length=args.max_completion_length,
            overlong_buffer_len=args.overlong_buffer_len,
            overlong_penalty_factor=args.overlong_penalty_factor,
            apply_overlong_penalty=args.apply_overlong_penalty,
        )

    return reward_fn


def main():
    """Main training function."""
    args = parse_args()

    # Accelerate state for multi-process awareness
    state = PartialState()

    # Set random seed with rank offset
    effective_seed = args.seed + state.process_index
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    # Create output directory
    if state.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    state.wait_for_everyone()

    if state.is_local_main_process:
        print("=" * 80)
        print("Online SFT Training with TRL")
        print("=" * 80)
        print(f"Model: {args.model_name_or_path}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Output directory: {args.output_dir}")
        print(f"Beta (KL coefficient): {args.beta}")
        print("=" * 80)

    # W&B setup
    if args.report_to and "wandb" in str(args.report_to).lower():
        if state.is_local_main_process:
            os.environ["WANDB_PROJECT"] = args.wandb_project
            print(f"Using W&B project: {args.wandb_project}")

    # vLLM setup
    if args.use_vllm:
        if state.is_local_main_process:
            print("\n[vLLM] use_vllm=True. Preparing vLLM runtime...")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_USE_V1", "1")

    # Load dataset
    if state.is_local_main_process:
        print("\n[1/4] Loading and preprocessing dataset...")
    with state.local_main_process_first():
        train_dataset, val_dataset, reward_lookup = prepare_dapo_dataset(
            dataset_name=args.dataset_name,
            train_split=args.train_split,
            val_split=args.val_split,
            eval_dataset_name=args.eval_dataset_name,
            eval_split=args.eval_split,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_prompt_length=args.max_prompt_length,
            val_fraction=args.val_fraction,
            val_seed=args.val_seed,
        )

    # Load model and tokenizer
    if state.is_local_main_process:
        print("\n[2/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    device_map_arg = (
        None
        if (
            args.device_map is not None
            and str(args.device_map).lower() == "none"
        )
        else args.device_map
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map=device_map_arg,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if state.is_local_main_process:
        print(f"Model loaded: {model.config.model_type}")
        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
        )

    # Create reward function
    if state.is_local_main_process:
        print("\n[3/4] Creating reward function...")
    reward_fn = create_reward_function_with_lookup(
        reward_lookup, args, tokenizer
    )

    # Configure Online SFT trainer
    if state.is_local_main_process:
        print("\n[4/4] Configuring Online SFT trainer...")

    training_args = OnlineSFTConfig(
        # Output and logging
        output_dir=args.output_dir,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        # Training hyperparameters
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        # Online SFT specific settings
        beta=args.beta,
        denom_eps=args.denom_eps,
        clip_weight_value=(
            args.clip_weight_value if args.clip_weight_value > 0 else None
        ),
        clip_logprob_min=args.clip_logprob_min,
        normalize_logprob_by_length=args.normalize_logprob_by_length,
        sync_ref_model=True,  # important for GRPO based model; untuned steps/weight!
        ref_model_sync_steps=200,
        # Group filtering
        filter_groups_enable=args.filter_groups_enable,
        filter_groups_min_std=args.filter_groups_min_std,
        max_filter_retries=args.max_filter_retries,
        # Generation settings
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        # Other settings
        bf16=args.bf16,
        seed=args.seed,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )

    if state.is_local_main_process:
        print("\nTraining configuration:")
        print(f"  Beta (KL coefficient): {args.beta}")
        print(f"  Weight denominator epsilon: {args.denom_eps}")
        print(f"  Clip weight value: {args.clip_weight_value}")
        print(f"  Clip log prob min: {args.clip_logprob_min}")
        print(f"  Normalize by length: {args.normalize_logprob_by_length}")
        print(f"  Generations per prompt: {args.num_generations}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size per device: {args.per_device_train_batch_size}")
        print(
            f"  Gradient accumulation steps: {args.gradient_accumulation_steps}"
        )
        print(f"  Group filtering: {args.filter_groups_enable}")
        print(f"  Group filtering metric: {args.filter_groups_metric}")
        print(f"  Max filter retries: {args.max_filter_retries}")

    # Initialize trainer
    trainer = OnlineSFTTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        filter_groups_enable=args.filter_groups_enable,
        filter_groups_metric=args.filter_groups_metric,
        filter_groups_min_std=args.filter_groups_min_std,
        max_filter_gen_batches=args.max_filter_retries,
    )

    if state.is_local_main_process:
        print("\n" + "=" * 80)
        print("Starting Online SFT training...")
        print("=" * 80 + "\n")

    # Train
    trainer.train()

    # Wait for all processes
    state.wait_for_everyone()

    # Save final model
    if state.is_local_main_process:
        print("\nSaving final model...")
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        print("\nTraining completed!")
        print(f"Model saved to: {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()
