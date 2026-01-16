# online_sft/train_distributed.py
# Distributed training script for Online SFT with FSDP and vLLM support
#
# Usage:
#   accelerate launch --num_processes 4 --use_fsdp \
#       --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
#       --fsdp_sharding_strategy FULL_SHARD \
#       -m online_sft.train_distributed --task_type math [args...]

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

try:
    from accelerate import Accelerator, PartialState
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from online_sft.distributed_trainer import (
    DistributedOnlineSFTTrainer,
    DistributedOnlineSFTConfig,
)
from online_sft.dataset_utils import (
    load_math_dataset,
    collate_fn_with_ground_truth,
)
from online_sft.reward_functions import MathVerificationReward
from online_sft.scheduler_utils import create_scheduler, estimate_total_steps
from online_sft.logging_utils import (
    format_metrics_log,
    StepTimer,
    print_training_header,
)


@dataclass
class ScriptArguments:
    """Arguments for dataset and experiment configuration."""

    task_type: str = field(
        default="math",
        metadata={"help": "Task type: math"},
    )

    # Dataset (Math)
    math_dataset_name: str = field(
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        metadata={"help": "Dataset name for math task"},
    )
    eval_dataset_name: Optional[str] = field(
        default="BytedTsinghua-SIA/AIME-2024",
        metadata={"help": "Optional separate eval dataset"},
    )
    val_fraction: float = field(
        default=0.1,
        metadata={"help": "Validation fraction for math dataset"},
    )

    # Data processing
    dataset_num_proc: int = field(
        default=4, metadata={"help": "Number of processes for dataset mapping"}
    )
    max_prompt_length: int = field(
        default=1024, metadata={"help": "Maximum prompt length in tokens"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Max training samples (for debugging)"}
    )
    max_val_samples: Optional[int] = field(
        default=None, metadata={"help": "Max validation samples (for debugging)"}
    )


@dataclass
class ModelConfig:
    """Model configuration arguments."""

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Math-7B",
        metadata={"help": "Model name or path"},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading models"}
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: float32, float16, bfloat16"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation: eager, sdpa, flash_attention_2"},
    )


@dataclass
class TrainingConfig:
    """Training configuration arguments."""

    # Output and logging
    output_dir: str = field(
        default="output/online-sft-distributed",
        metadata={"help": "Output directory for checkpoints and logs"},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Wandb run name"}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Evaluate every N steps"}
    )

    # Training hyperparameters
    total_episodes: int = field(
        default=50000, metadata={"help": "Total number of training episodes"}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-6, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay"})
    adam_eps: float = field(default=1e-8, metadata={"help": "Adam epsilon"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Adam beta2"})
    grad_clip_norm: float = field(
        default=1.0, metadata={"help": "Gradient clipping norm"}
    )

    # Scheduler
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Fraction of total steps for linear warmup"},
    )
    min_learning_rate: float = field(
        default=1e-8,
        metadata={"help": "Minimum learning rate"},
    )

    # Online SFT specific
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient"},
    )
    response_length: int = field(
        default=3072, metadata={"help": "Maximum response length in tokens"}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=1.0, metadata={"help": "Nucleus sampling top-p"}
    )
    top_k: Optional[int] = field(
        default=None, metadata={"help": "Top-k sampling"}
    )
    num_return_sequences: int = field(
        default=8,
        metadata={"help": "Number of sequences to generate per prompt"},
    )

    # Group filtering (DAPO-style)
    filter_groups_enable: bool = field(
        default=True,
        metadata={"help": "Enable group filtering: keep only groups with reward variance > 0"},
    )
    filter_groups_min_std: float = field(
        default=0.0,
        metadata={"help": "Minimum reward std to keep a group (> 0 means must have variance)"},
    )
    max_filter_retries: int = field(
        default=10,
        metadata={"help": "Max retries if all groups are filtered"},
    )

    # Stabilization
    denom_eps: float = field(
        default=1e-8, metadata={"help": "Epsilon for weight denominator"}
    )
    clip_weight_value: Optional[float] = field(
        default=1.0, metadata={"help": "Clip weight to [-clip, +clip]"}
    )
    clip_logprob_min: Optional[float] = field(
        default=-10.0,
        metadata={"help": "Clamp log prob from below"},
    )
    normalize_logprob_by_length: bool = field(
        default=True,
        metadata={"help": "Normalize log prob by response length"},
    )

    # Reward config
    correct_reward: float = field(
        default=1.0, metadata={"help": "Reward for correct math answer"}
    )
    incorrect_reward: float = field(
        default=-1.0, metadata={"help": "Reward for incorrect math answer"}
    )
    apply_overlong_penalty: bool = field(
        default=False, metadata={"help": "Apply overlong penalty for math"}
    )
    max_response_length: int = field(
        default=3072, metadata={"help": "Max response length for overlong penalty"}
    )

    # vLLM settings
    use_vllm: bool = field(
        default=False, metadata={"help": "Use vLLM for generation"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3, metadata={"help": "GPU memory fraction for vLLM"}
    )
    vllm_tensor_parallel_size: int = field(
        default=1, metadata={"help": "Tensor parallel size for vLLM"}
    )
    vllm_sync_interval: int = field(
        default=1, metadata={"help": "Sync weights to vLLM every N steps (colocate mode)"}
    )

    # Misc
    seed: int = field(default=42, metadata={"help": "Random seed"})
    wandb_project: str = field(
        default="online-sft-distributed", metadata={"help": "Wandb project name"}
    )
    report_to: str = field(
        default="wandb", metadata={"help": "Reporting tool: wandb, tensorboard, none"}
    )


def main():
    if not HAS_ACCELERATE:
        raise ImportError("accelerate is required for distributed training. Install with: pip install accelerate")

    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, TrainingConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision="bf16" if model_args.dtype == "bfloat16" else "no",
        log_with=training_args.report_to if training_args.report_to != "none" else None,
    )

    # Set up output directory
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize wandb on main process
    if accelerator.is_main_process and training_args.report_to == "wandb":
        accelerator.init_trackers(
            project_name=training_args.wandb_project,
            config={
                **vars(script_args),
                **vars(training_args),
                **vars(model_args),
            },
            init_kwargs={"wandb": {"name": training_args.run_name}},
        )

    # Set random seed with rank offset
    effective_seed = training_args.seed + accelerator.process_index
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    ################
    # Model & Tokenizer
    ################
    accelerator.print("=" * 80)
    accelerator.print("Online SFT Distributed Training")
    accelerator.print("=" * 80)
    accelerator.print(f"Model: {model_args.model_name_or_path}")
    accelerator.print(f"Task: {script_args.task_type}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print("=" * 80)

    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_args.dtype, torch.bfloat16)

    # Load tokenizer
    accelerator.print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models without device_map (FSDP will handle distribution)
    accelerator.print("\n[2/4] Loading models...")
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
        # Don't use device_map with FSDP - let FSDP handle sharding
    )
    if model_args.attn_implementation:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    accelerator.print(f"Model parameters: {sum(p.numel() for p in policy.parameters()) / 1e9:.2f}B")

    ################
    # Dataset & Reward
    ################
    accelerator.print("\n[3/4] Loading dataset...")

    with accelerator.main_process_first():
        train_dataset, eval_dataset, reward_lookup = load_math_dataset(
            dataset_name=script_args.math_dataset_name,
            eval_dataset_name=script_args.eval_dataset_name,
            tokenizer=tokenizer,
            max_prompt_length=script_args.max_prompt_length,
            val_fraction=script_args.val_fraction,
            max_train_samples=script_args.max_train_samples,
            max_val_samples=script_args.max_val_samples,
            num_proc=script_args.dataset_num_proc,
        )

    # Create reward function
    reward_fn = MathVerificationReward(
        reward_lookup=reward_lookup,
        tokenizer=tokenizer,
        correct_reward=training_args.correct_reward,
        incorrect_reward=training_args.incorrect_reward,
        apply_overlong_penalty=training_args.apply_overlong_penalty,
        max_response_length=training_args.max_response_length,
    )

    accelerator.print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        accelerator.print(f"Validation samples: {len(eval_dataset)}")

    ################
    # Trainer Setup
    ################
    accelerator.print("\n[4/4] Initializing distributed trainer...")

    trainer_cfg = DistributedOnlineSFTConfig(
        beta=training_args.beta,
        max_new_tokens=training_args.response_length,
        temperature=training_args.temperature,
        top_k=training_args.top_k,
        top_p=training_args.top_p,
        num_return_sequences=training_args.num_return_sequences,
        # Group filtering
        filter_groups_enable=training_args.filter_groups_enable,
        filter_groups_min_std=training_args.filter_groups_min_std,
        max_filter_retries=training_args.max_filter_retries,
        # Optimization
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        adam_eps=training_args.adam_eps,
        adam_betas=(training_args.adam_beta1, training_args.adam_beta2),
        grad_clip_norm=training_args.grad_clip_norm,
        grad_accum_steps=training_args.gradient_accumulation_steps,
        denom_eps=training_args.denom_eps,
        clip_weight_value=training_args.clip_weight_value,
        clip_logprob_min=training_args.clip_logprob_min,
        normalize_logprob_by_length=training_args.normalize_logprob_by_length,
        use_vllm=training_args.use_vllm,
        vllm_gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=training_args.vllm_tensor_parallel_size,
        vllm_sync_interval=training_args.vllm_sync_interval,
    )

    trainer = DistributedOnlineSFTTrainer(
        policy=policy,
        ref_policy=ref_policy,
        tokenizer=tokenizer,
        accelerator=accelerator,
        reward_fn=reward_fn,
        cfg=trainer_cfg,
    )

    # Set prompt source for backfilling (DAPO-style)
    trainer.set_prompt_source(train_dataset)
    accelerator.print(f"Prompt source set for backfilling ({len(train_dataset)} samples)")

    # Setup scheduler
    est_total_steps = estimate_total_steps(
        training_args.total_episodes,
        training_args.per_device_train_batch_size * accelerator.num_processes,
        training_args.gradient_accumulation_steps,
    )

    scheduler = create_scheduler(
        trainer.optimizer,
        total_steps=est_total_steps,
        warmup_ratio=training_args.warmup_ratio,
        min_learning_rate=training_args.min_learning_rate,
    )
    trainer.scheduler = scheduler

    accelerator.print(f"\nScheduler: {est_total_steps} total steps, {int(training_args.warmup_ratio * est_total_steps)} warmup steps")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_with_ground_truth,
    )

    # Prepare dataloader with accelerator
    train_dataloader = accelerator.prepare(train_dataloader)

    ################
    # Training Loop
    ################
    # Print detailed training header (main process only)
    if accelerator.is_main_process:
        print_training_header(
            model_name=model_args.model_name_or_path,
            dataset_name=script_args.math_dataset_name,
            num_gpus=accelerator.num_processes,
            batch_size=training_args.per_device_train_batch_size,
            grad_accum=training_args.gradient_accumulation_steps,
            num_return_sequences=training_args.num_return_sequences,
            total_episodes=training_args.total_episodes,
            learning_rate=training_args.learning_rate,
            beta=training_args.beta,
            filter_enabled=training_args.filter_groups_enable,
        )

    global_step = 0
    episode = 0
    train_iter = iter(train_dataloader)
    step_timer = StepTimer()

    while episode < training_args.total_episodes:
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        prompt_batch, ground_truths = batch

        # Training step with timing
        step_timer.start()
        metrics = trainer.step(prompt_batch, ground_truths=ground_truths)
        step_time = step_timer.stop()

        # Update counts
        batch_episodes = len(prompt_batch) * accelerator.num_processes
        episode += batch_episodes
        global_step += 1

        # Add timing to metrics
        metrics["timing/step_time"] = step_time

        # Logging
        if global_step % training_args.logging_steps == 0:
            metrics["episode"] = episode
            metrics["global_step"] = global_step

            try:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                metrics["lr"] = float(current_lr)
            except Exception:
                pass

            # Print detailed DAPO-style log to console (main process only)
            if accelerator.is_main_process:
                # Use compact mode for frequent logging, detailed for less frequent
                use_detailed = (global_step % (training_args.logging_steps * 10) == 0) or global_step <= 5
                log_str = format_metrics_log(
                    metrics=metrics,
                    global_step=global_step,
                    episode=episode,
                    total_episodes=training_args.total_episodes,
                    step_time=step_time,
                    compact=not use_detailed,
                )
                print(log_str)

            # Log to tracker
            accelerator.log(metrics, step=global_step)

        # Save checkpoint
        if global_step % training_args.save_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = os.path.join(
                    training_args.output_dir, f"checkpoint-{global_step}"
                )
                print(f"Saving checkpoint to {checkpoint_dir}")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Unwrap and save
                unwrapped = accelerator.unwrap_model(trainer.policy)
                unwrapped.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                state_path = os.path.join(checkpoint_dir, "training_state.json")
                with open(state_path, "w") as f:
                    json.dump({"global_step": global_step, "episode": episode}, f)

    ################
    # Final Save
    ################
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\nTraining complete! Saving final model...")
        final_dir = os.path.join(training_args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(trainer.policy)
        unwrapped.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Model saved to: {final_dir}")

    accelerator.end_training()
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
