# online_sft/train.py
# Unified training script for Online SFT supporting multiple tasks
# Usage:
#   python -m online_sft.train --task_type tldr [args...]
#   python -m online_sft.train --task_type math [args...]

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from online_sft.trainer import OnlineSFTTrainer, OnlineSFTTrainerConfig
from online_sft.dataset_utils import (
    load_tldr_dataset,
    load_math_dataset,
    collate_fn,
    collate_fn_with_ground_truth,
)
from online_sft.reward_functions import (
    NeuralRewardModel,
    MathVerificationReward,
)
from online_sft.scheduler_utils import (
    create_scheduler,
    estimate_total_steps,
    print_scheduler_summary,
)
from online_sft.logging_utils import (
    format_metrics_log,
    StepTimer,
    print_training_header,
)


@dataclass
class ScriptArguments:
    """Arguments for dataset and experiment configuration."""

    # Task configuration
    task_type: str = field(
        default="tldr",
        metadata={"help": "Task type: tldr, math"},
    )

    # Dataset (TLDR)
    dataset_name: str = field(
        default="trl-internal-testing/tldr-preference-sft-trl-style",
        metadata={"help": "Dataset name for TLDR task"},
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Training split name"}
    )
    dataset_test_split: str = field(
        default="validation", metadata={"help": "Validation split name"}
    )

    # Dataset (Math)
    math_dataset_name: str = field(
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        metadata={"help": "Dataset name for math task"},
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional separate eval dataset (e.g., BytedTsinghua-SIA/AIME-2024)"},
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
        default=512, metadata={"help": "Maximum prompt length in tokens"}
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
        default="EleutherAI/pythia-1b-deduped",
        metadata={"help": "Base model name or path"},
    )
    sft_model_path: str = field(
        default="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
        metadata={"help": "SFT model checkpoint to start from"},
    )
    reward_model_path: Optional[str] = field(
        default="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
        metadata={"help": "Reward model path (for neural reward)"},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading models"}
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Model dtype: auto, float32, float16, bfloat16"},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation: eager, sdpa, flash_attention_2"},
    )


@dataclass
class TrainingConfig:
    """Training configuration arguments."""

    # Output and logging
    output_dir: str = field(
        default="output/online-sft",
        metadata={"help": "Output directory for checkpoints and logs"},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Wandb run name"}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every N steps"}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every N steps"}
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Evaluate every N steps"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy: no, steps, epoch"},
    )

    # Training hyperparameters
    total_episodes: int = field(
        default=10000, metadata={"help": "Total number of training episodes"}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=8, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=3e-6, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
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
        default=5e-7,
        metadata={"help": "Minimum learning rate (eta_min) for cosine decay"},
    )

    # Online SFT specific
    beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty coefficient (beta)"},
    )
    response_length: int = field(
        default=256, metadata={"help": "Maximum response length in tokens"}
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Nucleus sampling top-p"}
    )
    top_k: Optional[int] = field(
        default=None, metadata={"help": "Top-k sampling"}
    )
    num_return_sequences: int = field(
        default=8,
        metadata={"help": "Number of sequences to generate per prompt"},
    )
    do_sample: bool = field(
        default=True, metadata={"help": "Use sampling for generation"}
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
        default=-200.0,
        metadata={"help": "Clamp log prob from below to prevent explosion"},
    )
    normalize_logprob_by_length: bool = field(
        default=False,
        metadata={"help": "Normalize log prob by response length"},
    )
    use_amp: bool = field(
        default=False, metadata={"help": "Use automatic mixed precision"}
    )

    # Reward config
    reward_scale: float = field(
        default=1.0, metadata={"help": "Scale reward values"}
    )
    reward_shift: float = field(
        default=0.0, metadata={"help": "Shift reward values"}
    )

    # Math-specific reward config
    correct_reward: float = field(
        default=1.0, metadata={"help": "Reward for correct math answer"}
    )
    incorrect_reward: float = field(
        default=-1.0, metadata={"help": "Reward for incorrect math answer"}
    )
    apply_overlong_penalty: bool = field(
        default=True, metadata={"help": "Apply overlong penalty for math"}
    )
    max_response_length: int = field(
        default=20480, metadata={"help": "Max response length for overlong penalty"}
    )
    overlong_buffer_len: int = field(
        default=4096, metadata={"help": "Buffer length for overlong penalty"}
    )
    overlong_penalty_factor: float = field(
        default=1.0, metadata={"help": "Overlong penalty factor"}
    )

    # Misc
    seed: int = field(default=42, metadata={"help": "Random seed"})
    use_wandb: bool = field(
        default=True, metadata={"help": "Use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="online-sft-rlhf", metadata={"help": "Wandb project name"}
    )


def evaluate(trainer, eval_dataloader, num_eval_batches=10, use_ground_truths=False):
    """Run evaluation on a subset of the validation set."""
    metrics_list = []

    for i, batch in enumerate(eval_dataloader):
        if i >= num_eval_batches:
            break

        if use_ground_truths:
            prompt_batch, ground_truths = batch
        else:
            prompt_batch = batch
            ground_truths = None

        with torch.no_grad():
            metrics = trainer.step(prompt_batch, ground_truths=ground_truths)

        metrics_list.append(metrics)

    if not metrics_list:
        return {}

    agg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            agg_metrics[f"eval/{key}"] = sum(values) / len(values)

    return agg_metrics


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, TrainingConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Set up output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Initialize wandb
    if training_args.use_wandb:
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name or os.path.basename(training_args.output_dir),
            config={
                **vars(script_args),
                **vars(training_args),
                **vars(model_args),
            },
        )

    # Set random seed
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    ################
    # Model & Tokenizer
    ################
    print("Loading models and tokenizer...")
    print(f"Task type: {script_args.task_type}")

    # Determine dtype
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load policy model
    print(f"Loading policy from {model_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.sft_model_path,
        **model_kwargs,
    )
    policy.resize_token_embeddings(len(tokenizer))

    # Load reference policy
    print(f"Loading reference policy from {model_args.sft_model_path}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_args.sft_model_path,
        **model_kwargs,
    )
    ref_policy.resize_token_embeddings(len(tokenizer))

    ################
    # Dataset & Reward
    ################
    reward_fn = None
    reward_model = None
    use_ground_truths = False
    reward_lookup = None

    if script_args.task_type == "tldr":
        print(f"Loading TLDR dataset {script_args.dataset_name}...")

        with PartialState().local_main_process_first():
            train_dataset, eval_dataset = load_tldr_dataset(
                dataset_name=script_args.dataset_name,
                train_split=script_args.dataset_train_split,
                val_split=script_args.dataset_test_split,
                tokenizer=tokenizer,
                max_prompt_length=script_args.max_prompt_length,
                num_proc=script_args.dataset_num_proc,
            )

        # Load neural reward model
        if model_args.reward_model_path:
            print(f"Loading reward model from {model_args.reward_model_path}")
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_args.reward_model_path,
                trust_remote_code=model_args.trust_remote_code,
                num_labels=1,
                device_map="auto",
            )

        current_collate_fn = collate_fn

    elif script_args.task_type == "math":
        print(f"Loading Math dataset {script_args.math_dataset_name}...")

        with PartialState().local_main_process_first():
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

        # Create math verification reward function
        reward_fn = MathVerificationReward(
            reward_lookup=reward_lookup,
            tokenizer=tokenizer,
            correct_reward=training_args.correct_reward,
            incorrect_reward=training_args.incorrect_reward,
            apply_overlong_penalty=training_args.apply_overlong_penalty,
            max_response_length=training_args.max_response_length,
            overlong_buffer_len=training_args.overlong_buffer_len,
            overlong_penalty_factor=training_args.overlong_penalty_factor,
        )
        use_ground_truths = True
        current_collate_fn = collate_fn_with_ground_truth

    else:
        raise ValueError(f"Unknown task type: {script_args.task_type}")

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    ################
    # Trainer Setup
    ################
    print("Initializing OnlineSFTTrainer...")

    trainer_cfg = OnlineSFTTrainerConfig(
        beta=training_args.beta,
        max_new_tokens=training_args.response_length,
        temperature=training_args.temperature,
        top_k=training_args.top_k,
        top_p=training_args.top_p,
        do_sample=training_args.do_sample,
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
        use_amp=training_args.use_amp,
        denom_eps=training_args.denom_eps,
        clip_weight_value=training_args.clip_weight_value,
        clip_logprob_min=training_args.clip_logprob_min,
        normalize_logprob_by_length=training_args.normalize_logprob_by_length,
        reward_scale=training_args.reward_scale,
        reward_shift=training_args.reward_shift,
    )

    trainer = OnlineSFTTrainer(
        policy=policy,
        ref_policy=ref_policy,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_fn=reward_fn,
        cfg=trainer_cfg,
    )

    # Set prompt source for backfilling (DAPO-style)
    # This allows the trainer to sample new prompts when groups are filtered
    trainer.set_prompt_source(train_dataset)
    print(f"Prompt source set for backfilling ({len(train_dataset)} samples)")

    # Setup scheduler
    est_total_steps = estimate_total_steps(
        training_args.total_episodes,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
    )
    warmup_steps = int(training_args.warmup_ratio * est_total_steps)
    cosine_steps = max(1, est_total_steps - warmup_steps)

    scheduler = create_scheduler(
        trainer.optimizer,
        total_steps=est_total_steps,
        warmup_ratio=training_args.warmup_ratio,
        min_learning_rate=training_args.min_learning_rate,
    )
    trainer.scheduler = scheduler

    print_scheduler_summary(
        total_steps=est_total_steps,
        warmup_steps=warmup_steps,
        cosine_steps=cosine_steps,
        base_lr=training_args.learning_rate,
        min_lr=training_args.min_learning_rate,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=current_collate_fn,
    )

    eval_dataloader = None
    if eval_dataset and training_args.eval_strategy != "no":
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=current_collate_fn,
        )

    ################
    # Training Loop
    ################
    # Print detailed training header
    print_training_header(
        model_name=model_args.sft_model_path,
        dataset_name=script_args.math_dataset_name if script_args.task_type == "math" else script_args.dataset_name,
        num_gpus=1,  # Single GPU for non-distributed
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

        # Handle batch format
        if use_ground_truths:
            prompt_batch, ground_truths = batch
        else:
            prompt_batch = batch
            ground_truths = None

        # Training step with timing
        step_timer.start()
        metrics = trainer.step(prompt_batch, ground_truths=ground_truths)
        step_time = step_timer.stop()

        episode += len(prompt_batch)
        global_step += 1

        # Add timing to metrics
        metrics["timing/step_time"] = step_time

        # Logging
        if global_step % training_args.logging_steps == 0:
            metrics["episode"] = episode
            metrics["global_step"] = global_step

            try:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                metrics["lr/current"] = float(current_lr)
            except Exception:
                pass

            # Print detailed DAPO-style log to console
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

            if training_args.use_wandb:
                wandb.log(metrics, step=global_step)

        # Evaluation
        if (
            training_args.eval_strategy == "steps"
            and eval_dataloader is not None
            and global_step % training_args.eval_steps == 0
        ):
            print(f"Running evaluation at step {global_step}...")
            eval_metrics = evaluate(
                trainer, eval_dataloader,
                use_ground_truths=use_ground_truths,
            )

            if eval_metrics:
                print(f"Eval metrics: {eval_metrics}")
                if training_args.use_wandb:
                    wandb.log(eval_metrics, step=global_step)

        # Save checkpoint
        if global_step % training_args.save_steps == 0:
            checkpoint_dir = os.path.join(
                training_args.output_dir, f"checkpoint-{global_step}"
            )
            print(f"Saving checkpoint to {checkpoint_dir}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            policy.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            state_path = os.path.join(checkpoint_dir, "training_state.json")
            with open(state_path, "w") as f:
                json.dump(
                    {
                        "global_step": global_step,
                        "episode": episode,
                    },
                    f,
                )

    ################
    # Final Save
    ################
    print("Training complete! Saving final model...")
    final_dir = os.path.join(training_args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if training_args.use_wandb:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()
