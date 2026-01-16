# train_online_sft.py
# Training script for OnlineSFTTrainer that mirrors the TRL PPO example
# This implements the "SFT = RLHF" algorithm from the paper

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from online_sft_trainer import OnlineSFTTrainer, OnlineSFTTrainerConfig


@dataclass
class ScriptArguments:
    """Arguments for dataset and experiment configuration."""

    dataset_name: str = field(
        default="trl-internal-testing/tldr-preference-sft-trl-style",
        metadata={"help": "Dataset name from HuggingFace hub"},
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration name"}
    )
    dataset_train_split: str = field(
        default="train", metadata={"help": "Training split name"}
    )
    dataset_test_split: str = field(
        default="validation", metadata={"help": "Validation split name"}
    )
    dataset_num_proc: int = field(
        default=4, metadata={"help": "Number of processes for dataset mapping"}
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
    reward_model_path: str = field(
        default="cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
        metadata={"help": "Reward model path"},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading models"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "Model revision to use"}
    )
    dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Model dtype: auto, float32, float16, bfloat16"},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": "Attention implementation: eager, sdpa, flash_attention_2"
        },
    )


@dataclass
class TrainingConfig:
    """Training configuration arguments."""

    # Output and logging
    output_dir: str = field(
        default="output/online-sft-tldr",
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
        metadata={"help": "KL penalty coefficient (β in the paper)"},
    )
    response_length: int = field(
        default=53, metadata={"help": "Maximum response length in tokens"}
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
        default=1,
        metadata={"help": "Number of sequences to generate per prompt"},
    )
    do_sample: bool = field(
        default=True, metadata={"help": "Use sampling for generation"}
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
        metadata={
            "help": "Clamp log π(y|x) from below to prevent loss explosion. "
            "Use larger values (e.g., -200) for sequence-level, smaller values (e.g., -0.5) for per-token normalization. "
            "Set to None to disable."
        },
    )
    normalize_logprob_by_length: bool = field(
        default=False,
        metadata={
            "help": "If True, normalize log π(y|x) by response length before computing loss. "
            "Prevents long sequences from dominating. When enabled, use smaller clip_logprob_min values."
        },
    )
    use_amp: bool = field(
        default=False, metadata={"help": "Use automatic mixed precision"}
    )

    # Reward model config
    reward_scale: float = field(
        default=1.0, metadata={"help": "Scale reward values"}
    )
    reward_shift: float = field(
        default=0.0, metadata={"help": "Shift reward values"}
    )

    # Misc
    seed: int = field(default=42, metadata={"help": "Random seed"})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Push model to HuggingFace Hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "Hub model ID for pushing"}
    )
    use_wandb: bool = field(
        default=True, metadata={"help": "Use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="online-sft-rlhf", metadata={"help": "Wandb project name"}
    )


def prepare_dataset(dataset, tokenizer, max_prompt_length=512):
    """Pre-tokenize the dataset before training; only collate during training."""

    def tokenize(element):
        # Access the prompt string from the messages field
        # Format: messages[0]["content"] contains the full prompt
        prompt_string = element["messages"][0]["content"]

        # Tokenize the prompt
        tokenized_output = tokenizer(
            prompt_string,
            padding=False,
            add_special_tokens=True,
        )

        input_ids_list = tokenized_output["input_ids"]

        return {
            "input_ids": input_ids_list,
            "lengths": len(input_ids_list),
            "prompt_text": prompt_string,
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=[
            c for c in dataset.column_names if c not in ["prompt_text"]
        ],
        num_proc=4,
    )

    # Filter out prompts that are too long
    dataset = dataset.filter(
        lambda x: x["lengths"] <= max_prompt_length,
        num_proc=4,
    )

    return dataset


def collate_fn(batch):
    """Simple collate function that returns a list of prompt texts."""
    return [item["prompt_text"] for item in batch]


def evaluate(trainer, eval_dataloader, num_eval_batches=10):
    """Run evaluation on a subset of the validation set."""
    metrics_list = []

    for i, prompt_batch in enumerate(eval_dataloader):
        if i >= num_eval_batches:
            break

        # Run one step without gradient updates (eval mode handled in trainer)
        with torch.no_grad():
            metrics = trainer.step(prompt_batch)

        metrics_list.append(metrics)

    # Aggregate metrics
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
    script_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    # Set up output directory
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Initialize wandb if enabled
    if training_args.use_wandb:
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.run_name
            or os.path.basename(training_args.output_dir),
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
    # Quick CUDA diagnostics
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
            print(f"CUDA_VISIBLE_DEVICES={visible}")
            n_devs = torch.cuda.device_count()
            print(f"Visible CUDA devices: {n_devs}")
            for i in range(n_devs):
                print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
        except Exception:
            pass

    # Determine dtype
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
        device_map="auto",  # Automatically map model to available GPU(s)
    )
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    # Add pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load models
    print(f"Loading policy from {model_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.sft_model_path,
        **model_kwargs,
    )
    # Resize embeddings if we added tokens
    policy.resize_token_embeddings(len(tokenizer))
    # Log device placement for policy
    try:
        print(f"Policy has hf_device_map: {hasattr(policy, 'hf_device_map')}")
        if hasattr(policy, "hf_device_map"):
            print(f"Policy device map: {policy.hf_device_map}")
        first_dev = next(policy.parameters()).device
        print(f"Policy first param device: {first_dev}")
    except Exception:
        pass

    print(f"Loading reference policy from {model_args.sft_model_path}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_args.sft_model_path,
        **model_kwargs,
    )
    ref_policy.resize_token_embeddings(len(tokenizer))
    # Log device placement for ref policy
    try:
        print(
            f"Ref policy has hf_device_map: {hasattr(ref_policy, 'hf_device_map')}"
        )
        if hasattr(ref_policy, "hf_device_map"):
            print(f"Ref policy device map: {ref_policy.hf_device_map}")
        first_dev_ref = next(ref_policy.parameters()).device
        print(f"Ref policy first param device: {first_dev_ref}")
    except Exception:
        pass

    print(f"Loading reward model from {model_args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        device_map="auto",
    )
    try:
        print(
            f"Reward model first param device (pre-trainer): {next(reward_model.parameters()).device}"
        )
    except Exception:
        pass

    ################
    # Dataset
    ################
    print(f"Loading dataset {script_args.dataset_name}...")
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config
    )

    # Prepare datasets with tokenization
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(
            dataset[script_args.dataset_train_split],
            tokenizer,
        )
        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_dataset = prepare_dataset(
                dataset[script_args.dataset_test_split],
                tokenizer,
            )

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    ################
    # Trainer Setup
    ################
    print("Initializing OnlineSFTTrainer...")

    # Create trainer config
    trainer_cfg = OnlineSFTTrainerConfig(
        beta=training_args.beta,
        max_new_tokens=training_args.response_length,
        temperature=training_args.temperature,
        top_k=training_args.top_k,
        top_p=training_args.top_p,
        do_sample=training_args.do_sample,
        num_return_sequences=training_args.num_return_sequences,
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

    # Initialize trainer
    trainer = OnlineSFTTrainer(
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        cfg=trainer_cfg,
    )

    # Setup LR scheduler: linear warmup + cosine decay to min LR
    # Estimate total optimizer steps accounting for gradient accumulation.
    # Each loop iteration processes one micro-batch with size `per_device_train_batch_size`.
    # One optimizer step occurs every `gradient_accumulation_steps` micro-batches.
    est_microbatches = max(
        1,
        (
            training_args.total_episodes
            + training_args.per_device_train_batch_size
            - 1
        )
        // training_args.per_device_train_batch_size,
    )
    est_total_steps = max(
        1, est_microbatches // training_args.gradient_accumulation_steps
    )
    warmup_steps = int(
        max(0.0, min(1.0, training_args.warmup_ratio)) * est_total_steps
    )
    cosine_steps = max(1, est_total_steps - warmup_steps)

    if warmup_steps > 0:
        # torch.optim.lr_scheduler.LinearLR requires 0 < start_factor <= 1
        # Use a tiny positive value to approximate zero without violating constraints
        warmup_start = 0.05
        warmup_sched = LinearLR(
            trainer.optimizer,
            start_factor=warmup_start,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_sched = CosineAnnealingLR(
            trainer.optimizer,
            T_max=cosine_steps,
            eta_min=training_args.min_learning_rate,
        )
        # Final clamp stage: keep LR fixed at eta_min once cosine finishes
        # Compute per-param-group scale factors so LambdaLR holds LR at eta_min
        base_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        clamp_factors = [
            (training_args.min_learning_rate / blr) if blr > 0 else 0.0
            for blr in base_lrs
        ]
        clamp_lambdas = [
            (lambda f: (lambda _: f))(f) for f in clamp_factors
        ]
        clamp_sched = LambdaLR(trainer.optimizer, lr_lambda=clamp_lambdas)
        scheduler = SequentialLR(
            trainer.optimizer,
            schedulers=[warmup_sched, cosine_sched, clamp_sched],
            milestones=[warmup_steps, warmup_steps + cosine_steps],
        )
    else:
        cosine_sched = CosineAnnealingLR(
            trainer.optimizer,
            T_max=cosine_steps,
            eta_min=training_args.min_learning_rate,
        )
        base_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        clamp_factors = [
            (training_args.min_learning_rate / blr) if blr > 0 else 0.0
            for blr in base_lrs
        ]
        clamp_lambdas = [
            (lambda f: (lambda _: f))(f) for f in clamp_factors
        ]
        clamp_sched = LambdaLR(trainer.optimizer, lr_lambda=clamp_lambdas)
        scheduler = SequentialLR(
            trainer.optimizer,
            schedulers=[cosine_sched, clamp_sched],
            milestones=[cosine_steps],
        )

    trainer.scheduler = scheduler
    # Clear, user-friendly scheduler summary
    base_lr = training_args.learning_rate
    print("LR Scheduler: Linear Warmup + Cosine Decay")
    print(
        f"  Optimizer steps (estimated): {est_total_steps}"
        f" | Microbatches (estimated): {est_microbatches}"
        f" | Accumulation: {training_args.gradient_accumulation_steps}"
    )
    print(
        f"  Warmup: {warmup_steps} steps ({training_args.warmup_ratio*100:.1f}% of optimizer steps)"
    )
    print(
        f"  Cosine decay: {cosine_steps} steps | base_lr={base_lr} -> min_lr={training_args.min_learning_rate}"
    )
    print("  Clamp: hold min_lr after cosine (prevent end bump)")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,  # Ensure fixed microbatch size for exact step accounting
        collate_fn=collate_fn,
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    ################
    # Training Loop
    ################
    print("Starting training...")
    print(f"Total episodes: {training_args.total_episodes}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(
        f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}"
    )
    print(
        f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )

    global_step = 0
    episode = 0
    train_iter = iter(train_dataloader)

    while episode < training_args.total_episodes:
        # Get next batch
        try:
            prompt_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            prompt_batch = next(train_iter)

        # Training step
        metrics = trainer.step(prompt_batch)

        episode += len(prompt_batch)
        global_step += 1

        # Logging
        if global_step % training_args.logging_steps == 0:
            metrics["episode"] = episode
            metrics["global_step"] = global_step
            # Report current learning rate from optimizer (first param group)
            try:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                metrics["lr/current"] = float(current_lr)
                metrics["lr/base"] = float(base_lr)
                # Optimizer step index and phase info
                opt_step = None
                if getattr(trainer, "scheduler", None) is not None and hasattr(
                    trainer.scheduler, "last_epoch"
                ):
                    # PyTorch schedulers start with last_epoch=-1 before first step
                    opt_step = max(0, int(trainer.scheduler.last_epoch))
                    metrics["lr/opt_step"] = float(opt_step)
                    metrics["lr/total_opt_steps"] = float(est_total_steps)
                    in_warmup = int(opt_step < warmup_steps)
                    clamp_start = warmup_steps + cosine_steps
                    in_cosine = int((opt_step >= warmup_steps) and (opt_step < clamp_start))
                    in_clamp = int(opt_step >= clamp_start)
                    metrics["lr/in_warmup"] = float(in_warmup)
                    metrics["lr/in_cosine"] = float(in_cosine)
                    metrics["lr/in_clamp"] = float(in_clamp)
                    if in_warmup:
                        phase = "warmup"
                        phase_step = opt_step
                    elif in_cosine:
                        phase = "cosine"
                        phase_step = max(0, opt_step - warmup_steps)
                    else:
                        phase = "clamp"
                        phase_step = max(0, opt_step - clamp_start)
                    metrics["lr/phase"] = phase
                    metrics["lr/phase_step"] = float(phase_step)
                # Extend console log with LR and phase info
                clamp_start = warmup_steps + cosine_steps
                if opt_step is not None and opt_step < warmup_steps:
                    phase_total = warmup_steps
                    log_phase = "warmup"
                elif opt_step is not None and opt_step < clamp_start:
                    phase_total = cosine_steps
                    log_phase = "cosine"
                else:
                    phase_total = max(1, est_total_steps - clamp_start)
                    log_phase = "clamp"
                log_str += f" | LR: {current_lr:.2e} ({log_phase} {phase_step}/{phase_total})"
            except Exception:
                pass

            # Print to console
            log_str = f"Step {global_step} | Episode {episode}/{training_args.total_episodes}"
            log_str += f" | Loss: {metrics.get('loss', 0):.4f}"
            log_str += f" | Reward: {metrics.get('reward/mean', 0):.4f}"
            log_str += (
                f" | KL: {metrics.get('policy/kl_y_ref_per_tok/mean', 0):.4f}"
            )
            print(log_str)

            # Log to wandb
            if training_args.use_wandb:
                wandb.log(metrics, step=global_step)

        # Evaluation
        if (
            training_args.eval_strategy == "steps"
            and eval_dataloader is not None
            and global_step % training_args.eval_steps == 0
        ):
            print(f"Running evaluation at step {global_step}...")
            eval_metrics = evaluate(trainer, eval_dataloader)

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

            # Save training state
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

    # Push to hub if requested
    if training_args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        hub_model_id = training_args.hub_model_id or training_args.output_dir
        policy.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    # Finish wandb
    if training_args.use_wandb:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()
