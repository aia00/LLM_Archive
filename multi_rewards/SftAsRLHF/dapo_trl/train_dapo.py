"""
Main training script for DAPO with TRL.
Reproduces the DAPO experiment using TRL's GRPOTrainer.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig

try:
    from .dynamic_grpo_trainer import DynamicGRPOTrainer  # package import
except ImportError:  # pragma: no cover - script mode fallback
    # When executed as a script (e.g., python dapo_trl/train_dapo.py),
    # the package context is missing; add the local folder to sys.path.
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.append(str(_this_dir))
    from dynamic_grpo_trainer import DynamicGRPOTrainer

from dataset_utils import create_reward_lookup, prepare_dapo_dataset
from reward_function import compute_dapo_reward


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DAPO Training with TRL")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g., 'auto' to shard across available GPUs)",
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
        help="Validation split name (if None, will create a deterministic split from train)",
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="BytedTsinghua-SIA/AIME-2024",
        help="Dataset name for evaluation (AIME-2024 to mirror original verl runs).",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="Eval split name (default uses 'train' for AIME-2024 parquet).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for testing)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--eval_subset_size",
        type=int,
        default=None,
        help="Limit the number of eval samples used in each eval run (None = use full eval set).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of the training split to hold out for validation when val_split is None",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="Random seed for creating the validation split when val_split is None",
    )

    # Training arguments (DAPO specific)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dapo_output",
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
        help="Maximum number of training steps (-1 for full epoch)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=None,
        help="Total completions generated per device before filtering (defaults to ~3x prompt oversampling).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--steps_per_generation",
        type=int,
        default=None,
        help=(
            "How many loss micro-steps reuse the same generated batch. "
            "Set to 1 to generate fresh completions every micro-step."
        ),
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

    # DAPO specific hyperparameters
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,
        help="Number of generations per prompt (n_resp_per_prompt)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Clip ratio low (clip_ratio_low)",
    )
    parser.add_argument(
        "--epsilon_high",
        type=float,
        default=0.28,
        help="Clip ratio high (clip_ratio_high) - DAPO's clip-higher",
    )
    parser.add_argument(
        "--clip_ratio_c",
        type=float,
        default=10.0,
        help="Dual-clip PPO lower bound for negative advantages (clip_ratio_c)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dapo",
        choices=["dapo", "grpo", "dr_grpo"],
        help="Loss type for GRPO",
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
        default=20480,
        help="Maximum completion length",
    )

    # Evaluation-time overrides to reduce memory
    parser.add_argument(
        "--eval_num_generations",
        type=int,
        default=2,
        help="Number of generations per prompt during evaluation (overrides num_generations).",
    )
    parser.add_argument(
        "--eval_max_completion_length",
        type=int,
        default=8192,
        help="Maximum completion length during evaluation (overrides max_completion_length).",
    )
    parser.add_argument(
        "--eval_temperature",
        type=float,
        default=None,
        help="Sampling temperature used only during evaluation (defaults to training temperature).",
    )
    parser.add_argument(
        "--eval_top_p",
        type=float,
        default=None,
        help="Top-p used only during evaluation (defaults to training top_p).",
    )

    # Overlong penalty arguments
    parser.add_argument(
        "--apply_overlong_penalty",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Apply overlong reward shaping (true/false)",
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
        default="DAPO-TRL",
        help="Static Weights & Biases project name when report_to includes 'wandb'",
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
        help="Use vLLM for generation to speed up sampling",
    )
    # vLLM configuration
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=["server", "colocate"],
        help="vLLM integration mode: 'server' connects to an external vLLM server; 'colocate' runs vLLM inline",
    )
    parser.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=None,
        help="Base URL for vLLM server, e.g., http://localhost:8010 (overrides host/port)",
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="127.0.0.1",
        help="Host of the vLLM server when vllm_mode=server",
    )
    parser.add_argument(
        "--vllm_server_port",
        type=int,
        default=8010,
        help="Port of the vLLM server when vllm_mode=server (default 8010 to avoid 8000 collisions)",
    )
    parser.add_argument(
        "--vllm_server_timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for vLLM server to be up before failing",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for vLLM when vllm_mode=colocate",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM when vllm_mode=colocate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient vs. reference model (beta=0 disables ref model; matches original DAPO).",
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.0,
        help="Entropy regularization coefficient (0 to match original DAPO).",
    )

    # Dynamic sampling / group filtering
    parser.add_argument(
        "--filter_groups_enable",
        action="store_true",
        default=False,
        help="Enable dynamic group filtering (drop prompts whose rewards have zero std across generations)",
    )
    parser.add_argument(
        "--filter_groups_metric",
        type=str,
        default="reward",
        help="Metric used for filtering (placeholder, reward std used by default)",
    )
    parser.add_argument(
        "--max_filter_gen_batches",
        type=int,
        default=0,
        help="Max extra generation batches for filtering (placeholder, not used in lightweight drop-only path)",
    )

    return parser.parse_args()


def create_reward_function_with_lookup(
    reward_lookup, args, tokenizer: Optional[AutoTokenizer] = None
):
    """
    Create a reward function that uses the reward lookup to get ground truths.

    Args:
        reward_lookup: Dictionary mapping prompts to ground truths
        args: Command line arguments

    Returns:
        Reward function callable
    """

    def reward_fn(completions, prompts=None, **kwargs):
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of completion texts in chat format
            prompts: List of prompt texts (used to lookup ground truths)

        Returns:
            List of reward scores
        """
        # Prefer ground truths passed via dataset extra fields
        gt_list = kwargs.get("ground_truth")

        # Build per-sample ground truths
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

    # Set random seed with rank offset for proper randomness per process in DDP
    effective_seed = args.seed + state.process_index
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    # Create output directory on main process only to avoid races
    if state.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    state.wait_for_everyone()

    if state.is_local_main_process:
        print("=" * 80)
        print("DAPO Training with TRL")
        print("=" * 80)
        print(f"Model: {args.model_name_or_path}")
        print(f"Dataset: {args.dataset_name}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)

    # If logging to wandb, set a static project name (main process only to avoid conflicts)
    try:
        if args.report_to and "wandb" in str(args.report_to).lower():
            if state.is_local_main_process:
                os.environ["WANDB_PROJECT"] = args.wandb_project
                print(f"Using W&B project: {os.environ['WANDB_PROJECT']}")
    except Exception:
        pass

    # If using vLLM, set recommended environment vars and validate import
    if args.use_vllm:
        if state.is_local_main_process:
            print("\n[vLLM] use_vllm=True. Preparing vLLM runtime...")
        # vLLM recommends spawn on many systems to avoid deadlocks
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Prefer v1 engine API if available
        os.environ.setdefault("VLLM_USE_V1", "1")
        try:
            import vllm  # noqa: F401

            ver = getattr(vllm, "__version__", "unknown")
            if state.is_local_main_process:
                print(f"[vLLM] Detected vllm version: {ver}")
        except Exception as e:
            if state.is_local_main_process:
                print(
                    f"[vLLM] Import error: {e}. vLLM may not be installed or incompatible."
                )
                print(
                    "[vLLM] Consider: pip install 'trl[vllm]' and a CUDA-compatible vllm."
                )
        # Warn about potential GPU contention when colocating
        if (
            args.vllm_mode == "colocate"
            and str(args.device_map).lower() != "cpu"
        ):
            if state.is_local_main_process:
                print(
                    "[vLLM] Colocated mode with training model on GPUs."
                    " Consider lowering --vllm_gpu_memory_utilization or using server mode if OOM/hangs occur."
                )

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
        if args.eval_subset_size is not None and val_dataset is not None:
            try:
                val_dataset = val_dataset.select(
                    range(min(int(args.eval_subset_size), len(val_dataset)))
                )
                if state.is_local_main_process:
                    print(
                        f"[eval] Using subset of {len(val_dataset)} samples for evaluation."
                    )
            except Exception as e:
                if state.is_local_main_process:
                    print(f"[eval] Could not subset eval dataset: {e}")

    # Load model and tokenizer
    if state.is_local_main_process:
        print("\n[2/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        trust_remote_code=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL's GRPO expects left padding
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass

    # Allow disabling HF sharding by passing --device_map none
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
    # Ensure model uses the same pad token id
    if (
        hasattr(model.config, "pad_token_id")
        and model.config.pad_token_id is None
        and tokenizer.pad_token_id is not None
    ):
        model.config.pad_token_id = tokenizer.pad_token_id

    if state.is_local_main_process:
        print(f"Model loaded: {model.config.model_type}")
        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
        )
    # If using device_map, show where shards are placed
    try:
        has_dm = hasattr(model, "hf_device_map")
        if state.is_local_main_process:
            print(f"Model has hf_device_map: {has_dm}")
            if has_dm:
                print(f"Device map: {model.hf_device_map}")
    except Exception:
        pass

    # Pre-flight check for vLLM colocate tensor parallel vs. world size
    try:
        if args.use_vllm and args.vllm_mode == "colocate":
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            tp = int(args.vllm_tensor_parallel_size)
            if world_size % tp != 0:
                print(
                    f"[vLLM] Warning: vllm_tensor_parallel_size ({tp}) must divide world size ({world_size}) evenly in colocate mode. "
                    "Either launch DDP with a matching WORLD_SIZE (e.g., accelerate/torchrun with 4 procs) or switch to server mode."
                )
    except Exception:
        pass

    # Create reward function
    if state.is_local_main_process:
        print("\n[3/4] Creating reward function...")
    reward_fn = create_reward_function_with_lookup(
        reward_lookup, args, tokenizer
    )

    # Calculate rank-specific vLLM port for colocate mode to avoid port collisions
    vllm_port = int(os.environ.get("TRL_VLLM_PORT", str(args.vllm_server_port)))
    if args.use_vllm and args.vllm_mode == "colocate":
        # Each DDP rank needs its own port
        vllm_port = vllm_port + state.process_index
        if state.is_local_main_process:
            print(
                f"[vLLM] Using rank-specific ports for colocate mode: base_port={args.vllm_server_port}, rank_ports={vllm_port - state.process_index} + rank_id"
            )

    # Configure GRPO trainer with DAPO settings
    if state.is_local_main_process:
        print("\n[4/4] Configuring GRPO trainer with DAPO settings...")
    # Oversample prompts for dynamic filtering: default 3x prompt batch -> completions
    default_generation_batch_size = (
        args.per_device_train_batch_size * args.num_generations * 3
    )
    generation_batch_size = (
        args.generation_batch_size
        if args.generation_batch_size is not None
        else default_generation_batch_size
    )
    # If the user forces steps_per_generation, let TRL derive generation_batch_size to avoid conflicts.
    if args.steps_per_generation is not None:
        generation_batch_size = None
        if state.is_local_main_process:
            print(
                f"[config] steps_per_generation={args.steps_per_generation} provided; "
                "letting GRPOConfig derive generation_batch_size."
            )
    training_args = GRPOConfig(
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
        generation_batch_size=generation_batch_size,
        steps_per_generation=args.steps_per_generation,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        # DAPO specific settings
        loss_type=args.loss_type,  # "dapo" for token-level normalization
        epsilon=args.epsilon,  # clip_ratio_low
        epsilon_high=args.epsilon_high,  # clip_ratio_high (DAPO's clip-higher)
        num_generations=args.num_generations,  # n_resp_per_prompt
        sync_ref_model=True,  # important for GRPO based model; untuned steps/weight!
        # ref_model_sync_steps=200,
        # Generation settings
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        # Other settings
        bf16=args.bf16,
        seed=args.seed,
        beta=args.beta,
        gradient_checkpointing=True,
        use_vllm=args.use_vllm,
        # vLLM settings
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=(
            args.vllm_server_base_url or os.environ.get("TRL_VLLM_BASE_URL")
        ),
        vllm_server_host=os.environ.get("TRL_VLLM_HOST", args.vllm_server_host),
        vllm_server_port=vllm_port,  # Use rank-specific port calculated above
        vllm_server_timeout=float(
            os.environ.get("TRL_VLLM_TIMEOUT", str(args.vllm_server_timeout))
        ),
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
    )

    # Stash evaluation overrides on the args object for later use
    # (GRPOConfig won't know about these; we'll apply them in a small wrapper.)
    try:
        setattr(
            training_args,
            "_eval_num_generations",
            int(args.eval_num_generations),
        )
        setattr(
            training_args,
            "_eval_max_completion_length",
            int(args.eval_max_completion_length),
        )
        setattr(
            training_args,
            "_eval_temperature",
            (
                args.eval_temperature
                if args.eval_temperature is not None
                else args.temperature
            ),
        )
        setattr(
            training_args,
            "_eval_top_p",
            args.eval_top_p if args.eval_top_p is not None else args.top_p,
        )
    except Exception:
        pass

    if state.is_local_main_process:
        print("\nTraining configuration:")
        print(f"  Loss type: {args.loss_type}")
        print(f"  Clip range: [{args.epsilon}, {args.epsilon_high}]")
        print(f"  Generations per prompt: {args.num_generations}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size per device: {args.per_device_train_batch_size}")
        print(
            f"  Gradient accumulation steps: {args.gradient_accumulation_steps}"
        )
        print(
            f"  Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}"
        )

    # Initialize trainer
    trainer = DynamicGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        filter_groups_enable=args.filter_groups_enable,
        filter_groups_metric=args.filter_groups_metric,
        max_filter_gen_batches=args.max_filter_gen_batches,
        entropy_coeff=args.entropy_coeff,
        clip_ratio_c=args.clip_ratio_c,
    )

    # Reduce memory pressure during evaluation by temporarily overriding
    # num_generations and max_completion_length inside prediction_step.
    # This avoids constructing massive [B*T*V] logits tensors for long generations.
    try:
        import types

        if hasattr(trainer, "prediction_step"):
            trainer._orig_prediction_step = trainer.prediction_step  # type: ignore[attr-defined]

            def _prediction_step_eval_overrides(
                self, model, inputs, prediction_loss_only, ignore_keys=None
            ):
                # Evaluation loop sets model.eval(); use that to detect eval mode
                if getattr(model, "training", True) is False:
                    old_num_gen = getattr(self.args, "num_generations", None)
                    old_max_comp = getattr(
                        self.args, "max_completion_length", None
                    )
                    old_temp = getattr(self.args, "temperature", None)
                    old_top_p = getattr(self.args, "top_p", None)
                    try:
                        eval_num_gen = getattr(
                            self.args, "_eval_num_generations", old_num_gen
                        )
                        eval_max_comp = getattr(
                            self.args,
                            "_eval_max_completion_length",
                            old_max_comp,
                        )
                        eval_temp = getattr(
                            self.args, "_eval_temperature", old_temp
                        )
                        eval_top_p = getattr(
                            self.args, "_eval_top_p", old_top_p
                        )
                        if eval_num_gen is not None:
                            self.args.num_generations = int(eval_num_gen)
                        if eval_max_comp is not None:
                            self.args.max_completion_length = int(eval_max_comp)
                        if eval_temp is not None:
                            self.args.temperature = float(eval_temp)
                        if eval_top_p is not None:
                            self.args.top_p = float(eval_top_p)
                        return self._orig_prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)  # type: ignore[attr-defined]
                    finally:
                        if old_num_gen is not None:
                            self.args.num_generations = old_num_gen
                        if old_max_comp is not None:
                            self.args.max_completion_length = old_max_comp
                        if old_temp is not None:
                            self.args.temperature = old_temp
                        if old_top_p is not None:
                            self.args.top_p = old_top_p
                # Training path: call through unmodified
                return self._orig_prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)  # type: ignore[attr-defined]

            trainer.prediction_step = types.MethodType(
                _prediction_step_eval_overrides, trainer
            )

            if state.is_local_main_process:
                print(
                    f"\n[Eval overrides] Using eval_num_generations={getattr(training_args, '_eval_num_generations', 'NA')}, "
                    f"eval_max_completion_length={getattr(training_args, '_eval_max_completion_length', 'NA')}, "
                    f"eval_temperature={getattr(training_args, '_eval_temperature', 'NA')}, "
                    f"eval_top_p={getattr(training_args, '_eval_top_p', 'NA')}"
                )
    except Exception as e:
        if state.is_local_main_process:
            print(
                f"[Eval overrides] Could not apply evaluation-time overrides: {e}"
            )

    if state.is_local_main_process:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

    # Train
    trainer.train()

    # Wait for all processes to finish training before saving
    state.wait_for_everyone()

    # Save final model
    if state.is_local_main_process:
        print("\nSaving final model...")
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        print("\nTraining completed!")
        print(f"Model saved to: {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()
