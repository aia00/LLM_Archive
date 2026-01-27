"""Train multi-objective RLHF baselines using SftAsRLHF GRPO/DAPO trainer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl import GRPOConfig
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("trl is required to run this script") from exc

# Add local roots to sys.path for script execution
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

# Add SftAsRLHF/dapo_trl to sys.path for DynamicGRPOTrainer
_SFT_DIR = _ROOT_DIR / "SftAsRLHF" / "dapo_trl"
if str(_SFT_DIR) not in sys.path:
    sys.path.append(str(_SFT_DIR))

from dynamic_grpo_trainer import DynamicGRPOTrainer  # noqa: E402

from mycode.aggregated_grpo_trainer import AggregatedGRPOTrainer  # noqa: E402
from mycode.aggregation_factory import build_reward_aggregator  # noqa: E402
from mycode.callbacks import ParetoMetaRewardCallback  # noqa: E402
from mycode.dataset_utils import MathDatasetConfig, prepare_math_dataset  # noqa: E402
from mycode.gradient_aggregation_trainer import GradientAggregationGRPOTrainer  # noqa: E402
from mycode.multi_objective_reward import (  # noqa: E402
    ParetoMetaReward,
    resolve_weights_with_names,
)
from mycode.multi_objective_trainer import GradientWeightedGRPOTrainer  # noqa: E402
from mycode.pareto import ParetoBuffer  # noqa: E402
from mycode.reward_components import build_reward_functions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-objective RLHF baselines")

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Dataset name",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--question_key", type=str, default=None)
    parser.add_argument("--answer_key", type=str, default=None)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful math assistant.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="Problem:\n{question}\n\nAnswer:",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        default="static",
        choices=[
            "static",
            "hypervolume",
            "dynamic",
            "constrained",
            "minimax",
            "chebyshev",
            "nash",
            "cvar",
            "contextual",
            "uncertainty",
            "pcgrad",
            "mgda",
            "cagrad",
            "nash_mtl",
        ],
        help=(
            "Multi-objective method: static/hypervolume/dynamic, "
            "constraints, minimax, nonlinear scalarization, contextual/uncertainty, "
            "or gradient aggregation (pcgrad/mgda/cagrad/nash_mtl)."
        ),
    )
    parser.add_argument(
        "--weight_preset",
        type=str,
        default="balanced",
        choices=["accuracy", "balanced", "efficiency"],
        help="Weight preset for static scalarization",
    )
    parser.add_argument("--weight_accuracy", type=float, default=None)
    parser.add_argument("--weight_conciseness", type=float, default=None)
    parser.add_argument("--weight_clarity", type=float, default=None)
    parser.add_argument("--pareto_max_size", type=int, default=128)
    parser.add_argument("--dynamic_eta", type=float, default=0.5)
    parser.add_argument("--dynamic_mu", type=float, default=1.0)

    # Constrained RLHF
    parser.add_argument("--primary_reward", type=str, default="accuracy")
    parser.add_argument("--constraint_rewards", type=str, default=None)
    parser.add_argument("--constraint_thresholds", type=str, default=None)
    parser.add_argument("--constraint_directions", type=str, default=None)
    parser.add_argument("--constraint_lambda_lr", type=float, default=0.05)
    parser.add_argument("--constraint_lambda_max", type=float, default=10.0)
    parser.add_argument("--constraint_disable_ema", action="store_true")
    parser.add_argument("--constraint_ema_alpha", type=float, default=0.05)

    # Minimax / adversarial weighting
    parser.add_argument("--minimax_eta", type=float, default=0.5)
    parser.add_argument("--minimax_min_weight", type=float, default=0.0)

    # Nonlinear scalarizations
    parser.add_argument("--chebyshev_reference", type=str, default=None)
    parser.add_argument("--nash_baseline", type=str, default=None)
    parser.add_argument("--nash_epsilon", type=float, default=1e-3)
    parser.add_argument("--cvar_alpha", type=float, default=0.2)

    # Contextual weighting
    parser.add_argument("--context_short_length", type=int, default=120)
    parser.add_argument("--context_long_length", type=int, default=300)
    parser.add_argument("--context_short_weights", type=str, default=None)
    parser.add_argument("--context_long_weights", type=str, default=None)
    parser.add_argument("--context_clarity_weights", type=str, default=None)
    parser.add_argument("--context_conciseness_weights", type=str, default=None)

    # Uncertainty-aware weighting
    parser.add_argument("--uncertainty_ema_alpha", type=float, default=0.05)
    parser.add_argument("--uncertainty_eps", type=float, default=1e-6)

    # Gradient aggregation
    parser.add_argument("--pcgrad_shuffle", action="store_true")
    parser.add_argument("--cagrad_alpha", type=float, default=0.5)
    parser.add_argument("--mgda_iters", type=int, default=50)

    # Training
    parser.add_argument("--output_dir", type=str, default="./multi_objective_output")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = PartialState()

    effective_seed = args.seed + state.process_index
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    if state.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    config = MathDatasetConfig(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        question_key=args.question_key,
        answer_key=args.answer_key,
        system_prompt=args.system_prompt,
        prompt_template=args.prompt_template,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    train_dataset, eval_dataset = prepare_math_dataset(config)

    if args.method == "hypervolume" and eval_dataset is None:
        raise ValueError("Method B requires --eval_split for validation.")

    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=args.device_map,
        torch_dtype=(torch.bfloat16 if args.bf16 else None),
    )

    reward_funcs, _ = build_reward_functions(tokenizer=tokenizer)
    reward_names = [getattr(fn, "__name__", f"reward_{idx}") for idx, fn in enumerate(reward_funcs)]

    weight_override = None
    if (
        args.weight_accuracy is not None
        or args.weight_conciseness is not None
        or args.weight_clarity is not None
    ):
        weight_override = (
            args.weight_accuracy or 0.0,
            args.weight_conciseness or 0.0,
            args.weight_clarity or 0.0,
        )
    base_weights = resolve_weights_with_names(
        args.weight_preset, weight_override, reward_names
    )

    generation_batch_size = args.generation_batch_size
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        generation_batch_size=generation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_generations=args.num_generations,
        sync_ref_model=True,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=args.bf16,
        seed=args.seed,
        gradient_checkpointing=True,
    )

    if args.method == "dynamic":
        trainer = GradientWeightedGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            dynamic_eta=args.dynamic_eta,
            dynamic_mu=args.dynamic_mu,
        )
    elif args.method in {"pcgrad", "mgda", "cagrad", "nash_mtl"}:
        trainer = GradientAggregationGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            aggregation_method=("nash" if args.method == "nash_mtl" else args.method),
            pcgrad_shuffle=args.pcgrad_shuffle,
            cagrad_alpha=args.cagrad_alpha,
            mgda_iters=args.mgda_iters,
        )
    elif args.method in {
        "constrained",
        "minimax",
        "chebyshev",
        "nash",
        "cvar",
        "contextual",
        "uncertainty",
    }:
        aggregator = build_reward_aggregator(
            args.method, reward_names, base_weights, args
        )
        trainer = AggregatedGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            reward_aggregator=aggregator,
        )
    else:
        trainer = DynamicGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

    trainer.reward_weights = torch.tensor(
        base_weights, device=trainer.accelerator.device
    )

    if args.method == "hypervolume":
        pareto = ParetoBuffer(max_size=args.pareto_max_size)
        meta_reward = ParetoMetaReward(buffer=pareto)
        callback = ParetoMetaRewardCallback(
            meta_reward=meta_reward,
            base_weights=base_weights,
            trainer=trainer,
        )
        trainer.add_callback(callback)

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
