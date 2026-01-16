# online_sft/scheduler_utils.py
# Learning rate scheduler utilities for Online SFT training

from typing import Any, Optional
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_learning_rate: float = 1e-7,
    scheduler_type: str = "cosine_with_warmup",
) -> Any:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        total_steps: Total number of optimization steps
        warmup_ratio: Fraction of total steps for linear warmup
        min_learning_rate: Minimum learning rate (eta_min for cosine)
        scheduler_type: Type of scheduler ("cosine_with_warmup", "linear", "constant")

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    warmup_steps = int(max(0.0, min(1.0, warmup_ratio)) * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)

    if scheduler_type == "linear":
        # Linear decay from base LR to min_learning_rate
        if warmup_steps > 0:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=0.05,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            # Linear decay
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]
            decay_factors = [
                (min_learning_rate / blr) if blr > 0 else 0.0
                for blr in base_lrs
            ]

            def make_linear_decay_fn(start_factor, end_factor, total_iters):
                def fn(step):
                    return start_factor + (end_factor - start_factor) * min(1.0, step / total_iters)
                return fn

            decay_fns = [
                make_linear_decay_fn(1.0, f, cosine_steps)
                for f in decay_factors
            ]
            decay_sched = LambdaLR(optimizer, lr_lambda=decay_fns)

            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, decay_sched],
                milestones=[warmup_steps],
            )
        else:
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]
            decay_factors = [
                (min_learning_rate / blr) if blr > 0 else 0.0
                for blr in base_lrs
            ]

            def make_linear_decay_fn(start_factor, end_factor, total_iters):
                def fn(step):
                    return start_factor + (end_factor - start_factor) * min(1.0, step / total_iters)
                return fn

            decay_fns = [
                make_linear_decay_fn(1.0, f, total_steps)
                for f in decay_factors
            ]
            return LambdaLR(optimizer, lr_lambda=decay_fns)

    # Default: cosine_with_warmup
    if warmup_steps > 0:
        # Linear warmup: start at 5% of base LR, end at 100%
        warmup_sched = LinearLR(
            optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine decay
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_learning_rate,
        )

        # Final clamp stage: keep LR fixed at eta_min once cosine finishes
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        clamp_factors = [
            (min_learning_rate / blr) if blr > 0 else 0.0
            for blr in base_lrs
        ]
        clamp_lambdas = [(lambda f: (lambda _: f))(f) for f in clamp_factors]
        clamp_sched = LambdaLR(optimizer, lr_lambda=clamp_lambdas)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched, clamp_sched],
            milestones=[warmup_steps, warmup_steps + cosine_steps],
        )
    else:
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_learning_rate,
        )
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        clamp_factors = [
            (min_learning_rate / blr) if blr > 0 else 0.0
            for blr in base_lrs
        ]
        clamp_lambdas = [(lambda f: (lambda _: f))(f) for f in clamp_factors]
        clamp_sched = LambdaLR(optimizer, lr_lambda=clamp_lambdas)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[cosine_sched, clamp_sched],
            milestones=[cosine_steps],
        )

    return scheduler


def estimate_total_steps(
    total_episodes: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Estimate total optimizer steps.

    Args:
        total_episodes: Total number of training episodes
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps

    Returns:
        Estimated total optimizer steps
    """
    est_microbatches = max(1, (total_episodes + batch_size - 1) // batch_size)
    est_total_steps = max(1, est_microbatches // gradient_accumulation_steps)
    return est_total_steps


def get_scheduler_info(
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    cosine_steps: int,
    total_steps: int,
) -> dict:
    """
    Get current scheduler state information.

    Args:
        scheduler: The scheduler
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        cosine_steps: Number of cosine decay steps
        total_steps: Total steps

    Returns:
        Dictionary with scheduler info
    """
    info = {}
    try:
        current_lr = optimizer.param_groups[0]["lr"]
        info["lr/current"] = float(current_lr)

        if hasattr(scheduler, "last_epoch"):
            opt_step = max(0, int(scheduler.last_epoch))
            info["lr/opt_step"] = float(opt_step)
            info["lr/total_opt_steps"] = float(total_steps)

            clamp_start = warmup_steps + cosine_steps
            in_warmup = int(opt_step < warmup_steps)
            in_cosine = int((opt_step >= warmup_steps) and (opt_step < clamp_start))
            in_clamp = int(opt_step >= clamp_start)

            info["lr/in_warmup"] = float(in_warmup)
            info["lr/in_cosine"] = float(in_cosine)
            info["lr/in_clamp"] = float(in_clamp)

            if in_warmup:
                info["lr/phase"] = "warmup"
                info["lr/phase_step"] = float(opt_step)
            elif in_cosine:
                info["lr/phase"] = "cosine"
                info["lr/phase_step"] = float(max(0, opt_step - warmup_steps))
            else:
                info["lr/phase"] = "clamp"
                info["lr/phase_step"] = float(max(0, opt_step - clamp_start))
    except Exception:
        pass

    return info


def print_scheduler_summary(
    total_steps: int,
    warmup_steps: int,
    cosine_steps: int,
    base_lr: float,
    min_lr: float,
    gradient_accumulation_steps: int,
):
    """Print a clear, user-friendly scheduler summary."""
    print("LR Scheduler: Linear Warmup + Cosine Decay")
    print(
        f"  Optimizer steps (estimated): {total_steps}"
        f" | Accumulation: {gradient_accumulation_steps}"
    )
    print(
        f"  Warmup: {warmup_steps} steps ({100*warmup_steps/total_steps:.1f}% of optimizer steps)"
    )
    print(
        f"  Cosine decay: {cosine_steps} steps | base_lr={base_lr} -> min_lr={min_lr}"
    )
    print("  Clamp: hold min_lr after cosine (prevent end bump)")
