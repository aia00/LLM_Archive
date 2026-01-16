# online_sft/logging_utils.py
# DAPO-style detailed logging utilities for Online SFT training
#
# Provides comprehensive training metrics display similar to DAPO:
# - Reward statistics (accuracy, correct/incorrect counts)
# - Response length statistics
# - KL divergence metrics
# - Weight statistics
# - Group filtering statistics
# - Timing information

import time
from typing import Dict, Optional


def format_metrics_log(
    metrics: Dict[str, float],
    global_step: int,
    episode: int,
    total_episodes: int,
    step_time: Optional[float] = None,
    compact: bool = False,
) -> str:
    """
    Format training metrics into a detailed DAPO-style log string.

    Args:
        metrics: Dictionary of metrics from trainer.step()
        global_step: Current training step
        episode: Current episode count
        total_episodes: Total episodes for training
        step_time: Optional time taken for this step (seconds)
        compact: If True, use single-line format; if False, use multi-line

    Returns:
        Formatted log string
    """
    if compact:
        return _format_compact(metrics, global_step, episode, total_episodes, step_time)
    else:
        return _format_detailed(metrics, global_step, episode, total_episodes, step_time)


def _format_compact(
    metrics: Dict[str, float],
    global_step: int,
    episode: int,
    total_episodes: int,
    step_time: Optional[float] = None,
) -> str:
    """Single-line compact format."""
    parts = [f"Step {global_step}"]
    parts.append(f"Ep {episode}/{total_episodes}")

    # Core metrics
    parts.append(f"loss={metrics.get('loss', 0):.4f}")

    # Reward
    acc = metrics.get('reward/accuracy', 0)
    reward_mean = metrics.get('reward/mean', 0)
    parts.append(f"acc={acc:.1%}")
    parts.append(f"r={reward_mean:+.3f}")

    # KL
    kl = metrics.get('policy/kl_y_ref_per_tok/mean', 0)
    parts.append(f"kl={kl:.4f}")

    # Response length
    resp_len = metrics.get('response/avg_len_tokens', 0)
    parts.append(f"len={resp_len:.0f}")

    # Weight
    w_mean = metrics.get('weight/mean', 0)
    parts.append(f"w={w_mean:+.3f}")

    # Filter stats (if available)
    if 'filter/num_groups_kept' in metrics:
        kept = int(metrics.get('filter/num_groups_kept', 0))
        total = int(metrics.get('filter/num_groups_total', 0))
        parts.append(f"grp={kept}/{total}")

    # Timing
    if step_time is not None:
        parts.append(f"t={step_time:.1f}s")

    return " | ".join(parts)


def _format_detailed(
    metrics: Dict[str, float],
    global_step: int,
    episode: int,
    total_episodes: int,
    step_time: Optional[float] = None,
) -> str:
    """Multi-line detailed format similar to DAPO."""
    lines = []

    # Header
    progress_pct = 100.0 * episode / total_episodes if total_episodes > 0 else 0
    header = f"{'='*60}"
    lines.append(header)
    lines.append(f"Step {global_step} | Episode {episode}/{total_episodes} ({progress_pct:.1f}%)")
    lines.append(header)

    # Loss
    loss = metrics.get('loss', 0)
    lines.append(f"  Loss: {loss:.6f}")

    # Reward section
    lines.append("")
    lines.append("  [Reward]")
    acc = metrics.get('reward/accuracy', 0)
    num_correct = int(metrics.get('reward/num_correct', 0))
    num_incorrect = int(metrics.get('reward/num_incorrect', 0))
    num_samples = int(metrics.get('response/num_samples', 0))
    reward_mean = metrics.get('reward/mean', 0)
    reward_std = metrics.get('reward/std', 0)
    lines.append(f"    Accuracy: {acc:.1%} ({num_correct}/{num_samples} correct, {num_incorrect} incorrect)")
    lines.append(f"    Mean: {reward_mean:+.4f}  Std: {reward_std:.4f}")

    # Response section
    lines.append("")
    lines.append("  [Response]")
    avg_len = metrics.get('response/avg_len_tokens', 0)
    min_len = metrics.get('response/min_len_tokens', 0)
    max_len = metrics.get('response/max_len_tokens', 0)
    lines.append(f"    Length: mean={avg_len:.1f}  min={min_len:.0f}  max={max_len:.0f} tokens")
    lines.append(f"    Samples: {num_samples}")

    # Policy section
    lines.append("")
    lines.append("  [Policy]")
    logp_mean = metrics.get('policy/logp_seq_raw/mean', 0)
    logp_std = metrics.get('policy/logp_seq_raw/std', 0)
    logp_min = metrics.get('policy/logp_seq_raw/min', 0)
    logp_max = metrics.get('policy/logp_seq_raw/max', 0)
    logp_per_tok = metrics.get('policy/logp_seq_per_tok/mean', 0)
    lines.append(f"    LogP(seq): mean={logp_mean:.2f}  std={logp_std:.2f}  min={logp_min:.2f}  max={logp_max:.2f}")
    lines.append(f"    LogP/tok:  {logp_per_tok:.4f}")

    # KL section
    kl_total = metrics.get('policy/kl_y_ref/mean', 0)
    kl_per_tok = metrics.get('policy/kl_y_ref_per_tok/mean', 0)
    lines.append(f"    KL(total): {kl_total:.4f}")
    lines.append(f"    KL/tok:    {kl_per_tok:.4f}")

    # Weight section
    lines.append("")
    lines.append("  [Weight]")
    w_mean = metrics.get('weight/mean', 0)
    w_std = metrics.get('weight/std', 0)
    w_min = metrics.get('weight/min', 0)
    w_max = metrics.get('weight/max', 0)
    lines.append(f"    Mean: {w_mean:+.4f}  Std: {w_std:.4f}")
    lines.append(f"    Min:  {w_min:+.4f}  Max: {w_max:+.4f}")

    # Clipping section (if any clipping occurred)
    if metrics.get('clip/num_logprob_clamped', 0) > 0:
        lines.append("")
        lines.append("  [Clipping]")
        num_clamped = int(metrics.get('clip/num_logprob_clamped', 0))
        pct_clamped = metrics.get('clip/pct_logprob_clamped', 0)
        lines.append(f"    LogP clamped: {num_clamped} ({pct_clamped:.1f}%)")

    # Filter section (if filtering is enabled)
    if 'filter/num_groups_kept' in metrics:
        lines.append("")
        lines.append("  [Group Filtering]")
        kept = int(metrics.get('filter/num_groups_kept', 0))
        total = int(metrics.get('filter/num_groups_total', 0))
        all_correct = int(metrics.get('filter/num_groups_all_correct', 0))
        all_wrong = int(metrics.get('filter/num_groups_all_wrong', 0))
        mixed = int(metrics.get('filter/num_groups_mixed', 0))
        pct_kept = metrics.get('filter/pct_groups_kept', 0)
        gen_batches = int(metrics.get('filter/total_gen_batches', 1))
        backfilled = int(metrics.get('filter/total_backfilled', 0))

        lines.append(f"    Kept: {kept}/{total} ({pct_kept:.1f}%)")
        lines.append(f"    All Correct: {all_correct}  All Wrong: {all_wrong}  Mixed: {mixed}")
        if gen_batches > 1 or backfilled > 0:
            lines.append(f"    Gen Batches: {gen_batches}  Backfilled: {backfilled}")

    # Learning rate (if available)
    if 'lr/current' in metrics or 'lr' in metrics:
        lines.append("")
        lines.append("  [Optimizer]")
        lr = metrics.get('lr/current', metrics.get('lr', 0))
        lines.append(f"    Learning Rate: {lr:.2e}")

    # Timing (if available)
    if step_time is not None:
        lines.append("")
        lines.append(f"  [Timing] Step: {step_time:.2f}s")

    # Warnings
    warnings = []
    if metrics.get('warn/all_identical_responses', 0) > 0:
        warnings.append("All responses identical!")
    if metrics.get('warn/no_valid_groups', 0) > 0:
        warnings.append("No valid groups found!")
    if metrics.get('skipped/empty_batch', 0) > 0:
        warnings.append("Batch skipped (empty)!")

    if warnings:
        lines.append("")
        lines.append("  [WARNINGS]")
        for w in warnings:
            lines.append(f"    ! {w}")

    lines.append("")

    return "\n".join(lines)


class StepTimer:
    """Simple timer for measuring step duration."""

    def __init__(self):
        self.start_time = None
        self.last_step_time = None

    def start(self):
        """Start timing a step."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timing and return duration in seconds."""
        if self.start_time is None:
            return 0.0
        self.last_step_time = time.time() - self.start_time
        self.start_time = None
        return self.last_step_time

    def get_last_time(self) -> Optional[float]:
        """Get the last measured step time."""
        return self.last_step_time


def print_training_header(
    model_name: str,
    dataset_name: str,
    num_gpus: int,
    batch_size: int,
    grad_accum: int,
    num_return_sequences: int,
    total_episodes: int,
    learning_rate: float,
    beta: float,
    filter_enabled: bool = False,
):
    """Print a formatted training header."""
    effective_batch = batch_size * num_gpus * grad_accum

    print("\n" + "=" * 70)
    print("  ONLINE SFT TRAINING")
    print("=" * 70)
    print(f"  Model:           {model_name}")
    print(f"  Dataset:         {dataset_name}")
    print("-" * 70)
    print(f"  GPUs:            {num_gpus}")
    print(f"  Batch/GPU:       {batch_size}")
    print(f"  Grad Accum:      {grad_accum}")
    print(f"  Effective Batch: {effective_batch}")
    print(f"  Responses/Prompt: {num_return_sequences}")
    print("-" * 70)
    print(f"  Total Episodes:  {total_episodes}")
    print(f"  Learning Rate:   {learning_rate:.2e}")
    print(f"  KL Beta:         {beta}")
    print(f"  Group Filtering: {'Enabled' if filter_enabled else 'Disabled'}")
    print("=" * 70 + "\n")
