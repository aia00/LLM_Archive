# online_sft_trl/online_sft_trainer.py
"""
Online SFT Trainer implementing the "Online SFT = RLHF" algorithm using TRL's GRPO framework.

The Online SFT approach provides an unbiased gradient estimator that achieves the same
effect as RLHF but without PPO's complexity.

Core loss (from Proposition 3 in the paper):
    L_hat = - [ w_theta(x,y) * log_pi_theta(y|x) + 0.5 * stop_grad(w_theta(x,y)) * (log_pi_theta(y|x))^2 ]

where:
    w_theta(x,y) = [ r(x,y) - beta * (log pi_theta(y|x) - log pi_ref(y|x)) ] / log pi_theta(y|x)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from dapo_trl.dynamic_grpo_trainer import DynamicGRPOTrainer
from trl.trainer.grpo_trainer import nanmax, nanmin, nanstd

from .online_sft_config import OnlineSFTConfig


class OnlineSFTTrainer(DynamicGRPOTrainer):
    """
    Online SFT Trainer extending DynamicGRPOTrainer with the novel Online SFT loss function.

    This trainer implements the "Online SFT = RLHF" algorithm which provides an unbiased
    gradient estimator achieving the same effect as RLHF without PPO.

    Inherits from DynamicGRPOTrainer to get:
    - Group filtering with retry logic (DAPO-style dynamic sampling)
    - Dual-clip PPO support
    - Entropy regularization

    The key differences from standard GRPO:
    1. Custom loss function using weight w_theta(x,y) instead of advantages
    2. Quadratic term in the loss for variance reduction
    3. Optional length normalization and log probability clamping

    Example:
        ```python
        from online_sft_trl import OnlineSFTTrainer, OnlineSFTConfig

        config = OnlineSFTConfig(
            beta=0.1,
            clip_logprob_min=-10.0,
            normalize_logprob_by_length=True,
        )
        trainer = OnlineSFTTrainer(
            model="Qwen/Qwen2.5-Math-7B",
            reward_funcs=my_reward_fn,
            args=config,
            train_dataset=dataset,
        )
        trainer.train()
        ```
    """

    _tag_names = ["trl", "online-sft"]
    _name = "OnlineSFT"

    def __init__(
        self,
        model,
        reward_funcs,
        args: OnlineSFTConfig = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        rollout_func: Optional[Callable] = None,
        *,
        # Online SFT specific parameters (can override config)
        filter_groups_enable: Optional[bool] = None,
        filter_groups_metric: str = "acc",  # Use accuracy-based filtering by default
        filter_groups_min_std: Optional[
            float
        ] = None,  # kept for API compatibility
        max_filter_gen_batches: int = 0,
    ):
        # Use OnlineSFTConfig defaults if no config provided
        if args is None:
            args = OnlineSFTConfig(output_dir="./online_sft_output")

        # Get filter parameters from config or use defaults
        _filter_groups_enable = (
            filter_groups_enable
            if filter_groups_enable is not None
            else getattr(args, "filter_groups_enable", True)
        )
        _max_filter_gen_batches = (
            max_filter_gen_batches
            if max_filter_gen_batches > 0
            else getattr(args, "max_filter_retries", 10)
        )

        # Store loss-specific parameters
        self.denom_eps = getattr(args, "denom_eps", 1e-8)
        self.clip_weight_value = getattr(args, "clip_weight_value", 1.0)
        self.clip_logprob_min = getattr(args, "clip_logprob_min", -10.0)
        self.normalize_logprob_by_length = getattr(
            args, "normalize_logprob_by_length", False
        )

        # Call parent __init__ with filter parameters
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
            filter_groups_enable=_filter_groups_enable,
            filter_groups_metric=filter_groups_metric,
            max_filter_gen_batches=_max_filter_gen_batches,
        )

    def _compute_online_sft_weight(
        self,
        rewards: torch.Tensor,
        per_token_logps: torch.Tensor,
        ref_per_token_logps: Optional[torch.Tensor],
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Online SFT weight w_theta(x,y).

        w = (r - beta * (log_pi - log_ref)) / (sign(log_pi) * clamp(|log_pi|, min=eps))

        Args:
            rewards: (B,) reward tensor
            per_token_logps: (B, T) per-token log probabilities
            ref_per_token_logps: (B, T) reference model per-token log probs, or None
            completion_mask: (B, T) mask for valid completion tokens

        Returns:
            w: (B,) weight tensor
        """
        # Compute sequence-level log probs
        seq_logp = (per_token_logps * completion_mask).sum(dim=-1)  # (B,)
        seq_len = completion_mask.sum(dim=-1).clamp(min=1.0).float()  # (B,)

        # Optionally normalize by length
        if self.normalize_logprob_by_length:
            seq_logp_for_loss = seq_logp / seq_len
        else:
            seq_logp_for_loss = seq_logp

        # Compute reference sequence log probs if beta > 0
        if self.beta != 0.0 and ref_per_token_logps is not None:
            ref_seq_logp = (ref_per_token_logps * completion_mask).sum(dim=-1)
            if self.normalize_logprob_by_length:
                ref_seq_logp = ref_seq_logp / seq_len
            kl_term = seq_logp_for_loss - ref_seq_logp
        else:
            kl_term = torch.zeros_like(seq_logp_for_loss)

        # Compute numerator: r - beta * KL
        numerator = rewards - self.beta * kl_term

        # Compute denominator: sign(log_pi) * clamp(|log_pi|, min=eps)
        # This handles the case where log_pi can be negative (which it always is)
        denom = torch.sign(seq_logp_for_loss) * torch.clamp(
            torch.abs(seq_logp_for_loss), min=self.denom_eps
        )

        # Compute weight
        w = numerator / denom

        # Optionally clip weight
        if self.clip_weight_value is not None:
            w = torch.clamp(
                w,
                min=-self.clip_weight_value,
                max=self.clip_weight_value,
            )

        return w, seq_logp_for_loss, seq_logp, kl_term

    def _compute_loss(self, model, inputs):
        """
        Compute the Online SFT loss.

        Loss = - [ w * s + 0.5 * stop_grad(w) * s^2 ].mean()

        where:
            w = weight computed by _compute_online_sft_weight
            s = sequence log probability (optionally normalized and clamped)
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Get per-token log probabilities and entropies
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Get reference model log probs if beta > 0
        ref_per_token_logps = inputs.get("ref_per_token_logps")

        # Get rewards - DynamicGRPOTrainer passes raw rewards in the output
        # Use advantages as fallback (which is rewards - mean for GRPO)
        rewards = inputs.get("rewards")
        if rewards is None:
            rewards = inputs["advantages"]

        # Compute Online SFT weight
        w, seq_logp_normalized, seq_logp_raw, kl_term = (
            self._compute_online_sft_weight(
                rewards=rewards,
                per_token_logps=per_token_logps,
                ref_per_token_logps=ref_per_token_logps,
                completion_mask=completion_mask,
            )
        )

        # Apply log prob clamping if configured
        s = seq_logp_normalized
        num_clamped = 0
        if self.clip_logprob_min is not None:
            s_clamped = torch.clamp(s, min=self.clip_logprob_min)
            num_clamped = int((s < self.clip_logprob_min).sum().item())
        else:
            s_clamped = s

        # Compute Online SFT loss:
        # L = - [ w * s + 0.5 * stop_grad(w) * s^2 ].mean()
        loss = -(w * s_clamped + 0.5 * w.detach() * (s_clamped**2)).mean()

        # Scale loss by gradient accumulation
        loss = loss / self.current_gradient_accumulation_steps

        # Log metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        # Weight metrics
        # Gather weights from all processes first, then compute statistics
        gathered_w = self.accelerator.gather(w)
        self._metrics[mode]["online_sft/weight/mean"].append(
            gathered_w.nanmean().item()
        )
        # Compute std across all gathered weights (not mean of per-process stds)
        if gathered_w.numel() > 1:
            self._metrics[mode]["online_sft/weight/std"].append(
                nanstd(gathered_w).item()
            )
        else:
            self._metrics[mode]["online_sft/weight/std"].append(0.0)
        self._metrics[mode]["online_sft/weight/min"].append(
            nanmin(gathered_w).item()
        )
        self._metrics[mode]["online_sft/weight/max"].append(
            nanmax(gathered_w).item()
        )

        # Log probability metrics
        self._metrics[mode]["online_sft/seq_logp_raw/mean"].append(
            self.accelerator.gather(seq_logp_raw.mean()).nanmean().item()
        )
        self._metrics[mode]["online_sft/seq_logp_normalized/mean"].append(
            self.accelerator.gather(seq_logp_normalized.mean()).nanmean().item()
        )

        # KL metrics
        if self.beta != 0.0:
            self._metrics[mode]["online_sft/kl/mean"].append(
                self.accelerator.gather(kl_term.mean()).nanmean().item()
            )

        # Entropy metrics
        mean_entropy = (
            entropies * completion_mask
        ).sum() / completion_token_count
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        # Clamping metrics
        if num_clamped > 0:
            self._metrics[mode]["online_sft/logp_clamped_count"].append(
                float(num_clamped)
            )
            self._metrics[mode]["online_sft/logp_clamped_pct"].append(
                100.0 * num_clamped / len(s)
            )

        return loss
