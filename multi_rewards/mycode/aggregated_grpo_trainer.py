"""GRPO trainer that aggregates rewards via a configurable aggregator."""

from __future__ import annotations

from typing import Optional

import torch

from dynamic_grpo_trainer import DynamicGRPOTrainer
from .aggregators import RewardAggregator


class AggregatedGRPOTrainer(DynamicGRPOTrainer):
    """Use a reward aggregator to combine per-objective rewards."""

    def __init__(
        self,
        *args,
        reward_aggregator: RewardAggregator,
        **kwargs,
    ):
        self.reward_aggregator = reward_aggregator
        super().__init__(*args, **kwargs)

    def _compute_group_advantages(
        self, rewards: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        device = rewards.device
        total = rewards.shape[0]
        if total == 0:
            return torch.zeros(0, device=device)

        group_ids = torch.div(
            torch.arange(total, device=device, dtype=torch.long),
            group_size,
            rounding_mode="floor",
        )
        group_count = group_ids.max().item() + 1

        counts = torch.bincount(group_ids, minlength=group_count).float()
        counts = counts.clamp(min=1.0)
        sums = torch.bincount(group_ids, weights=rewards, minlength=group_count)
        mean_per_group = sums / counts

        sq_sums = torch.bincount(
            group_ids, weights=rewards * rewards, minlength=group_count
        )
        var_per_group = torch.clamp(
            sq_sums / counts - mean_per_group * mean_per_group, min=0.0
        )
        std_per_group = torch.sqrt(var_per_group)
        std_per_group = torch.where(
            counts > 1, std_per_group, torch.ones_like(std_per_group)
        )

        mean_grouped = mean_per_group[group_ids]
        std_grouped = std_per_group[group_ids]
        advantages = rewards - mean_grouped
        if self.norm_adv_by_std_in_grpo:
            advantages = advantages / (std_grouped + 1e-6)
        return advantages

    def _prepare_loss_terms(
        self,
        inputs,
        per_token_logps: torch.Tensor,
        entropies: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> dict:
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        else:
            per_token_kl = None

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1)
            log_importance_weights = (
                log_importance_weights
                / completion_mask.sum(-1).clamp(min=1.0)
            )
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                "Unknown importance sampling level: "
                f"{self.importance_sampling_level}"
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
        )
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        return {
            "per_token_kl": per_token_kl,
            "entropies": entropies,
            "coef_1": coef_1,
            "coef_2": coef_2,
        }

    def _compute_loss_from_advantages(
        self,
        inputs,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_terms: dict,
    ) -> torch.Tensor:
        coef_1 = loss_terms["coef_1"]
        coef_2 = loss_terms["coef_2"]
        per_token_kl = loss_terms["per_token_kl"]
        entropies = loss_terms["entropies"]

        advantages_expanded = advantages.unsqueeze(1)
        pg_losses1 = -advantages_expanded * coef_1
        pg_losses2 = -advantages_expanded * coef_2
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

        pg_losses3 = -advantages_expanded * self.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        per_token_loss = torch.where(
            advantages_expanded < 0, clip_pg_losses2, clip_pg_losses1
        )

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.entropy_coeff != 0.0:
            per_token_loss = per_token_loss - self.entropy_coeff * entropies

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
            loss = loss / getattr(self, "current_gradient_accumulation_steps", 1)
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
                min=1.0
            )
            loss = loss / getattr(self, "current_gradient_accumulation_steps", 1)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            loss = loss / getattr(self, "current_gradient_accumulation_steps", 1)
        elif self.loss_type == "dapo":
            normalizer = (
                inputs["num_items_in_batch"].to(per_token_loss.dtype)
                / float(max(1, self.accelerator.num_processes))
            ).clamp(min=1.0)
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _compute_loss(self, model, inputs):
        rewards_per_func: Optional[torch.Tensor] = inputs.get("rewards_per_func")
        if rewards_per_func is None:
            return super()._compute_loss(model, inputs)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        if prompt_ids.size(0) == 0 or completion_ids.size(0) == 0:
            # Return a real tensor tied to model params to keep autograd happy.
            for param in model.parameters():
                if param.requires_grad:
                    return param.sum() * 0.0
            return torch.tensor(0.0, device=completion_mask.device)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

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

        loss_terms = self._prepare_loss_terms(
            inputs, per_token_logps, entropies, completion_mask
        )

        total = rewards_per_func.shape[0]
        num_gen = max(1, int(self.num_generations))
        group_size = num_gen if total % num_gen == 0 else 1

        scalar_rewards = self.reward_aggregator.aggregate(
            rewards_per_func, inputs=inputs
        )
        metrics = self.reward_aggregator.update(
            rewards_per_func, inputs=inputs, training=self.model.training
        )
        if metrics and hasattr(self, "_metrics"):
            bucket = "train" if self.model.training else "eval"
            self._metrics.setdefault(bucket, {})
            for key, value in metrics.items():
                metric_key = f"aggregator/{self.reward_aggregator.name}/{key}"
                self._metrics[bucket].setdefault(metric_key, []).append(value)

        advantages = self._compute_group_advantages(scalar_rewards, group_size)

        loss = self._compute_loss_from_advantages(
            inputs, advantages, completion_mask, loss_terms
        )
        return loss
