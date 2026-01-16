"""Trainer callbacks for multi-objective reward scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object

from .multi_objective_reward import ParetoMetaReward


@dataclass
class ParetoMetaRewardCallback(TrainerCallback):
    """Update reward weights using hypervolume gain on evaluation."""

    meta_reward: ParetoMetaReward
    base_weights: Sequence[float]
    trainer: Optional[object] = None

    def __post_init__(self):
        self.last_meta_scale: Optional[float] = None
        self.last_delta_hv: Optional[float] = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = self.trainer or kwargs.get("trainer")
        point = self._extract_point(metrics, trainer)
        if point is None:
            return control

        meta_scale, delta_hv = self.meta_reward.update_with_delta(point)
        self.last_meta_scale = meta_scale
        self.last_delta_hv = delta_hv

        if trainer is not None and hasattr(trainer, "reward_weights"):
            weights = [w * meta_scale for w in self.base_weights]
            trainer.reward_weights = trainer.reward_weights.new_tensor(weights)

        if metrics is not None:
            metrics["pareto/meta_scale"] = meta_scale
            metrics["pareto/hv_delta"] = delta_hv

        return control

    @staticmethod
    def _extract_point(metrics: Optional[Dict], trainer) -> Optional[tuple]:
        keys = {
            "accuracy": [
                "eval/rewards/accuracy/mean",
                "rewards/accuracy/mean",
            ],
            "conciseness": [
                "eval/rewards/conciseness/mean",
                "rewards/conciseness/mean",
            ],
            "clarity": [
                "eval/rewards/clarity/mean",
                "rewards/clarity/mean",
            ],
        }

        def get_metric(name):
            if metrics:
                for key in keys[name]:
                    if key in metrics:
                        return metrics[key]
            if trainer is not None and hasattr(trainer, "_metrics"):
                eval_metrics = trainer._metrics.get("eval", {})
                key = f"rewards/{name}/mean"
                if key in eval_metrics and eval_metrics[key]:
                    return eval_metrics[key][-1]
            return None

        acc = get_metric("accuracy")
        conc = get_metric("conciseness")
        clar = get_metric("clarity")
        if acc is None or conc is None or clar is None:
            return None
        return (float(acc), float(conc), float(clar))
