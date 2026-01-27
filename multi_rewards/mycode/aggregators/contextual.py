"""Context-dependent reward weights based on prompt heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from .base import RewardAggregator, normalize_weights


@dataclass
class ContextualWeightAggregator(RewardAggregator):
    """Heuristic contextual weighting based on prompt length/keywords."""

    base_weights: torch.Tensor
    short_length: int = 120
    long_length: int = 300
    short_weights: Optional[Sequence[float]] = None
    long_weights: Optional[Sequence[float]] = None
    clarity_weights: Optional[Sequence[float]] = None
    conciseness_weights: Optional[Sequence[float]] = None

    clarity_keywords: Sequence[str] = (
        "prove",
        "show",
        "derive",
        "explain",
        "reason",
    )
    conciseness_keywords: Sequence[str] = (
        "short",
        "brief",
        "concise",
        "quick",
    )

    name: str = "contextual"

    def __post_init__(self) -> None:
        if not isinstance(self.base_weights, torch.Tensor):
            self.base_weights = torch.tensor(self.base_weights, dtype=torch.float32)
        self.base_weights = normalize_weights(self.base_weights)
        base_len = self.base_weights.numel()
        if self.short_weights is None:
            self.short_weights = self.base_weights.tolist()
        if self.long_weights is None:
            self.long_weights = self.base_weights.tolist()
        if self.clarity_weights is None:
            self.clarity_weights = self.base_weights.tolist()
        if self.conciseness_weights is None:
            self.conciseness_weights = self.base_weights.tolist()
        for name, weights in (
            ("short_weights", self.short_weights),
            ("long_weights", self.long_weights),
            ("clarity_weights", self.clarity_weights),
            ("conciseness_weights", self.conciseness_weights),
        ):
            if len(weights) != base_len:
                raise ValueError(
                    f"{name} length {len(weights)} does not match base weights {base_len}"
                )

    def _select_weights(self, prompt: str) -> torch.Tensor:
        prompt_lower = prompt.lower()
        if any(k in prompt_lower for k in self.conciseness_keywords):
            return torch.tensor(self.conciseness_weights, dtype=torch.float32)
        if any(k in prompt_lower for k in self.clarity_keywords):
            return torch.tensor(self.clarity_weights, dtype=torch.float32)
        if len(prompt_lower) <= self.short_length:
            return torch.tensor(self.short_weights, dtype=torch.float32)
        if len(prompt_lower) >= self.long_length:
            return torch.tensor(self.long_weights, dtype=torch.float32)
        return self.base_weights.clone().detach()

    def aggregate(
        self,
        rewards_per_func: torch.Tensor,
        *,
        inputs: Optional[Mapping] = None,
    ) -> torch.Tensor:
        device = rewards_per_func.device
        prompts = []
        if inputs:
            prompts = (
                inputs.get("prompts_text")
                or inputs.get("prompts")
                or inputs.get("prompt")
                or []
            )
        if not isinstance(prompts, list):
            prompts = list(prompts) if prompts is not None else []

        if not prompts or len(prompts) != rewards_per_func.shape[0]:
            weights = self.base_weights.to(device)
            return (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)

        weights_list = []
        for prompt in prompts:
            w = self._select_weights(str(prompt))
            weights_list.append(normalize_weights(w))
        weight_tensor = torch.stack(weights_list, dim=0).to(device)
        return (rewards_per_func * weight_tensor).sum(dim=1)
