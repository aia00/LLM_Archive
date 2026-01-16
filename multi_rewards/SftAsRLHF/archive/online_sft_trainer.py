# online_sft_trainer.py
# MIT License
# (c) 2025 - Example implementation of "Online SFT = RLHF" trainer
# Implements the gradient estimator from the paper that shows RLHF
# equals weighted online SFT when using the weight in Eq. (2) and the
# unbiased gradient estimator of Proposition 3.
#
# The core loss we minimize for each (prompt x, sampled response y) is:
#     L_hat = - [ w_theta(x,y) * log_pi_theta(y|x) + 0.5 * stop_grad(w_theta(x,y)) * (log_pi_theta(y|x))^2 ]
# whose gradient equals the unbiased estimator of ∇_θ E_{y~π_θ}[-w_θ log π_θ]
# derived in the paper (see Proposition 3).
#
# References:
#  - Weighted online SFT objective and gradient: Eq. (1), Proposition 6
#  - RLHF ↔ weighted online SFT equivalence and weight: Eq. (2), Proposition 7
#  - Unbiased pathwise gradient estimator with on-policy sampling: Proposition 3
#
# Requirements:
#   torch >= 2.0
#   transformers >= 4.38
#
# Notes:
#   * This trainer is intentionally lightweight and does not require Accelerate.
#   * It mirrors the ergonomics of TRL trainers (e.g., step() returns metrics).
#   * The reward model is evaluated under torch.no_grad(); no gradients flow
#     into it. The reference model is fixed/frozen.
#
# Author: ChatGPT (GPT-5 Pro)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _shifted_log_probs(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-token log-probabilities log p(y_t | x, y_<t) at each time step,
    using teacher forcing. `logits` and `labels` should already be time-aligned
    such that logits[:, t] predicts labels[:, t].

    Args:
        logits: (B, T, V)
        labels: (B, T) token ids; positions with -100 are ignored

    Returns:
        token_logps: (B, T) log-probabilities with 0 at ignored positions
    """
    logprobs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Mask for valid positions (not -100)
    mask = labels != -100

    # Replace -100 with 0 (any valid index) for gather operation
    # The actual values don't matter since we'll mask them out
    labels_for_gather = labels.clone()
    labels_for_gather[~mask] = 0

    # Use gather to pick the probability of the true next token at each position
    gather = torch.gather(
        logprobs, dim=-1, index=labels_for_gather.unsqueeze(-1)
    ).squeeze(
        -1
    )  # (B, T)

    # Mask out ignored positions
    token_logps = gather * mask.to(gather.dtype)
    return token_logps


def sequence_logprob_sum(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the *sequence-level* log-probability SUM over response tokens:
        log π(y | x) = sum_{t in response} log p(y_t | x, y_<t)

    Args:
        model: causal LM (AutoModelForCausalLM or compatible)
        input_ids: (B, S) prompt + response tokens returned by `generate`
        attention_mask: (B, S)
        response_mask: (B, S) boolean mask that is 1 exactly at positions
                       belonging to the response (not the prompt). The first
                       token has no next-token label; we handle shift below.

    Returns:
        seq_logp: (B,) sum of log-probs over response tokens only
    """
    # Shift for next-token prediction
    labels = input_ids.clone()
    # Ignore prompt tokens & the very first position (no previous token)
    labels[~response_mask] = -100
    labels[:, 0] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, S, V)

    # Align logits & labels for next-token prediction
    token_logps = _shifted_log_probs(
        logits[:, :-1, :], labels[:, 1:]
    )  # (B, S-1)

    # Sum only over response positions (labels != -100)
    seq_logp = token_logps.sum(dim=-1)  # (B,)
    return seq_logp


@dataclass
class OnlineSFTTrainerConfig:
    beta: float = 0.1
    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    num_return_sequences: int = 1

    # Optimization
    lr: float = 1e-6
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = False
    # Gradient accumulation (optimizer step every N microbatches)
    grad_accum_steps: int = 1

    # Stabilization
    denom_eps: float = 1e-8  # epsilon to avoid divide-by-zero in logπ
    clip_weight_value: Optional[float] = (
        1  # optional |w| clipping (None to disable)
    )
    clip_logprob_min: Optional[float] = (
        -200.0  # clamp log π(y|x) from below to prevent loss explosion (None to disable)
        # NOTE: When normalize_logprob_by_length=True, use smaller values like -0.5 or -1.0
    )
    normalize_logprob_by_length: bool = (
        False  # If True, normalize log π(y|x) by response length before computing loss
        # This prevents long sequences from dominating the loss and provides better stability
    )

    # Reward handling
    reward_use_sigmoid: Optional[bool] = (
        None  # If None: auto-detect (num_labels==1 -> raw; num_labels==2 -> sigmoid)
    )
    reward_scale: float = 1.0  # scale r before using in w
    reward_shift: float = 0.0  # shift r before using in w

    # Logging
    logit_mean_max: bool = False  # extra metric: mean/max logits (debug)
    length_normalize_kl: bool = True  # report KL per token approximation


class OnlineSFTTrainer:
    """
    A minimal trainer that performs "online SFT" updates with the
    weight w_θ(x,y) from Eq. (2) and the unbiased gradient estimator
    of Proposition 3, achieving the same effect as RLHF but without PPO.
    """

    def __init__(
        self,
        policy: PreTrainedModel,
        ref_policy: PreTrainedModel,
        reward_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # Optional adapter to convert reward model outputs to scalar rewards
        # Signature: (model_output, tokenized_inputs, texts) -> torch.Tensor[B]
        reward_output_adapter: Optional[
            Callable[[Any, Dict[str, torch.Tensor], List[str]], torch.Tensor]
        ] = None,
        cfg: OnlineSFTTrainerConfig = OnlineSFTTrainerConfig(),
        device: Optional[torch.device] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        # Use RM's tokenizer if provided; otherwise fall back to policy tokenizer
        self.reward_tokenizer = reward_tokenizer or tokenizer

        # Ensure tokenizers/models have pad token ids configured for batch > 1
        # Reward tokenizer must have a pad token for RM forward on batch inputs
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        # Also ensure the primary tokenizer has a pad token for generation/forward
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Propagate pad_token_id to model configs to avoid GPT-NeoX batch>1 error
        pad_id_rm = self.reward_tokenizer.pad_token_id
        if (
            getattr(
                getattr(self.reward_model, "config", None), "pad_token_id", None
            )
            is None
        ):
            try:
                self.reward_model.config.pad_token_id = pad_id_rm
            except Exception:
                pass
        pad_id_pol = self.tokenizer.pad_token_id
        for m in (self.policy, self.ref_policy):
            if (
                getattr(getattr(m, "config", None), "pad_token_id", None)
                is None
            ):
                try:
                    m.config.pad_token_id = pad_id_pol
                except Exception:
                    pass

        self.cfg = cfg
        self.device = device or next(policy.parameters()).device
        # Optional adapter/hook for RM outputs
        self.reward_output_adapter = reward_output_adapter

        # Only move models to device if they don't already have device_map set
        # (device_map="auto" already places models on appropriate devices)
        if not hasattr(self.policy, "hf_device_map"):
            self.policy.to(self.device)
        if not hasattr(self.ref_policy, "hf_device_map"):
            self.ref_policy.to(self.device)
        if not hasattr(self.reward_model, "hf_device_map"):
            self.reward_model.to(self.device)

        self.ref_policy.eval()  # keep frozen
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # Use new torch.amp.GradScaler API to avoid deprecation
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            device_type,
            enabled=cfg.use_amp and self.device.type == "cuda",
        )

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=cfg.lr,
                betas=cfg.adam_betas,
                eps=cfg.adam_eps,
                weight_decay=cfg.weight_decay,
            )
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Gradient accumulation state
        self.grad_accum_steps = max(
            1, int(getattr(self.cfg, "grad_accum_steps", 1))
        )
        self._accum_count = 0

    @torch.no_grad()
    def _generate(
        self, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run autoregressive sampling from the current policy.

        Returns:
            seqs: (B * K, S_total) prompt+response
            attn: (B * K, S_total)
            prompt_lens: (B * K,) prompt lengths for each sequence
        """
        B = prompt_ids.size(0)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = eos_id
        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
            top_p=self.cfg.top_p,
            num_return_sequences=self.cfg.num_return_sequences,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        self.policy.eval()
        seqs = self.policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            **gen_kwargs,
        )
        self.policy.train()

        # When num_return_sequences>1, HF repeats each prompt K times in order.
        K = self.cfg.num_return_sequences
        if K > 1:
            prompt_lens = torch.repeat_interleave(prompt_attn.sum(dim=1), K)
        else:
            prompt_lens = prompt_attn.sum(dim=1)

        # Build attention for the full sequences (already right-padded by generate)
        attn = (seqs != self.tokenizer.pad_token_id).long()
        return seqs, attn, prompt_lens

    @staticmethod
    def _make_response_mask(
        seqs: torch.Tensor, prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        B, S = seqs.shape
        device = seqs.device
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        # response positions are >= prompt_len for each sequence
        resp_mask = positions >= prompt_lens.unsqueeze(1)
        return resp_mask

    @torch.no_grad()
    def _compute_rewards(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Compute scalar rewards for each text by running the reward model and
        post-processing its outputs. If a custom adapter is provided, it is used;
        otherwise apply a default heuristic using cfg.reward_use_sigmoid and the
        reward model's config.
        """
        # Tokenize with reward tokenizer for RM compatibility
        toks = self.reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Long enough for prompt + response
        ).to(self.device)

        # If the reward tokenizer uses a newly-added PAD that the RM didn't resize for,
        # remap padded positions to EOS to avoid OOV embedding lookups.
        try:
            attn = toks.get("attention_mask")
            if attn is not None and "input_ids" in toks:
                input_ids = toks["input_ids"]
                pad_positions = attn == 0
                if pad_positions.any():
                    eos_id_rm = self.reward_tokenizer.eos_token_id
                    if eos_id_rm is not None:
                        input_ids[pad_positions] = eos_id_rm
                        toks["input_ids"] = input_ids
        except Exception:
            pass

        out = self.reward_model(**toks)

        # Use adapter hook if provided
        if self.reward_output_adapter is not None:
            r = self.reward_output_adapter(out, toks, texts)
        else:
            # Default behavior: extract logits and map to scalar rewards
            logits = out.logits if hasattr(out, "logits") else out[0]

            use_sigmoid = self.cfg.reward_use_sigmoid
            num_labels = getattr(
                getattr(self.reward_model, "config", None), "num_labels", None
            )

            # Auto-detect if not specified
            if use_sigmoid is None:
                if (num_labels == 2) or (
                    logits.dim() > 1 and logits.size(-1) == 2
                ):
                    use_sigmoid = True
                else:
                    use_sigmoid = False

            if use_sigmoid:
                # Map to probability of the "positive" class
                if logits.dim() > 1 and logits.size(-1) == 2:
                    # Two-logit classifier -> softmax prob for class 1
                    r = torch.softmax(logits, dim=-1)[..., 1]
                else:
                    # Single logit -> sigmoid
                    r = torch.sigmoid(logits)
            else:
                # Treat as regression/raw scores
                r = logits

            # Reduce to per-example scalar: allow shapes [B], [B,1], [B,k,...]
            while r.dim() > 1:
                if r.size(-1) == 1:
                    r = r.squeeze(-1)
                else:
                    r = r.mean(dim=-1)

        r = r * self.cfg.reward_scale + self.cfg.reward_shift
        return r.detach()

    def _compute_weight(
        self,
        reward: torch.Tensor,
        logp_pi: torch.Tensor,
        logp_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        w_θ(x,y) = [ r(x,y) - β * (log π_θ(y|x) - log π_ref(y|x)) ] / log π_θ(y|x)

        A tiny ε is added to the denominator to avoid division by (near) zero.
        """
        eps = self.cfg.denom_eps

        # Ensure |denom| >= eps to prevent division by near-zero
        # Log probabilities are always ≤ 0, so we clamp the absolute value
        denom = torch.sign(logp_pi) * torch.clamp(torch.abs(logp_pi), min=eps)

        w = (reward - self.cfg.beta * (logp_pi - logp_ref)) / denom
        if self.cfg.clip_weight_value is not None:
            w = torch.clamp(
                w,
                min=-self.cfg.clip_weight_value,
                max=self.cfg.clip_weight_value,
            )
        return w

    def step(self, prompt_batch: List[str]) -> Dict[str, float]:
        """
        One on-policy update step:
          1) sample y ~ π_θ(.|x) for each prompt x
          2) compute r(x,y), log-probs under π_θ and π_ref
          3) form w_θ(x,y) and minimize the surrogate loss L_hat

        Returns:
          metrics dict suitable for logging.
        """
        # 1) Tokenize prompts
        toks = self.tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,  # Reasonable max for prompts
        ).to(self.device)
        prompt_ids = toks["input_ids"]
        prompt_attn = toks["attention_mask"]

        # 2) Sample continuations from current policy
        seqs, attn, prompt_lens = self._generate(
            prompt_ids, prompt_attn
        )  # (B*K, S), (B*K, S), (B*K,)
        B = seqs.size(0)
        resp_mask = self._make_response_mask(seqs, prompt_lens) & (attn.bool())
        resp_lens = resp_mask.sum(dim=1)
        # If any responses are empty, attempt one resampling. Warn if all empty or all identical.
        if (resp_lens == 0).any():
            # Resample only once to avoid infinite loops
            seqs2, attn2, prompt_lens2 = self._generate(prompt_ids, prompt_attn)
            resp_mask2 = self._make_response_mask(seqs2, prompt_lens2) & (
                attn2.bool()
            )
            resp_lens2 = resp_mask2.sum(dim=1)
            # Replace empty ones with resampled sequences
            replace_idx = resp_lens == 0
            if replace_idx.any():
                seqs[replace_idx] = seqs2[replace_idx]
                attn[replace_idx] = attn2[replace_idx]
                prompt_lens[replace_idx] = prompt_lens2[replace_idx]
                resp_mask[replace_idx] = resp_mask2[replace_idx]
                resp_lens[replace_idx] = resp_lens2[replace_idx]

        # Filter out any remaining empty responses
        keep_idx = resp_lens > 0
        num_removed = int((~keep_idx).sum().item())
        if not keep_idx.any():
            # Nothing to train on this step
            return {
                "warn/all_empty_responses": 1.0,
                "skipped/empty_batch": 1.0,
                "filter/num_removed": float(num_removed),
            }

        # Slice tensors/texts to keep only non-empty responses
        seqs = seqs[keep_idx]
        attn = attn[keep_idx]
        prompt_lens = prompt_lens[keep_idx]
        resp_mask = resp_mask[keep_idx]
        resp_lens = resp_lens[keep_idx]
        # Detect completely identical decoded outputs on kept items
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        all_same = len(set(texts)) == 1

        # 3-7) Compute log-probs, rewards, loss, and optimize under enabled grad
        with torch.enable_grad():
            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.cfg.use_amp and self.device.type == "cuda",
            ):
                logp_pi = sequence_logprob_sum(
                    self.policy, seqs, attn, resp_mask
                )  # (B,)
                with torch.no_grad():
                    logp_ref = sequence_logprob_sum(
                        self.ref_policy, seqs, attn, resp_mask
                    )  # (B,)

            # 4) Compute rewards r(x,y)
            rewards = self._compute_rewards(texts)  # (B,)

            # 5) Form weight w_θ(x,y)
            # NOTE: w depends on θ via logp_pi in both numerator and denominator
            w = self._compute_weight(rewards, logp_pi, logp_ref)  # (B,)

            # 6) Build the "pseudo-loss" L_hat whose gradient matches Proposition 3
            #     L_hat = - [ w * s + 0.5 * stop_grad(w) * s^2 ],  where s = logπ_θ(y|x)
            s = logp_pi
            s_raw = logp_pi  # Keep raw value for metrics

            # Compute response lengths for normalization
            resp_lens = resp_mask.sum(dim=1).clamp_min(1)

            # Optional: Normalize by sequence length (Solution 2)
            # This prevents long sequences from dominating the loss
            if self.cfg.normalize_logprob_by_length:
                s = s / resp_lens

            # Clamp log probabilities to prevent loss explosion from very negative values (Solution 1)
            # When log π(y|x) is very negative (e.g., -100), the s^2 term causes instability
            s_clamped = s
            num_clamped = 0
            if self.cfg.clip_logprob_min is not None:
                s_clamped = torch.clamp(s, min=self.cfg.clip_logprob_min)
                num_clamped = int((s < self.cfg.clip_logprob_min).sum().item())

            loss = -(w * s_clamped + 0.5 * w.detach() * (s_clamped**2)).mean()

            # 7) Optimize policy
            # Gradient accumulation: only zero at start of window
            if self._accum_count == 0:
                self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self._accum_count += 1
            if self._accum_count >= self.grad_accum_steps:
                # Clip and step when finishing an accumulation window
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.policy.parameters(), self.cfg.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self._accum_count = 0

        # 8) Metrics
        # Approximate pathwise KL: E[ log π_θ - log π_ref ] over samples equals KL(π_θ || π_ref)
        kl_est = logp_pi - logp_ref
        # Token-length for normalization (resp_lens already computed above during loss computation)
        if self.cfg.length_normalize_kl:
            kl_per_tok = kl_est / resp_lens
            s_raw_per_tok = s_raw / resp_lens
        else:
            kl_per_tok = kl_est
            s_raw_per_tok = s_raw

        with torch.no_grad():
            metrics = {
                "loss": float(loss.detach().cpu()),
                "reward/mean": float(rewards.mean().cpu()),
                "reward/std": float(rewards.std(unbiased=False).cpu()),
                # Raw sequence-level log probs (before any normalization)
                "policy/logp_seq_raw/mean": float(s_raw.mean().cpu()),
                "policy/logp_seq_raw/std": float(s_raw.std(unbiased=False).cpu()),
                "policy/logp_seq_raw/min": float(s_raw.min().cpu()),
                "policy/logp_seq_raw/max": float(s_raw.max().cpu()),
                "policy/logp_seq_per_tok/mean": float(s_raw_per_tok.mean().cpu()),
                "policy/kl_y_ref/mean": float(kl_est.mean().cpu()),
                "policy/kl_y_ref_per_tok/mean": float(kl_per_tok.mean().cpu()),
                "weight/mean": float(w.mean().detach().cpu()),
                "weight/std": float(w.std(unbiased=False).detach().cpu()),
                "weight/min": float(w.min().detach().cpu()),
                "weight/max": float(w.max().detach().cpu()),
                "response/avg_len_tokens": float(
                    resp_lens.float().mean().cpu()
                ),
            }
            # Report the value of s used in loss (could be normalized)
            if self.cfg.normalize_logprob_by_length:
                metrics["policy/logp_used_in_loss/mean"] = float(s.mean().cpu())
                metrics["policy/logp_used_in_loss/std"] = float(s.std(unbiased=False).cpu())
                metrics["policy/logp_used_in_loss/min"] = float(s.min().cpu())
                metrics["policy/logp_used_in_loss/max"] = float(s.max().cpu())
                metrics["config/normalized_by_length"] = 1.0
            # Track log probability clamping statistics
            if self.cfg.clip_logprob_min is not None and num_clamped > 0:
                metrics["clip/num_logprob_clamped"] = float(num_clamped)
                metrics["clip/pct_logprob_clamped"] = (
                    100.0 * num_clamped / len(s)
                )
                metrics["policy/logp_clamped/mean"] = float(
                    s_clamped.mean().cpu()
                )
            # Add warnings/filters to metrics for user visibility
            if num_removed > 0:
                metrics["filter/num_removed"] = float(num_removed)
            if all_same:
                metrics["warn/all_identical_responses"] = 1.0
            if self.cfg.logit_mean_max:
                # Optional debug: mean & max of logits at response positions
                with torch.no_grad():
                    out = self.policy(input_ids=seqs, attention_mask=attn)
                    logits = out.logits[:, :-1, :]
                    # mask to just response positions (aligned to logits)
                    rmask = resp_mask[:, 1:]
                    sel = logits[rmask]
                    if sel.numel() > 0:
                        metrics["debug/logits/mean"] = float(sel.mean().cpu())
                        metrics["debug/logits/max"] = float(sel.max().cpu())
        return metrics
