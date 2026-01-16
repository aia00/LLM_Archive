# online_sft/trainer.py
# MIT License
# Implements the "Online SFT = RLHF" trainer with modular reward support
#
# The core loss we minimize for each (prompt x, sampled response y) is:
#     L_hat = - [ w_theta(x,y) * log_pi_theta(y|x) + 0.5 * stop_grad(w_theta(x,y)) * (log_pi_theta(y|x))^2 ]
# whose gradient equals the unbiased estimator of nabla_theta E_{y~pi_theta}[-w_theta log pi_theta]
# derived in the paper (see Proposition 3).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    labels_for_gather = labels.clone()
    labels_for_gather[~mask] = 0

    # Use gather to pick the probability of the true next token at each position
    gather = torch.gather(
        logprobs, dim=-1, index=labels_for_gather.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

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
        log pi(y | x) = sum_{t in response} log p(y_t | x, y_<t)

    Args:
        model: causal LM (AutoModelForCausalLM or compatible)
        input_ids: (B, S) prompt + response tokens returned by `generate`
        attention_mask: (B, S)
        response_mask: (B, S) boolean mask that is 1 exactly at positions
                       belonging to the response (not the prompt).

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
    """Configuration for OnlineSFTTrainer."""

    beta: float = 0.1

    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    num_return_sequences: int = 8  # Multiple responses per prompt for variance

    # Group filtering (DAPO-style)
    # Keep only groups (all responses for a prompt) with reward variance > 0
    # i.e., at least one correct and one incorrect response
    filter_groups_enable: bool = True
    filter_groups_min_std: float = 0.0  # Minimum std to keep a group (> 0 means must have variance)
    max_filter_retries: int = 10  # Max retries if all groups filtered

    # Optimization
    lr: float = 1e-6
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = False
    grad_accum_steps: int = 1

    # Stabilization
    denom_eps: float = 1e-8
    clip_weight_value: Optional[float] = 1.0
    clip_logprob_min: Optional[float] = -200.0
    normalize_logprob_by_length: bool = False

    # Reward handling (only used when using neural reward model)
    reward_use_sigmoid: Optional[bool] = None
    reward_scale: float = 1.0
    reward_shift: float = 0.0

    # Logging
    logit_mean_max: bool = False
    length_normalize_kl: bool = True


class OnlineSFTTrainer:
    """
    A minimal trainer that performs "online SFT" updates with the
    weight w_theta(x,y) from Eq. (2) and the unbiased gradient estimator
    of Proposition 3, achieving the same effect as RLHF but without PPO.

    Supports both:
    - Neural reward models (AutoModelForSequenceClassification)
    - Rule-based reward functions (e.g., math verification)
    """

    def __init__(
        self,
        policy: PreTrainedModel,
        ref_policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        # Reward: either a model or a callable function
        reward_model: Optional[PreTrainedModel] = None,
        reward_fn: Optional[Callable] = None,
        reward_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # Optional adapter to convert reward model outputs to scalar rewards
        reward_output_adapter: Optional[
            Callable[[Any, Dict[str, torch.Tensor], List[str]], torch.Tensor]
        ] = None,
        cfg: OnlineSFTTrainerConfig = None,
        device: Optional[torch.device] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        if cfg is None:
            cfg = OnlineSFTTrainerConfig()

        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer or tokenizer

        # Validate reward configuration
        if reward_model is None and reward_fn is None:
            raise ValueError("Either reward_model or reward_fn must be provided")

        # Ensure tokenizers have pad tokens configured
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Propagate pad_token_id to model configs
        pad_id_pol = self.tokenizer.pad_token_id
        for m in (self.policy, self.ref_policy):
            if getattr(getattr(m, "config", None), "pad_token_id", None) is None:
                try:
                    m.config.pad_token_id = pad_id_pol
                except Exception:
                    pass

        if self.reward_model is not None:
            pad_id_rm = self.reward_tokenizer.pad_token_id
            if getattr(getattr(self.reward_model, "config", None), "pad_token_id", None) is None:
                try:
                    self.reward_model.config.pad_token_id = pad_id_rm
                except Exception:
                    pass

        self.cfg = cfg
        self.device = device or next(policy.parameters()).device
        self.reward_output_adapter = reward_output_adapter

        # Only move models to device if they don't have device_map
        if not hasattr(self.policy, "hf_device_map"):
            self.policy.to(self.device)
        if not hasattr(self.ref_policy, "hf_device_map"):
            self.ref_policy.to(self.device)
        if self.reward_model is not None and not hasattr(self.reward_model, "hf_device_map"):
            self.reward_model.to(self.device)

        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # AMP scaler
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
        self.grad_accum_steps = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        self._accum_count = 0

        # Prompt source for backfilling (set via set_prompt_source)
        self._prompt_source = None
        self._prompt_source_iter = None

    def set_prompt_source(self, dataset, collate_fn=None):
        """
        Set the prompt source for backfilling filtered groups.

        Args:
            dataset: A dataset that yields (prompt, ground_truth) tuples or just prompts
            collate_fn: Optional collate function for batching
        """
        self._prompt_source = dataset
        self._prompt_source_collate_fn = collate_fn
        self._prompt_source_iter = None  # Will be created on first use

    def _sample_backfill_prompts(self, num_prompts: int) -> Tuple[List[str], Optional[List[str]]]:
        """
        Sample new prompts for backfilling filtered groups.

        Args:
            num_prompts: Number of prompts to sample

        Returns:
            prompts: List of prompt strings
            ground_truths: List of ground truth strings (or None if not available)
        """
        if self._prompt_source is None:
            return [], None

        # Create iterator if needed
        if self._prompt_source_iter is None:
            import random
            # Create a shuffled list of indices
            indices = list(range(len(self._prompt_source)))
            random.shuffle(indices)
            self._prompt_source_indices = indices
            self._prompt_source_idx = 0

        prompts = []
        ground_truths = []
        has_ground_truths = False

        for _ in range(num_prompts):
            # Wrap around if we've exhausted the dataset
            if self._prompt_source_idx >= len(self._prompt_source_indices):
                import random
                random.shuffle(self._prompt_source_indices)
                self._prompt_source_idx = 0

            idx = self._prompt_source_indices[self._prompt_source_idx]
            self._prompt_source_idx += 1

            item = self._prompt_source[idx]

            # Handle different dataset formats
            if isinstance(item, dict):
                if "prompt" in item:
                    prompts.append(item["prompt"])
                elif "query" in item:
                    prompts.append(item["query"])
                else:
                    prompts.append(str(item))

                if "ground_truth" in item:
                    ground_truths.append(item["ground_truth"])
                    has_ground_truths = True
                elif "answer" in item:
                    ground_truths.append(item["answer"])
                    has_ground_truths = True
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                prompts.append(item[0])
                ground_truths.append(item[1])
                has_ground_truths = True
            else:
                prompts.append(str(item))

        return prompts, ground_truths if has_ground_truths else None

    @torch.no_grad()
    def _generate(
        self, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run autoregressive sampling from the current policy."""
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

        K = self.cfg.num_return_sequences
        if K > 1:
            prompt_lens = torch.repeat_interleave(prompt_attn.sum(dim=1), K)
        else:
            prompt_lens = prompt_attn.sum(dim=1)

        attn = (seqs != self.tokenizer.pad_token_id).long()
        return seqs, attn, prompt_lens

    @staticmethod
    def _make_response_mask(
        seqs: torch.Tensor, prompt_lens: torch.Tensor
    ) -> torch.Tensor:
        B, S = seqs.shape
        device = seqs.device
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        resp_mask = positions >= prompt_lens.unsqueeze(1)
        return resp_mask

    @torch.no_grad()
    def _compute_rewards_neural(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """Compute scalar rewards using neural reward model."""
        toks = self.reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Handle OOV padding tokens
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

        if self.reward_output_adapter is not None:
            r = self.reward_output_adapter(out, toks, texts)
        else:
            logits = out.logits if hasattr(out, "logits") else out[0]

            use_sigmoid = self.cfg.reward_use_sigmoid
            num_labels = getattr(
                getattr(self.reward_model, "config", None), "num_labels", None
            )

            if use_sigmoid is None:
                if (num_labels == 2) or (logits.dim() > 1 and logits.size(-1) == 2):
                    use_sigmoid = True
                else:
                    use_sigmoid = False

            if use_sigmoid:
                if logits.dim() > 1 and logits.size(-1) == 2:
                    r = torch.softmax(logits, dim=-1)[..., 1]
                else:
                    r = torch.sigmoid(logits)
            else:
                r = logits

            while r.dim() > 1:
                if r.size(-1) == 1:
                    r = r.squeeze(-1)
                else:
                    r = r.mean(dim=-1)

        r = r * self.cfg.reward_scale + self.cfg.reward_shift
        return r.detach()

    @torch.no_grad()
    def _compute_rewards(
        self,
        texts: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute rewards using either neural model or rule-based function."""
        if self.reward_fn is not None:
            # Use rule-based reward function
            rewards = self.reward_fn(
                completions=texts,
                prompts=prompts or [""] * len(texts),
                **kwargs,
            )
            return torch.tensor(rewards, device=self.device, dtype=torch.float32)
        else:
            # Use neural reward model
            return self._compute_rewards_neural(texts)

    def _compute_weight(
        self,
        reward: torch.Tensor,
        logp_pi: torch.Tensor,
        logp_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        w_theta(x,y) = [ r(x,y) - beta * (log pi_theta(y|x) - log pi_ref(y|x)) ] / log pi_theta(y|x)
        """
        eps = self.cfg.denom_eps
        denom = torch.sign(logp_pi) * torch.clamp(torch.abs(logp_pi), min=eps)
        w = (reward - self.cfg.beta * (logp_pi - logp_ref)) / denom
        if self.cfg.clip_weight_value is not None:
            w = torch.clamp(
                w,
                min=-self.cfg.clip_weight_value,
                max=self.cfg.clip_weight_value,
            )
        return w

    def _filter_groups_by_reward_std(
        self,
        rewards: torch.Tensor,
        K: int,
        num_prompts: int,
    ) -> Tuple[torch.Tensor, int, int, Dict[str, float]]:
        """
        Filter groups by reward standard deviation.

        Groups with std <= filter_groups_min_std are filtered out (all correct or all wrong).
        This ensures training signal has both positive and negative examples.

        Args:
            rewards: (B,) tensor of rewards where B = num_prompts * K
            K: number of responses per prompt
            num_prompts: number of prompts in the batch

        Returns:
            keep_mask: (B,) boolean mask of which samples to keep
            num_kept_groups: number of groups kept
            num_filtered_groups: number of groups filtered
            filter_stats: detailed filtering statistics
        """
        # Reshape rewards to (num_prompts, K)
        rewards_grouped = rewards.view(num_prompts, K)

        # Compute stats per group
        group_mean = rewards_grouped.mean(dim=1)  # (num_prompts,)
        group_std = rewards_grouped.std(dim=1, unbiased=False)  # (num_prompts,)

        # Categorize groups
        # All correct: min > 0 (all positive rewards)
        # All wrong: max < 0 (all negative rewards)
        # Mixed: has both positive and negative
        num_correct_per_group = (rewards_grouped > 0).sum(dim=1)  # count of correct per group

        all_correct_groups = (num_correct_per_group == K)  # all K responses correct
        all_wrong_groups = (num_correct_per_group == 0)    # all K responses wrong
        mixed_groups = ~all_correct_groups & ~all_wrong_groups  # has both

        # Groups to keep: std > min_std (i.e., has variance)
        groups_to_keep = group_std > self.cfg.filter_groups_min_std

        num_kept_groups = int(groups_to_keep.sum().item())
        num_filtered_groups = num_prompts - num_kept_groups

        # Expand group mask to individual samples
        keep_mask = groups_to_keep.unsqueeze(1).expand(-1, K).reshape(-1)  # (B,)

        # Compute detailed statistics
        filter_stats = {
            "filter/num_groups_total": float(num_prompts),
            "filter/num_groups_kept": float(num_kept_groups),
            "filter/num_groups_filtered": float(num_filtered_groups),
            "filter/num_groups_all_correct": float(all_correct_groups.sum().item()),
            "filter/num_groups_all_wrong": float(all_wrong_groups.sum().item()),
            "filter/num_groups_mixed": float(mixed_groups.sum().item()),
            "filter/group_reward_mean": float(group_mean.mean().item()),
            "filter/group_reward_std_mean": float(group_std.mean().item()),
            "filter/pct_groups_kept": float(num_kept_groups / num_prompts * 100) if num_prompts > 0 else 0.0,
            "filter/pct_groups_all_wrong": float(all_wrong_groups.sum().item() / num_prompts * 100) if num_prompts > 0 else 0.0,
        }

        return keep_mask, num_kept_groups, num_filtered_groups, filter_stats

    def step(
        self,
        prompt_batch: List[str],
        ground_truths: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        One on-policy update step with DAPO-style group filtering and backfilling.

        When filter_groups_enable=True and num_return_sequences > 1:
        - Generate K responses per prompt
        - Compute rewards for all responses
        - Filter out groups (all K responses for a prompt) with reward std == 0
        - RETRY failed prompts up to max_filter_retries times before replacing
        - Backfill with NEW prompts only after exhausting retries
        - Only train on groups with variance (at least one correct, one incorrect)

        Args:
            prompt_batch: List of prompts
            ground_truths: Optional list of ground truth answers (for rule-based rewards)
            **kwargs: Additional kwargs passed to reward function

        Returns:
            metrics dict suitable for logging.
        """
        K = self.cfg.num_return_sequences
        target_num_prompts = len(prompt_batch)
        filter_enabled = self.cfg.filter_groups_enable and K > 1
        max_retries = self.cfg.max_filter_retries

        # Track filtering metrics
        total_gen_batches = 0
        total_filtered_groups = 0
        total_backfilled = 0
        total_retries = 0
        last_filter_stats = {}

        # Current prompts to process with their retry counts
        pending_prompts = []
        for i, p in enumerate(prompt_batch):
            gt = ground_truths[i] if ground_truths is not None else None
            pending_prompts.append({"prompt": p, "ground_truth": gt, "retries": 0})

        # Accumulated valid data across iterations
        valid_seqs_list = []
        valid_attn_list = []
        valid_prompt_lens_list = []
        valid_resp_mask_list = []
        valid_rewards_list = []
        valid_texts_list = []

        # Main loop - keep going until we have enough valid groups or run out of options
        max_total_iterations = max_retries * 3  # Safety limit
        for iteration in range(max_total_iterations if filter_enabled else 1):
            if not pending_prompts:
                break

            # Check if we have enough valid groups
            if len(valid_seqs_list) >= target_num_prompts:
                break

            total_gen_batches += 1
            num_current_prompts = len(pending_prompts)

            # Extract current batch
            current_prompts = [p["prompt"] for p in pending_prompts]
            current_ground_truths = [p["ground_truth"] for p in pending_prompts]
            if all(gt is None for gt in current_ground_truths):
                current_ground_truths = None

            # 1) Tokenize current prompts
            toks = self.tokenizer(
                current_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)
            prompt_ids = toks["input_ids"]
            prompt_attn = toks["attention_mask"]

            # 2) Generate responses
            seqs, attn, prompt_lens = self._generate(prompt_ids, prompt_attn)
            resp_mask = self._make_response_mask(seqs, prompt_lens) & (attn.bool())
            resp_lens = resp_mask.sum(dim=1)

            # Handle empty responses - retry all
            non_empty_mask = resp_lens > 0
            if not non_empty_mask.any():
                for p in pending_prompts:
                    p["retries"] += 1
                pending_prompts = [p for p in pending_prompts if p["retries"] < max_retries]
                if not pending_prompts and self._prompt_source is not None:
                    new_prompts, new_gt = self._sample_backfill_prompts(target_num_prompts - len(valid_seqs_list))
                    if new_prompts:
                        for i, np in enumerate(new_prompts):
                            ngt = new_gt[i] if new_gt else None
                            pending_prompts.append({"prompt": np, "ground_truth": ngt, "retries": 0})
                        total_backfilled += len(new_prompts)
                continue

            # Decode texts
            texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

            # Prepare expanded prompts and ground_truths for reward
            if K > 1:
                expanded_prompts = [p for p in current_prompts for _ in range(K)]
            else:
                expanded_prompts = current_prompts

            reward_kwargs = dict(kwargs)
            if current_ground_truths is not None:
                if K > 1:
                    expanded_gt = [gt for gt in current_ground_truths for _ in range(K)]
                    reward_kwargs["ground_truths"] = expanded_gt
                else:
                    reward_kwargs["ground_truths"] = current_ground_truths

            # 3) Compute rewards
            rewards = self._compute_rewards(
                texts,
                prompts=expanded_prompts,
                **reward_kwargs,
            )

            # 4) Group filtering
            if filter_enabled:
                _, num_kept_groups, num_filtered, filter_stats = self._filter_groups_by_reward_std(
                    rewards, K, num_current_prompts
                )
                total_filtered_groups += num_filtered
                last_filter_stats = filter_stats

                # Identify which groups are valid (have variance)
                rewards_grouped = rewards.view(num_current_prompts, K)
                group_std = rewards_grouped.std(dim=1, unbiased=False)
                valid_group_mask = group_std > self.cfg.filter_groups_min_std

                # Process each prompt
                next_pending = []
                for i in range(num_current_prompts):
                    if valid_group_mask[i]:
                        # Valid group - collect it
                        start_idx = i * K
                        end_idx = (i + 1) * K
                        valid_seqs_list.append(seqs[start_idx:end_idx])
                        valid_attn_list.append(attn[start_idx:end_idx])
                        valid_prompt_lens_list.append(prompt_lens[start_idx:end_idx])
                        valid_resp_mask_list.append(resp_mask[start_idx:end_idx])
                        valid_rewards_list.append(rewards[start_idx:end_idx])
                        valid_texts_list.extend(texts[start_idx:end_idx])
                    else:
                        # Invalid group - check if we should retry or replace
                        pending_prompts[i]["retries"] += 1
                        total_retries += 1
                        if pending_prompts[i]["retries"] < max_retries:
                            # Retry this prompt
                            next_pending.append(pending_prompts[i])
                        else:
                            # Exhausted retries - need to backfill
                            if self._prompt_source is not None:
                                new_prompts, new_gt = self._sample_backfill_prompts(1)
                                if new_prompts:
                                    ngt = new_gt[0] if new_gt else None
                                    next_pending.append({"prompt": new_prompts[0], "ground_truth": ngt, "retries": 0})
                                    total_backfilled += 1

                pending_prompts = next_pending

                # Log filtering status
                num_valid_collected = len(valid_seqs_list)
                print(
                    f"  [Filter] Iter {iteration+1}: "
                    f"tried={num_current_prompts}, valid={num_kept_groups}, "
                    f"collected={num_valid_collected}/{target_num_prompts}, "
                    f"pending={len(pending_prompts)}, "
                    f"retries={total_retries}, backfilled={total_backfilled}"
                )

                # Check if we have enough
                if num_valid_collected >= target_num_prompts:
                    break

                # If no pending prompts and not enough valid groups, try to sample more
                if not pending_prompts and num_valid_collected < target_num_prompts:
                    needed = target_num_prompts - num_valid_collected
                    if self._prompt_source is not None:
                        new_prompts, new_gt = self._sample_backfill_prompts(needed)
                        if new_prompts:
                            for i, np in enumerate(new_prompts):
                                ngt = new_gt[i] if new_gt else None
                                pending_prompts.append({"prompt": np, "ground_truth": ngt, "retries": 0})
                            total_backfilled += len(new_prompts)
                        else:
                            break
                    else:
                        break
            else:
                # No filtering, use all samples
                valid_seqs_list.append(seqs)
                valid_attn_list.append(attn)
                valid_prompt_lens_list.append(prompt_lens)
                valid_resp_mask_list.append(resp_mask)
                valid_rewards_list.append(rewards)
                valid_texts_list.extend(texts)
                break

        # Check if we have any valid data
        if not valid_seqs_list:
            return {
                "warn/no_valid_groups": 1.0,
                "skipped/empty_batch": 1.0,
                "filter/total_gen_batches": float(total_gen_batches),
                "filter/total_filtered_groups": float(total_filtered_groups),
                "filter/total_backfilled": float(total_backfilled),
                "filter/total_retries": float(total_retries),
            }

        # Pad sequences to same length before concatenating
        # (different generation batches may have different sequence lengths)
        max_seq_len = max(s.size(1) for s in valid_seqs_list)
        pad_id = self.tokenizer.pad_token_id or 0

        padded_seqs = []
        padded_attn = []
        padded_resp_mask = []
        for seq, att, rmask in zip(valid_seqs_list, valid_attn_list, valid_resp_mask_list):
            if seq.size(1) < max_seq_len:
                pad_len = max_seq_len - seq.size(1)
                seq = torch.nn.functional.pad(seq, (0, pad_len), value=pad_id)
                att = torch.nn.functional.pad(att, (0, pad_len), value=0)
                rmask = torch.nn.functional.pad(rmask, (0, pad_len), value=False)
            padded_seqs.append(seq)
            padded_attn.append(att)
            padded_resp_mask.append(rmask)

        # Concatenate all valid data
        seqs = torch.cat(padded_seqs, dim=0)
        attn = torch.cat(padded_attn, dim=0)
        prompt_lens = torch.cat(valid_prompt_lens_list, dim=0)
        resp_mask = torch.cat(padded_resp_mask, dim=0)
        rewards = torch.cat(valid_rewards_list, dim=0)
        texts = valid_texts_list

        # Update filter stats with final counts
        num_valid_groups = len(valid_seqs_list)
        last_filter_stats["filter/num_groups_kept"] = float(num_valid_groups)
        last_filter_stats["filter/total_backfilled"] = float(total_backfilled)
        last_filter_stats["filter/total_gen_batches"] = float(total_gen_batches)
        last_filter_stats["filter/total_retries"] = float(total_retries)

        all_same = len(set(texts)) == 1
        resp_lens = resp_mask.sum(dim=1)

        # 5-8) Compute log-probs, loss, and optimize
        with torch.enable_grad():
            with torch.amp.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.cfg.use_amp and self.device.type == "cuda",
            ):
                logp_pi = sequence_logprob_sum(self.policy, seqs, attn, resp_mask)
                with torch.no_grad():
                    logp_ref = sequence_logprob_sum(self.ref_policy, seqs, attn, resp_mask)

            # Compute weight
            w = self._compute_weight(rewards, logp_pi, logp_ref)

            # Build the pseudo-loss
            s = logp_pi
            s_raw = logp_pi.detach()
            resp_lens_f = resp_mask.sum(dim=1).clamp_min(1).float()

            if self.cfg.normalize_logprob_by_length:
                s = s / resp_lens_f

            s_clamped = s
            num_clamped = 0
            if self.cfg.clip_logprob_min is not None:
                s_clamped = torch.clamp(s, min=self.cfg.clip_logprob_min)
                num_clamped = int((s < self.cfg.clip_logprob_min).sum().item())

            loss = -(w * s_clamped + 0.5 * w.detach() * (s_clamped**2)).mean()

            # Optimize
            if self._accum_count == 0:
                self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self._accum_count += 1
            if self._accum_count >= self.grad_accum_steps:
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self._accum_count = 0

        # Metrics
        kl_est = logp_pi.detach() - logp_ref
        if self.cfg.length_normalize_kl:
            kl_per_tok = kl_est / resp_lens_f
            s_raw_per_tok = s_raw / resp_lens_f
        else:
            kl_per_tok = kl_est
            s_raw_per_tok = s_raw

        with torch.no_grad():
            # Compute reward statistics
            num_correct = int((rewards > 0).sum().item())
            num_incorrect = int((rewards < 0).sum().item())

            metrics = {
                "loss": float(loss.detach().cpu()),
                "reward/mean": float(rewards.mean().cpu()),
                "reward/std": float(rewards.std(unbiased=False).cpu()),
                "reward/num_correct": float(num_correct),
                "reward/num_incorrect": float(num_incorrect),
                "reward/accuracy": float(num_correct / len(rewards)) if len(rewards) > 0 else 0.0,
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
                "response/avg_len_tokens": float(resp_lens_f.mean().cpu()),
                "response/min_len_tokens": float(resp_lens_f.min().cpu()),
                "response/max_len_tokens": float(resp_lens_f.max().cpu()),
                "response/num_samples": float(len(rewards)),
            }

            if self.cfg.normalize_logprob_by_length:
                metrics["policy/logp_used_in_loss/mean"] = float(s.mean().cpu())
                metrics["policy/logp_used_in_loss/std"] = float(s.std(unbiased=False).cpu())
                metrics["policy/logp_used_in_loss/min"] = float(s.min().cpu())
                metrics["policy/logp_used_in_loss/max"] = float(s.max().cpu())
                metrics["config/normalized_by_length"] = 1.0

            if self.cfg.clip_logprob_min is not None and num_clamped > 0:
                metrics["clip/num_logprob_clamped"] = float(num_clamped)
                metrics["clip/pct_logprob_clamped"] = 100.0 * num_clamped / len(s)
                metrics["policy/logp_clamped/mean"] = float(s_clamped.mean().cpu())

            if all_same:
                metrics["warn/all_identical_responses"] = 1.0

            # Group filtering metrics
            if filter_enabled:
                metrics["filter/total_gen_batches"] = float(total_gen_batches)
                metrics["filter/total_backfilled"] = float(total_backfilled)
                metrics["filter/total_retries"] = float(total_retries)
                # Add all detailed filter stats
                metrics.update(last_filter_stats)

        return metrics
