# online_sft/distributed_trainer.py
# Distributed Online SFT Trainer with Accelerate FSDP and vLLM support
#
# Key optimizations:
# - FSDP: Shards model/optimizer/gradients across GPUs
# - vLLM: Optional high-throughput generation
# - Gradient accumulation with proper synchronization

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizerBase

try:
    from accelerate import Accelerator
    from accelerate.utils import gather_object

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    Accelerator = None

try:
    from vllm import LLM, SamplingParams

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None
    SamplingParams = None


def _shifted_log_probs(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-token log-probabilities using memory-efficient cross_entropy.
    logits: (B, T, V), labels: (B, T) with -100 = ignore.
    """
    B, T, V = logits.shape

    token_losses = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,  # explicitly ignore masked positions
        reduction="none",
    ).reshape(B, T)

    # loss = -log p, so log p = -loss
    token_logps = -token_losses
    return token_logps


def sequence_logprob_sum(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level log-probability sum over response tokens."""
    labels = input_ids.clone()
    labels[~response_mask] = -100
    labels[:, 0] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    token_logps = _shifted_log_probs(logits[:, :-1, :], labels[:, 1:])
    seq_logp = token_logps.sum(dim=-1)
    return seq_logp


def sequence_logprob_sum_chunked(
    model, input_ids, attention_mask, response_mask, chunk_size=8
):
    B = input_ids.size(0)
    outs = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        seq_logp = sequence_logprob_sum(
            model,
            input_ids[start:end],
            attention_mask[start:end],
            response_mask[start:end],
        )
        outs.append(seq_logp)
    return torch.cat(outs, dim=0)


@dataclass
class DistributedOnlineSFTConfig:
    """Configuration for distributed Online SFT training."""

    beta: float = 0.1

    # Generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    num_return_sequences: int = 8  # Multiple responses per prompt for variance

    # Group filtering (DAPO-style)
    filter_groups_enable: bool = True
    filter_groups_min_std: float = 0.0
    max_filter_retries: int = 10

    # Optimization
    lr: float = 1e-6
    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: Optional[float] = 1.0
    grad_accum_steps: int = 1

    # Stabilization
    denom_eps: float = 1e-8
    clip_weight_value: Optional[float] = 1.0
    clip_logprob_min: Optional[float] = -200.0
    normalize_logprob_by_length: bool = False

    # Reward
    reward_scale: float = 1.0
    reward_shift: float = 0.0

    # vLLM settings
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    vllm_dtype: str = "bfloat16"
    vllm_sync_interval: int = 1  # Sync weights every N steps (1 = every step)

    # Distributed
    sync_ref_model: bool = True  # Sync ref model with DDP


class DistributedOnlineSFTTrainer:
    """
    Distributed Online SFT Trainer with FSDP and vLLM support.

    Use with accelerate launch:
        accelerate launch --num_processes 4 --use_fsdp ... train.py
    """

    def __init__(
        self,
        policy: PreTrainedModel,
        ref_policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: "Accelerator",
        reward_fn: Optional[Callable] = None,
        reward_model: Optional[PreTrainedModel] = None,
        cfg: Optional[DistributedOnlineSFTConfig] = None,
    ):
        if cfg is None:
            cfg = DistributedOnlineSFTConfig()

        self.cfg = cfg
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set pad token id on model configs
        pad_id = self.tokenizer.pad_token_id
        for m in [policy, ref_policy]:
            if (
                getattr(getattr(m, "config", None), "pad_token_id", None)
                is None
            ):
                try:
                    m.config.pad_token_id = pad_id
                except Exception:
                    pass

        # Prepare models with accelerator (handles FSDP wrapping)
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model

        # Freeze reference policy
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # Freeze reward model if provided
        if self.reward_model is not None:
            self.reward_model.eval()
            for p in self.reward_model.parameters():
                p.requires_grad_(False)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.lr,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )

        # Prepare policy and optimizer with accelerator (FSDP wrapping)
        self.policy, self.optimizer = accelerator.prepare(
            self.policy, self.optimizer
        )

        # Also prepare ref_policy with accelerator for proper device placement and FSDP sharding
        # This is important for memory efficiency - ref_policy will be sharded across GPUs
        # Note: ref_policy is frozen (requires_grad=False) so it won't be trained
        self.ref_policy = accelerator.prepare(self.ref_policy)

        # Initialize vLLM if enabled
        self.vllm_engine = None
        if cfg.use_vllm and HAS_VLLM:
            self._init_vllm()

        self.scheduler = None
        self._accum_count = 0
        self._step_count = 0  # For vLLM weight sync interval

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
        self._prompt_source_iter = None

    def _sample_backfill_prompts(
        self, num_prompts: int
    ) -> Tuple[List[str], Optional[List[str]]]:
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

            indices = list(range(len(self._prompt_source)))
            random.shuffle(indices)
            self._prompt_source_indices = indices
            self._prompt_source_idx = 0

        prompts = []
        ground_truths = []
        has_ground_truths = False

        for _ in range(num_prompts):
            if self._prompt_source_idx >= len(self._prompt_source_indices):
                import random

                random.shuffle(self._prompt_source_indices)
                self._prompt_source_idx = 0

            idx = self._prompt_source_indices[self._prompt_source_idx]
            self._prompt_source_idx += 1

            item = self._prompt_source[idx]

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

    def _init_vllm(self):
        """Initialize vLLM engine for fast generation in colocate mode.

        Uses vLLM's external_launcher backend so each rank creates its own vLLM
        engine on its assigned GPU. This requires RANK, LOCAL_RANK, WORLD_SIZE
        environment variables to be set (accelerate launch sets these automatically).
        """
        if not HAS_VLLM:
            self.accelerator.print(
                "[vLLM] vLLM not installed, falling back to HF generate"
            )
            return

        self.accelerator.print(
            f"[vLLM] Initializing colocate mode on rank {self.accelerator.process_index}"
        )

        # Only set env vars if not already set (accelerate launch should have set them)
        # Setting them after FSDP init can corrupt FSDP state
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(self.accelerator.process_index)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)

        # Ensure MASTER_ADDR and MASTER_PORT are set (accelerate should set these too)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            # Find a free port (only on rank 0, then broadcast)
            import socket

            if self.accelerator.is_main_process:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    port = s.getsockname()[1]
            else:
                port = None
            # Broadcast port to all processes
            port_list = [port]
            torch.distributed.broadcast_object_list(port_list, src=0)
            os.environ["MASTER_PORT"] = str(port_list[0])

        try:
            unwrapped = self.accelerator.unwrap_model(self.policy)
            model_name = getattr(unwrapped.config, "_name_or_path", None)

            if model_name:
                # Use external_launcher backend so vLLM uses the existing distributed setup
                # Each rank creates its own vLLM engine on its assigned GPU
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=self.cfg.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.cfg.vllm_gpu_memory_utilization,
                    dtype=self.cfg.vllm_dtype,
                    trust_remote_code=True,
                    enforce_eager=True,  # Required for weight syncing
                    distributed_executor_backend="external_launcher",  # Use existing distributed group
                    # Use same seed within TP groups to ensure consistent sampling
                    seed=self.accelerator.process_index
                    // self.cfg.vllm_tensor_parallel_size,
                )
                self.accelerator.print(
                    f"[vLLM] Engine initialized on rank {self.accelerator.process_index}"
                )

                # Sync weights from policy to vLLM at initialization
                self._sync_vllm_weights()
            else:
                self.accelerator.print(
                    "[vLLM] Could not determine model name, using HF generate"
                )
        except Exception as e:
            self.accelerator.print(
                f"[vLLM] Initialization failed: {e}, using HF generate"
            )
            import traceback

            traceback.print_exc()
            self.vllm_engine = None

    def _sync_vllm_weights(self):
        """Sync weights from the trained policy to vLLM engine."""
        if self.vllm_engine is None:
            return

        try:
            # For FSDP, we need to gather the full state dict
            # IMPORTANT: Use the wrapped model (self.policy), not unwrapped
            # Passing unwrapped model corrupts FSDP's internal state
            if hasattr(self.accelerator, "get_state_dict"):
                # Use accelerator's method if available (handles FSDP)
                state_dict = self.accelerator.get_state_dict(self.policy)
            else:
                # Fallback: unwrap and get state dict (non-FSDP case)
                unwrapped = self.accelerator.unwrap_model(self.policy)
                state_dict = unwrapped.state_dict()

            if state_dict is None:
                self.accelerator.print(
                    "[vLLM] Warning: Could not get state dict for weight sync"
                )
                return

            # Get vLLM's model and load the weights
            # vLLM stores model in: llm_engine.model_executor.driver_worker.model_runner.model
            try:
                vllm_model = (
                    self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                )
                vllm_model.load_weights(state_dict.items())
                self.accelerator.print("[vLLM] Weights synced successfully")
            except AttributeError:
                # Try alternative path for different vLLM versions
                try:
                    # For vLLM >= 0.4.0
                    model_runner = (
                        self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner
                    )
                    if hasattr(model_runner, "model"):
                        model_runner.model.load_weights(state_dict.items())
                        self.accelerator.print(
                            "[vLLM] Weights synced successfully (v2)"
                        )
                except Exception as e2:
                    self.accelerator.print(f"[vLLM] Weight sync failed: {e2}")

        except Exception as e:
            self.accelerator.print(f"[vLLM] Weight sync error: {e}")

    @torch.no_grad()
    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """Generate using vLLM engine."""
        sampling_params = SamplingParams(
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k if self.cfg.top_k else -1,
            n=self.cfg.num_return_sequences,
        )

        outputs = self.vllm_engine.generate(prompts, sampling_params)

        # Collect generated texts
        generated_texts = []
        for output in outputs:
            for completion in output.outputs:
                # Full text = prompt + completion
                generated_texts.append(output.prompt + completion.text)

        return generated_texts

    @torch.no_grad()
    def _generate_hf(
        self, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate using HuggingFace generate with FSDP support."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

        # For FSDP FULL_SHARD, parameters are sharded across GPUs.
        # generate() doesn't work with sharded params, so we need to
        # temporarily gather full parameters using summon_full_params().
        # This increases memory usage during generation but is required.
        #
        # Check if model is FSDP-wrapped (may be nested in Accelerate wrappers)
        is_fsdp = isinstance(self.policy, FSDP) or (
            hasattr(self.policy, "module")
            and isinstance(self.policy.module, FSDP)
        )

        if is_fsdp:
            # Get the actual FSDP module
            fsdp_model = (
                self.policy
                if isinstance(self.policy, FSDP)
                else self.policy.module
            )
            with FSDP.summon_full_params(
                fsdp_model, writeback=False, recurse=True
            ):
                seqs = self.policy.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attn,
                    **gen_kwargs,
                )
        else:
            seqs = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attn,
                **gen_kwargs,
            )

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
    def _compute_rewards(
        self,
        texts: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute rewards using reward function or model."""
        if self.reward_fn is not None:
            rewards = self.reward_fn(
                completions=texts,
                prompts=prompts or [""] * len(texts),
                **kwargs,
            )
            device = self.accelerator.device
            return torch.tensor(rewards, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                "Either reward_fn or reward_model must be provided"
            )

    def _compute_weight(
        self,
        reward: torch.Tensor,
        logp_pi: torch.Tensor,
        logp_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weight w_theta(x,y)."""
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
        group_mean = rewards_grouped.mean(dim=1)
        group_std = rewards_grouped.std(dim=1, unbiased=False)

        # Categorize groups
        num_correct_per_group = (rewards_grouped > 0).sum(dim=1)
        all_correct_groups = num_correct_per_group == K
        all_wrong_groups = num_correct_per_group == 0
        mixed_groups = ~all_correct_groups & ~all_wrong_groups

        # Groups to keep: std > min_std (i.e., has variance)
        groups_to_keep = group_std > self.cfg.filter_groups_min_std

        num_kept_groups = int(groups_to_keep.sum().item())
        num_filtered_groups = num_prompts - num_kept_groups

        # Expand group mask to individual samples
        keep_mask = groups_to_keep.unsqueeze(1).expand(-1, K).reshape(-1)

        # Compute detailed statistics
        filter_stats = {
            "filter/num_groups_total": float(num_prompts),
            "filter/num_groups_kept": float(num_kept_groups),
            "filter/num_groups_filtered": float(num_filtered_groups),
            "filter/num_groups_all_correct": float(
                all_correct_groups.sum().item()
            ),
            "filter/num_groups_all_wrong": float(all_wrong_groups.sum().item()),
            "filter/num_groups_mixed": float(mixed_groups.sum().item()),
            "filter/group_reward_mean": float(group_mean.mean().item()),
            "filter/group_reward_std_mean": float(group_std.mean().item()),
            "filter/pct_groups_kept": (
                float(num_kept_groups / num_prompts * 100)
                if num_prompts > 0
                else 0.0
            ),
            "filter/pct_groups_all_wrong": (
                float(all_wrong_groups.sum().item() / num_prompts * 100)
                if num_prompts > 0
                else 0.0
            ),
        }

        return keep_mask, num_kept_groups, num_filtered_groups, filter_stats

    def step(
        self,
        prompt_batch: List[str],
        ground_truths: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        One distributed on-policy update step with DAPO-style group filtering and backfilling.

        When filter_groups_enable=True and num_return_sequences > 1:
        - Generate K responses per prompt
        - Compute rewards for all responses
        - Filter out groups (all K responses for a prompt) with reward std == 0
        - RETRY failed prompts up to max_filter_retries times before replacing
        - Backfill with NEW prompts only after exhausting retries
        - Only train on groups with variance (at least one correct, one incorrect)

        Args:
            prompt_batch: List of prompts
            ground_truths: Optional list of ground truth answers
            **kwargs: Additional kwargs for reward function

        Returns:
            Metrics dict
        """
        device = self.accelerator.device
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
        # Each entry: (prompt, ground_truth, retry_count)
        pending_prompts = []
        for i, p in enumerate(prompt_batch):
            gt = ground_truths[i] if ground_truths is not None else None
            pending_prompts.append(
                {"prompt": p, "ground_truth": gt, "retries": 0}
            )

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
                max_length=2048,
            ).to(device)
            prompt_ids = toks["input_ids"]
            prompt_attn = toks["attention_mask"]

            # 2) Generate responses
            if self.vllm_engine is not None and self.cfg.use_vllm:
                if self._step_count % self.cfg.vllm_sync_interval == 0:
                    self._sync_vllm_weights()
                generated_texts = self._generate_vllm(current_prompts)
                full_toks = self.tokenizer(
                    generated_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048 + self.cfg.max_new_tokens,
                ).to(device)
                seqs = full_toks["input_ids"]
                attn = full_toks["attention_mask"]
                prompt_lens = []
                for p in current_prompts:
                    p_len = len(
                        self.tokenizer(p, add_special_tokens=True)["input_ids"]
                    )
                    prompt_lens.extend([p_len] * K)
                prompt_lens = torch.tensor(prompt_lens, device=device)
            else:
                seqs, attn, prompt_lens = self._generate_hf(
                    prompt_ids, prompt_attn
                )
                generated_texts = self.tokenizer.batch_decode(
                    seqs, skip_special_tokens=True
                )

            resp_mask = self._make_response_mask(seqs, prompt_lens) & (
                attn.bool()
            )
            resp_lens = resp_mask.sum(dim=1)

            # Handle empty responses - retry all
            non_empty_mask = resp_lens > 0
            if not non_empty_mask.any():
                # Increment retry counts for all
                for p in pending_prompts:
                    p["retries"] += 1
                # Remove prompts that exceeded max retries
                pending_prompts = [
                    p for p in pending_prompts if p["retries"] < max_retries
                ]
                if not pending_prompts and self._prompt_source is not None:
                    # All prompts exhausted, sample new ones
                    new_prompts, new_gt = self._sample_backfill_prompts(
                        target_num_prompts - len(valid_seqs_list)
                    )
                    if new_prompts:
                        for i, np in enumerate(new_prompts):
                            ngt = new_gt[i] if new_gt else None
                            pending_prompts.append(
                                {
                                    "prompt": np,
                                    "ground_truth": ngt,
                                    "retries": 0,
                                }
                            )
                        total_backfilled += len(new_prompts)
                continue

            texts = generated_texts

            # Prepare expanded prompts and ground_truths for reward
            if K > 1:
                expanded_prompts = [
                    p for p in current_prompts for _ in range(K)
                ]
            else:
                expanded_prompts = current_prompts

            reward_kwargs = dict(kwargs)
            if current_ground_truths is not None:
                if K > 1:
                    expanded_gt = [
                        gt for gt in current_ground_truths for _ in range(K)
                    ]
                    reward_kwargs["ground_truths"] = expanded_gt
                else:
                    reward_kwargs["ground_truths"] = current_ground_truths

            # 3) Compute rewards
            rewards = self._compute_rewards(
                texts, prompts=expanded_prompts, **reward_kwargs
            )

            # 4) Group filtering
            if filter_enabled:
                _, num_kept_groups, num_filtered, filter_stats = (
                    self._filter_groups_by_reward_std(
                        rewards, K, num_current_prompts
                    )
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
                        valid_prompt_lens_list.append(
                            prompt_lens[start_idx:end_idx]
                        )
                        valid_resp_mask_list.append(
                            resp_mask[start_idx:end_idx]
                        )
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
                                new_prompts, new_gt = (
                                    self._sample_backfill_prompts(1)
                                )
                                if new_prompts:
                                    ngt = new_gt[0] if new_gt else None
                                    next_pending.append(
                                        {
                                            "prompt": new_prompts[0],
                                            "ground_truth": ngt,
                                            "retries": 0,
                                        }
                                    )
                                    total_backfilled += 1

                pending_prompts = next_pending

                # Log filtering status
                num_valid_collected = len(valid_seqs_list)
                self.accelerator.print(
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
                if (
                    not pending_prompts
                    and num_valid_collected < target_num_prompts
                ):
                    needed = target_num_prompts - num_valid_collected
                    if self._prompt_source is not None:
                        new_prompts, new_gt = self._sample_backfill_prompts(
                            needed
                        )
                        if new_prompts:
                            for i, np in enumerate(new_prompts):
                                ngt = new_gt[i] if new_gt else None
                                pending_prompts.append(
                                    {
                                        "prompt": np,
                                        "ground_truth": ngt,
                                        "retries": 0,
                                    }
                                )
                            total_backfilled += len(new_prompts)
                        else:
                            break  # No more prompts available
                    else:
                        break  # No prompt source
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
        for seq, att, rmask in zip(
            valid_seqs_list, valid_attn_list, valid_resp_mask_list
        ):
            if seq.size(1) < max_seq_len:
                pad_len = max_seq_len - seq.size(1)
                seq = torch.nn.functional.pad(seq, (0, pad_len), value=pad_id)
                att = torch.nn.functional.pad(att, (0, pad_len), value=0)
                rmask = torch.nn.functional.pad(
                    rmask, (0, pad_len), value=False
                )
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

        # Update filter stats
        num_valid_groups = len(valid_seqs_list)
        last_filter_stats["filter/num_groups_kept"] = float(num_valid_groups)
        last_filter_stats["filter/total_backfilled"] = float(total_backfilled)
        last_filter_stats["filter/total_gen_batches"] = float(total_gen_batches)
        last_filter_stats["filter/total_retries"] = float(total_retries)

        all_same = len(set(texts)) == 1
        resp_lens = resp_mask.sum(dim=1)

        # 5-8) Compute log-probs, loss, and optimize
        self.policy.train()
        logp_pi = sequence_logprob_sum(self.policy, seqs, attn, resp_mask)

        with torch.no_grad():
            logp_ref = sequence_logprob_sum_chunked(
                self.ref_policy, seqs, attn, resp_mask
            )

        w = self._compute_weight(rewards, logp_pi, logp_ref)

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

        if self._accum_count == 0:
            self.optimizer.zero_grad(set_to_none=True)

        self.accelerator.backward(loss)
        self._accum_count += 1

        if self._accum_count >= self.cfg.grad_accum_steps:
            if self.cfg.grad_clip_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.grad_clip_norm
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self._accum_count = 0

        # Metrics
        kl_est = logp_pi.detach() - logp_ref
        kl_per_tok = kl_est / resp_lens_f

        with torch.no_grad():
            num_correct = int((rewards > 0).sum().item())
            num_incorrect = int((rewards < 0).sum().item())

            metrics = {
                "loss": float(loss.detach().cpu()),
                # Reward metrics
                "reward/mean": float(rewards.mean().cpu()),
                "reward/std": float(rewards.std(unbiased=False).cpu()),
                "reward/num_correct": float(num_correct),
                "reward/num_incorrect": float(num_incorrect),
                "reward/accuracy": (
                    float(num_correct / len(rewards))
                    if len(rewards) > 0
                    else 0.0
                ),
                # Policy metrics
                "policy/logp_seq_raw/mean": float(s_raw.mean().cpu()),
                "policy/logp_seq_raw/std": float(
                    s_raw.std(unbiased=False).cpu()
                ),
                "policy/logp_seq_raw/min": float(s_raw.min().cpu()),
                "policy/logp_seq_raw/max": float(s_raw.max().cpu()),
                "policy/logp_seq_per_tok/mean": float(
                    (s_raw / resp_lens_f).mean().cpu()
                ),
                "policy/kl_y_ref/mean": float(kl_est.mean().cpu()),
                "policy/kl_y_ref_per_tok/mean": float(kl_per_tok.mean().cpu()),
                # Weight metrics
                "weight/mean": float(w.mean().detach().cpu()),
                "weight/std": float(w.std(unbiased=False).detach().cpu()),
                "weight/min": float(w.min().detach().cpu()),
                "weight/max": float(w.max().detach().cpu()),
                # Response metrics
                "response/avg_len_tokens": float(resp_lens_f.mean().cpu()),
                "response/min_len_tokens": float(resp_lens_f.min().cpu()),
                "response/max_len_tokens": float(resp_lens_f.max().cpu()),
                "response/num_samples": float(len(rewards)),
            }

            if num_clamped > 0:
                metrics["clip/num_logprob_clamped"] = float(num_clamped)
                metrics["clip/pct_logprob_clamped"] = (
                    100.0 * num_clamped / len(s)
                )
            if all_same:
                metrics["warn/all_identical_responses"] = 1.0

            if filter_enabled:
                metrics["filter/total_gen_batches"] = float(total_gen_batches)
                metrics["filter/total_backfilled"] = float(total_backfilled)
                metrics["filter/total_retries"] = float(total_retries)
                metrics.update(last_filter_stats)

        # Increment step counter for vLLM weight sync
        self._step_count += 1

        return metrics


def create_accelerator(
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    log_with: Optional[str] = "wandb",
    project_name: str = "online-sft",
) -> "Accelerator":
    """
    Create an Accelerator instance for distributed training.

    Use with: accelerate launch --use_fsdp ...
    """
    if not HAS_ACCELERATE:
        raise ImportError("accelerate is required for distributed training")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_config={"project_name": project_name} if log_with else None,
    )
    return accelerator
