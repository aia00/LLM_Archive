"""
Dynamic/filtered GRPO trainer that mirrors verl's group filtering and dual-clip PPO.

Features:
1. Group filtering: Drop groups whose rewards have zero std across the n responses
   (`num_generations`) to avoid collapsed batches. This is a lightweight,
   single-process approximation of verl's dynamic sampling loop.
2. Dual-clip PPO: Apply a lower bound (clip_ratio_c) for negative advantages,
   matching verl's compute_policy_loss_vanilla implementation.
3. Entropy regularization: Optional entropy bonus to encourage exploration.
4. Token-level loss normalization (DAPO mode): Normalize by total tokens instead
   of per-sequence averaging.

If all groups would be dropped, we keep the full batch to avoid stalls.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist

from trl.trainer.grpo_trainer import (
    GRPOTrainer,
    apply_chat_template,
    gather_object,
    is_conversational,
    nanmax,
    nanmin,
    nanstd,
    pad,
    prepare_multimodal_messages,
)

# Support both package and script execution.
#
# - When running as a package module (e.g., `python -m dapo_trl.train_dapo`),
#   the relative import works.
# - When executing `train_dapo.py` as a standalone script (as in run_dapo.sh),
#   the package context is missing; fall back to absolute import.
try:
    from .reward_function import compute_score
except ImportError:  # pragma: no cover - script mode fallback
    from reward_function import compute_score


class DynamicGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        rollout_func: Optional[Callable] = None,
        *,
        filter_groups_enable: bool = False,
        filter_groups_metric: str = "acc",
        max_filter_gen_batches: int = 0,
        entropy_coeff: float = 0.0,
        clip_ratio_c: float = 10.0,
        norm_adv_by_std_in_grpo: bool = True,
    ):
        self.filter_groups_enable = filter_groups_enable
        self.filter_groups_metric = filter_groups_metric
        self.max_filter_gen_batches = (
            max_filter_gen_batches  # placeholder for future resampling logic
        )
        self.entropy_coeff = entropy_coeff
        self.clip_ratio_c = clip_ratio_c  # Dual-clip PPO lower bound
        self._filter_progress_last_kept: Optional[int] = None
        self.norm_adv_by_std_in_grpo = norm_adv_by_std_in_grpo

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
        )
        if getattr(self, "scale_rewards", None) == "batch":
            if self.accelerator.is_local_main_process:
                self.accelerator.print(
                    "[grpo] overriding scale_rewards='batch' to 'group' for DAPO parity with verl"
                )
            self.scale_rewards = "group"
        # Move reward weights once to avoid per-step device copies
        if hasattr(self, "reward_weights"):
            self.reward_weights = self.reward_weights.to(
                self.accelerator.device
            )

    def _sample_backfill_inputs(self, num_needed: int) -> list[dict[str, Any]]:
        """
        Quickly sample extra prompts from the training set to backfill filtered-out groups.
        Uses random indexing to avoid another DataLoader and to keep overhead low.
        """
        if num_needed <= 0:
            return []
        dataset = getattr(self, "train_dataset", None)
        if dataset is None:
            return []
        try:
            length = len(dataset)
        except Exception:
            return []
        if length == 0:
            return []

        indices = torch.randint(
            low=0, high=length, size=(num_needed,), device="cpu"
        ).tolist()
        return [dataset[int(i)] for i in indices]

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Copied from TRL's GRPOTrainer with an added filtering step that drops
        groups whose rewards have zero std (dynamic sampling / group filtering).
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        filter_enabled = self.filter_groups_enable and mode == "train"

        # ---------------------------------------------------------------------
        # Stable target group count (DAPO-style)
        # ---------------------------------------------------------------------
        # In training, we always optimize on a *fixed* number of prompt groups
        # per device (per_device_train_batch_size). This prevents per-rank batch
        # size drift (e.g., last batch, iterable datasets) which can otherwise
        # lead to mismatched collective calls and NCCL timeouts, and it keeps the
        # effective optimization batch size stable across steps.
        if mode == "train":
            target_groups = int(
                getattr(self.args, "per_device_train_batch_size", 1)
            )
            target_groups = max(1, target_groups)

            # Normalize the incoming `inputs` to exactly `target_groups` prompt
            # groups, then expand each group to `num_generations` items.
            def _collapse_to_groups(
                examples: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                if not examples:
                    return []
                if len(examples) % self.num_generations == 0:
                    # Assume prompts are repeated contiguously in blocks of `num_generations`.
                    return [
                        examples[i]
                        for i in range(0, len(examples), self.num_generations)
                    ]
                return list(examples)

            def _expand_groups_to_completions(
                group_examples: list[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                for item in group_examples:
                    for _ in range(self.num_generations):
                        out.append(item.copy())
                return out

            group_inputs = _collapse_to_groups(inputs)

            if len(group_inputs) > target_groups:
                group_inputs = group_inputs[:target_groups]
            elif len(group_inputs) < target_groups:
                need = target_groups - len(group_inputs)
                backfill = self._sample_backfill_inputs(need)
                if not backfill:
                    # As a last resort, repeat the last group to keep shapes consistent.
                    if group_inputs:
                        backfill = [group_inputs[-1]] * need
                    else:
                        backfill = []
                group_inputs = group_inputs + [
                    b.copy() for b in backfill[:need]
                ]

            # If we're still short (should be rare), repeat the last group.
            if len(group_inputs) < target_groups and group_inputs:
                need = target_groups - len(group_inputs)
                group_inputs = group_inputs + [
                    group_inputs[-1].copy() for _ in range(need)
                ]

            inputs = _expand_groups_to_completions(group_inputs)
        else:
            # Eval path: don't resample; just ensure divisibility for group metrics.
            if len(inputs) % self.num_generations != 0:
                expanded_inputs: list[dict[str, Any]] = []
                for item in inputs:
                    for _ in range(self.num_generations):
                        expanded_inputs.append(item.copy())
                inputs = expanded_inputs

            target_groups = max(1, len(inputs) // self.num_generations)

        attempt = 0
        keep_group_mask = None
        rewards_matrix = None
        accum = None
        filtered = False
        # How many *valid* prompt groups (kept by the filter) we have accumulated so far.
        valid_groups_accumulated = 0
        # The group indices (within the most recent generation batch) that were appended to `accum`.
        # Used to avoid duplicating groups when we need to top up with unfiltered data after retries.
        last_selected_groups = None
        # Informative hint before entering the filtering loop (stdout only, keeps wandb clean)
        if filter_enabled and self.accelerator.is_local_main_process:
            max_retries = (
                "inf"
                if self.max_filter_gen_batches <= 0
                else self.max_filter_gen_batches
            )
            self.accelerator.print(
                f"[filter] target={target_groups} groups, gen={self.num_generations}, max_retries={max_retries}"
            )
        while True:
            # Ensure each prompt is expanded to num_generations when the incoming batch
            # size is smaller than expected (e.g., backfill batches).
            if len(inputs) % self.num_generations != 0:
                expanded_inputs: list[dict[str, torch.Tensor | Any]] = []
                for item in inputs:
                    for _ in range(self.num_generations):
                        expanded_inputs.append(item.copy())
                inputs = expanded_inputs

            current_inputs = inputs
            prompts = [x["prompt"] for x in current_inputs]
            total_completions = len(prompts)

            if "images" in current_inputs[0]:
                images = [example.get("images") for example in current_inputs]
            elif "image" in current_inputs[0]:
                images = [
                    (
                        [example.get("image")]
                        if example.get("image") is not None
                        else None
                    )
                    for example in current_inputs
                ]
            else:
                images = None
            # Transformers requires at least one image in the batch, otherwise it throws an error
            if images is not None and all(
                img_list == [] for img_list in images
            ):
                images = None

            # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
            # [{"role": "user", "content": "What color is the sky?"}] to
            # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
            if images is not None:
                prompts = [
                    prepare_multimodal_messages(prompt, image_list)
                    for prompt, image_list in zip(prompts, images, strict=True)
                ]

            generated = self._generate(prompts)
            if len(generated) == 5:
                (
                    prompt_ids_list,
                    completion_ids_list,
                    num_items_in_batch,
                    sampling_per_token_logps_list,
                    extra_fields,
                ) = generated
            else:
                prompt_ids_list = generated[0]
                completion_ids_list = generated[1]
                # TRL returns (prompt_ids, completion_ids, tool_mask, completions, total_tokens, logprobs, extra_fields)
                num_items_in_batch = generated[4] if len(generated) > 4 else None
                sampling_per_token_logps_list = (
                    generated[5] if len(generated) > 5 else None
                )
                extra_fields = generated[6] if len(generated) > 6 else None

            # Convert lists of token IDs to padded tensors
            prompt_ids = [
                torch.tensor(ids, device=device) for ids in prompt_ids_list
            ]
            prompt_mask = [
                torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids
            ]
            prompt_ids = pad(
                prompt_ids, padding_value=self.pad_token_id, padding_side="left"
            )
            prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids_list
            ]
            completion_mask = [
                torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids,
                padding_value=self.pad_token_id,
                padding_side="right",
            )
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right"
            )
            if sampling_per_token_logps_list is not None:
                sampling_per_token_logps = [
                    torch.tensor(logps, device=device)
                    for logps in sampling_per_token_logps_list
                ]
                sampling_per_token_logps = pad(
                    sampling_per_token_logps,
                    padding_value=0.0,
                    padding_side="right",
                )
            else:
                sampling_per_token_logps = None

            # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
            if self.mask_truncated_completions:
                eos_and_pad = [self.eos_token_id, self.pad_token_id]
                is_truncated = torch.tensor(
                    [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                    device=device,
                )
                completion_mask = (
                    completion_mask * (~is_truncated).unsqueeze(1).int()
                )

            # Concatenate prompt_mask with completion_mask for logit computation
            prompt_completion_ids = torch.cat(
                [prompt_ids, completion_ids], dim=1
            )  # (B, P+C)
            attention_mask = torch.cat(
                [prompt_mask, completion_mask], dim=1
            )  # (B, P+C)

            logits_to_keep = completion_ids.size(
                1
            )  # we only need to compute the logits for the completion tokens
            batch_size = (
                self.args.per_device_train_batch_size
                if mode == "train"
                else self.args.per_device_eval_batch_size
            )

            num_images = (
                [len(img_list) for img_list in images]
                if images is not None
                else None
            )

            # Get forward_kwargs for models with multimodal inputs
            if images is not None:
                prompts_text = [
                    apply_chat_template(
                        {"prompt": prompt},
                        self.processing_class,
                        **self.chat_template_kwargs,
                    )["prompt"]
                    for prompt in prompts
                ]
                prompt_inputs = self.processing_class(
                    images=images,
                    text=prompts_text,
                    padding=True,
                    return_tensors="pt",
                )
                prompt_inputs = super()._prepare_inputs(prompt_inputs)
                forward_kwargs = {
                    k: v
                    for k, v in prompt_inputs.items()
                    if k not in ["input_ids", "attention_mask"]
                }
            else:
                forward_kwargs = {}

            # If token_type_ids are used, extend them with zeros for the completion part
            if "token_type_ids" in forward_kwargs:
                token_type_ids = forward_kwargs["token_type_ids"]
                forward_kwargs["token_type_ids"] = torch.cat(
                    [
                        token_type_ids,
                        token_type_ids.new_zeros(completion_ids.shape),
                    ],
                    dim=1,
                )

            with torch.no_grad():
                # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
                # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
                # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
                # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
                # old_per_token_logps to None.
                # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
                # distribution mismatch between vLLM and the training model can be large and harm the training.
                generate_every = (
                    self.args.steps_per_generation * self.num_iterations
                )  # generation frequency
                if (
                    self.args.gradient_accumulation_steps % generate_every != 0
                    or (
                        self.use_vllm
                        and self.vllm_importance_sampling_correction
                    )
                ):
                    old_per_token_logps, _ = (
                        self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
                    )
                else:
                    old_per_token_logps = None

                importance_sampling_ratio = None
                # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
                if self.use_vllm and self.vllm_importance_sampling_correction:
                    importance_sampling_ratio = torch.exp(
                        old_per_token_logps - sampling_per_token_logps
                    )
                    importance_sampling_ratio = torch.clamp(
                        importance_sampling_ratio,
                        max=self.vllm_importance_sampling_cap,
                    )

                # Compute the per-token log probabilities for the reference model
                if self.beta != 0.0:
                    if self.ref_model is not None:
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.ref_model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                            )
                        )
                    else:
                        with self.accelerator.unwrap_model(
                            self.model
                        ).disable_adapter():
                            ref_per_token_logps, _ = (
                                self._get_per_token_logps_and_entropies(
                                    self.model,
                                    prompt_completion_ids,
                                    attention_mask,
                                    logits_to_keep,
                                    batch_size=batch_size,
                                    num_images=num_images,
                                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                                )
                            )
                else:
                    ref_per_token_logps = None

            # Decode
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            if is_conversational(inputs[0]):
                # IMPORTANT: do not mutate prompts in-place (e.g. via pop()).
                # Some dataloaders / samplers may reuse prompt objects across
                # the expanded batch, and in-place mutation can corrupt other
                # samples.
                completions = []
                prompts_for_rewards = []
                for prompt, completion in zip(
                    prompts, completions_text, strict=True
                ):
                    # If the prompt already ends with an assistant message,
                    # treat it as bootstrap text for the completion.
                    if prompt and prompt[-1].get("role") == "assistant":
                        bootstrap = prompt[-1].get("content", "")
                        prompt_wo_bootstrap = prompt[:-1]
                    else:
                        bootstrap = ""
                        prompt_wo_bootstrap = prompt

                    prompts_for_rewards.append(prompt_wo_bootstrap)
                    completions.append(
                        [
                            {
                                "role": "assistant",
                                "content": bootstrap + completion,
                            }
                        ]
                    )

                # Use the non-mutated, bootstrap-stripped prompts for reward
                # calculation (mirrors the intent of the previous pop()-based
                # implementation).
                prompts = prompts_for_rewards
            else:
                completions = completions_text

            # Per-completion accuracy (used for filtering and eval pass rate)
            acc_vals = []
            for comp_text, inp in zip(
                completions_text, current_inputs, strict=True
            ):
                gt = inp.get("ground_truth", "")
                acc = compute_score(comp_text, gt).get("acc", False)
                acc_vals.append(1.0 if acc else 0.0)
            acc_tensor = torch.tensor(
                acc_vals, device=device, dtype=torch.float32
            )

            # Merge extra_fields from rollout_func into inputs for reward functions
            if extra_fields:
                for i, inp in enumerate(inputs):
                    for key, values in extra_fields.items():
                        if isinstance(values, list) and i < len(values):
                            inp[key] = values[i]
                        elif not isinstance(values, list):
                            inp[key] = values

            # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
            # important because rewards will be normalized per group, and completions are distributed. We will later slice
            # rewards_per_func to extract each process's subset.
            rewards_per_func_all = self._calculate_rewards(
                current_inputs, prompts, completions, completion_ids_list
            )

            # rewards_per_func_all is gathered across processes; slice out the local completions
            # NOTE: len(prompts) already accounts for num_generations because prompts are
            # duplicated by the sampler, so we must not multiply by num_generations again.
            local_batch_size = len(prompts)
            local_sample_size = local_batch_size
            start = self.accelerator.process_index * local_sample_size
            end = start + local_sample_size
            if rewards_per_func_all.shape[0] < end:
                if self.accelerator.is_local_main_process:
                    self.accelerator.print(
                        f"[filter_groups] gathered rewards rows ({rewards_per_func_all.shape[0]}) "
                        f"smaller than expected slice end ({end}); using local rewards only."
                    )
                rewards_per_func = rewards_per_func_all[:local_sample_size]
            else:
                rewards_per_func = rewards_per_func_all[start:end]

            # Apply weights to each reward function's output and sum
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)

            # Build filtering metric tensor (supports reward or acc)
            filter_metric_tensor = rewards
            if (
                self.filter_groups_metric
                and self.filter_groups_metric.lower() == "acc"
            ):
                filter_metric_tensor = acc_tensor

            # === Group filtering (dynamic sampling) ===
            filtered = False

            # Calculate how many groups this rank has already accumulated
            local_already_kept = (
                sum(chunk.shape[0] for chunk in accum["rewards"])
                // self.num_generations
                if accum and "rewards" in accum and accum["rewards"]
                else 0
            )

            # Rank identifier for multi-GPU logging
            rank_id = (
                self.accelerator.process_index
                if dist.is_available() and dist.is_initialized()
                else 0
            )
            num_ranks = (
                self.accelerator.num_processes
                if dist.is_available() and dist.is_initialized()
                else 1
            )

            # Determine group size for filtering.
            #
            # We *expect* the batch to be divisible by num_generations because
            # we expand inputs above. However, in edge-cases (eval overrides,
            # batch sync truncation, etc.) this may not hold. Using a group_size
            # that does not evenly divide the batch would make `.view(-1, group_size)`
            # throw. In that case, fall back to group_size=1 (i.e., keep all groups).
            if filter_metric_tensor.numel() % self.num_generations == 0:
                group_size = self.num_generations
            else:
                group_size = 1

            rewards_matrix = filter_metric_tensor.view(-1, group_size)

            if filter_enabled:
                filtered = True
                group_std = rewards_matrix.std(dim=1)
                keep_group_mask = (group_std > 0) | (
                    rewards_matrix.shape[1] == 1
                )
                # NOTE: We do NOT sync keep_group_mask across ranks because each rank
                # has different local groups. Instead, each rank filters independently,
                # and we sync the total_prompts count later to ensure consistent retry decisions.
                kept_groups = torch.nonzero(
                    keep_group_mask, as_tuple=False
                ).flatten()

                # Only take as many valid groups as we still need (cap at target_groups)
                needed_groups = max(0, target_groups - local_already_kept)
                groups_to_add = (
                    kept_groups[:needed_groups]
                    if needed_groups > 0
                    else kept_groups[:0]
                )
                last_selected_groups = groups_to_add

                local_kept_this_attempt = int(groups_to_add.numel())
                local_accumulated = local_already_kept + local_kept_this_attempt

                # Compact progress bar (only print when progress changes)
                bar_len = 20
                pct = (
                    0.0
                    if target_groups == 0
                    else min(1.0, local_accumulated / float(target_groups))
                )
                filled = int(round(bar_len * pct))
                bar = "#" * filled + "-" * (bar_len - filled)

                if self._filter_progress_last_kept != local_accumulated:
                    # Print compact progress: [filter] R0: [####----] 2/4 (+1) @1
                    self.accelerator.print(
                        f"[filter] R{rank_id}: [{bar}] {local_accumulated}/{target_groups} "
                        f"(+{local_kept_this_attempt}) @{attempt + 1}"
                    )
                    self._filter_progress_last_kept = local_accumulated

                if kept_groups.numel() == 0:
                    # No valid groups this attempt - will fall through to backfill logic
                    pass

                # Only slice and accumulate if we have valid groups this attempt
                # AND this rank hasn't already reached its target
                if groups_to_add.numel() > 0:
                    flat_indices = torch.cat(
                        [
                            torch.arange(
                                int(g.item()) * self.num_generations,
                                int(g.item() + 1) * self.num_generations,
                                device=device,
                            )
                            for g in groups_to_add
                        ]
                    )

                    # Slice current kept batch
                    prompt_ids_slice = prompt_ids[flat_indices]
                    prompt_mask_slice = prompt_mask[flat_indices]
                    completion_ids_slice = completion_ids[flat_indices]
                    completion_mask_slice = completion_mask[flat_indices]
                    rewards_slice = rewards[flat_indices]
                    rewards_per_func_slice = rewards_per_func[flat_indices]
                    sampling_per_token_logps_slice = (
                        sampling_per_token_logps[flat_indices]
                        if sampling_per_token_logps is not None
                        else None
                    )
                    importance_sampling_ratio_slice = (
                        importance_sampling_ratio[flat_indices]
                        if importance_sampling_ratio is not None
                        else None
                    )
                    old_per_token_logps_slice = (
                        old_per_token_logps[flat_indices]
                        if old_per_token_logps is not None
                        else None
                    )
                    ref_per_token_logps_slice = (
                        ref_per_token_logps[flat_indices]
                        if ref_per_token_logps is not None
                        else None
                    )
                    images_slice = (
                        [images[i] for i in flat_indices.cpu().tolist()]
                        if images is not None
                        else None
                    )
                    num_images_slice = (
                        [
                            len(img) if img is not None else 0
                            for img in images_slice
                        ]
                        if images_slice is not None
                        else None
                    )
                    prompts_text_slice = [
                        prompts_text[i] for i in flat_indices.cpu().tolist()
                    ]
                    completions_text_slice = [
                        completions_text[i] for i in flat_indices.cpu().tolist()
                    ]
                    completions_slice = (
                        [completions[i] for i in flat_indices.cpu().tolist()]
                        if isinstance(completions, list)
                        else completions_text_slice
                    )

                    if accum is None:
                        accum = {
                            "prompt_ids": [prompt_ids_slice],
                            "prompt_mask": [prompt_mask_slice],
                            "completion_ids": [completion_ids_slice],
                            "completion_mask": [completion_mask_slice],
                            "rewards": [rewards_slice],
                            "rewards_per_func": [rewards_per_func_slice],
                            "sampling_per_token_logps": (
                                [sampling_per_token_logps_slice]
                                if sampling_per_token_logps_slice is not None
                                else []
                            ),
                            "importance_sampling_ratio": (
                                [importance_sampling_ratio_slice]
                                if importance_sampling_ratio_slice is not None
                                else []
                            ),
                            "old_per_token_logps": (
                                [old_per_token_logps_slice]
                                if old_per_token_logps_slice is not None
                                else []
                            ),
                            "ref_per_token_logps": (
                                [ref_per_token_logps_slice]
                                if ref_per_token_logps_slice is not None
                                else []
                            ),
                            "images": (
                                [images_slice]
                                if images_slice is not None
                                else []
                            ),
                            "num_images": (
                                [num_images_slice]
                                if num_images_slice is not None
                                else []
                            ),
                            "prompts_text": [prompts_text_slice],
                            "completions_text": [completions_text_slice],
                            "completions": (
                                [completions_slice]
                                if isinstance(completions_slice, list)
                                else []
                            ),
                        }
                    else:
                        accum["prompt_ids"].append(prompt_ids_slice)
                        accum["prompt_mask"].append(prompt_mask_slice)
                        accum["completion_ids"].append(completion_ids_slice)
                        accum["completion_mask"].append(completion_mask_slice)
                        accum["rewards"].append(rewards_slice)
                        accum["rewards_per_func"].append(rewards_per_func_slice)
                        if sampling_per_token_logps_slice is not None:
                            accum["sampling_per_token_logps"].append(
                                sampling_per_token_logps_slice
                            )
                        if importance_sampling_ratio_slice is not None:
                            accum["importance_sampling_ratio"].append(
                                importance_sampling_ratio_slice
                            )
                        if old_per_token_logps_slice is not None:
                            accum["old_per_token_logps"].append(
                                old_per_token_logps_slice
                            )
                        if ref_per_token_logps_slice is not None:
                            accum["ref_per_token_logps"].append(
                                ref_per_token_logps_slice
                            )
                        if images_slice is not None:
                            accum["images"].append(images_slice)
                            accum["num_images"].append(num_images_slice or [])
                        accum["prompts_text"].append(prompts_text_slice)
                        accum["completions_text"].append(completions_text_slice)
                        if isinstance(completions_slice, list):
                            accum["completions"].append(completions_slice)

                valid_groups_accumulated += local_kept_this_attempt

                # Calculate local accumulated groups for this rank
                local_accumulated_groups = 0
                if accum and "rewards" in accum and accum["rewards"]:
                    local_accumulated_groups = (
                        sum(chunk.shape[0] for chunk in accum["rewards"])
                        // self.num_generations
                    )

                # Sync MINIMUM accumulated groups across ranks to ensure ALL ranks have enough
                # before proceeding (not just some ranks)
                min_accumulated_groups = local_accumulated_groups
                if dist.is_available() and dist.is_initialized():
                    min_accumulated_tensor = torch.tensor(
                        local_accumulated_groups,
                        device=device,
                        dtype=torch.long,
                    )
                    dist.all_reduce(
                        min_accumulated_tensor, op=dist.ReduceOp.MIN
                    )
                    min_accumulated_groups = int(min_accumulated_tensor.item())

                # Continue retrying if ANY rank needs more groups (min < target)
                # and max retries not exhausted
                retry_condition = min_accumulated_groups < target_groups and (
                    self.max_filter_gen_batches <= 0
                    or attempt < self.max_filter_gen_batches
                )

                if retry_condition:
                    attempt += 1

                    # IMPORTANT: all ranks must generate the SAME number of
                    # backfill prompt groups, otherwise reward gathering/slicing
                    # can silently misalign (or hang) under DDP.
                    global_deficit = max(
                        0, target_groups - min_accumulated_groups
                    )
                    # With retry_condition true, this should be >= 1, but be safe.
                    if global_deficit <= 0:
                        global_deficit = 1

                    backfill_inputs = self._sample_backfill_inputs(
                        global_deficit
                    )
                    if not backfill_inputs:
                        # Fall back to last inputs to stay in sync, or break if absolutely no data.
                        backfill_inputs = inputs[-1:] if inputs else []
                        if not backfill_inputs:
                            self.accelerator.print(
                                f"[filter] R{rank_id}: ERROR - no data for backfill"
                            )
                            break

                    # inputs = backfill_inputs
                    inputs = []
                    for item in backfill_inputs:
                        for _ in range(self.num_generations):
                            inputs.append(item.copy())

                    continue  # generate more prompts to fill the batch

                # -----------------------------------------------------------------
                # Finalize filtered batch to a fixed size (DAPO-style)
                # -----------------------------------------------------------------
                def _pad_and_cat(tensors, pad_token_id, left=True):
                    if not tensors:
                        return None
                    max_len = max(t.shape[1] for t in tensors)
                    padded = []
                    for t in tensors:
                        if t.shape[1] < max_len:
                            pad_width = max_len - t.shape[1]
                            pad = torch.full(
                                (t.shape[0], pad_width),
                                pad_token_id,
                                dtype=t.dtype,
                                device=t.device,
                            )
                            t = torch.cat([pad, t] if left else [t, pad], dim=1)
                        padded.append(t)
                    return torch.cat(padded, dim=0)

                def _cat_tensor_list(tensors):
                    if tensors is None:
                        return None
                    return torch.cat(tensors, dim=0) if tensors else None

                # How many prompt groups (of size `num_generations`) we have collected so far.
                accumulated_groups = (
                    sum(chunk.shape[0] for chunk in accum["rewards"])
                    // self.num_generations
                    if accum is not None and accum.get("rewards")
                    else 0
                )
                missing_groups = max(0, target_groups - accumulated_groups)

                if missing_groups > 0:
                    # We could not collect enough *valid* groups within the retry budget.
                    # To keep the optimization batch size stable (and avoid DDP/NCCL hangs),
                    # top up with unfiltered groups from the **last generated batch**.
                    num_groups_in_last = max(
                        1, rewards.shape[0] // self.num_generations
                    )
                    all_groups = torch.arange(
                        num_groups_in_last, device=device, dtype=torch.long
                    )

                    # Avoid reusing groups that we already appended from this last batch.
                    if (
                        last_selected_groups is not None
                        and last_selected_groups.numel() > 0
                    ):
                        mask = torch.ones(
                            num_groups_in_last, device=device, dtype=torch.bool
                        )
                        mask[last_selected_groups] = False
                        candidate_groups = all_groups[mask]
                    else:
                        candidate_groups = all_groups

                    # If everything was already used (rare), fall back to allowing reuse.
                    if candidate_groups.numel() == 0:
                        candidate_groups = all_groups

                    # Repeat candidates if needed (defensive; usually not needed).
                    if candidate_groups.numel() < missing_groups:
                        repeats = (
                            missing_groups + candidate_groups.numel() - 1
                        ) // max(1, candidate_groups.numel())
                        candidate_groups = candidate_groups.repeat(repeats)

                    fill_groups = candidate_groups[:missing_groups]

                    flat_indices = torch.cat(
                        [
                            torch.arange(
                                g * self.num_generations,
                                (g + 1) * self.num_generations,
                                device=device,
                                dtype=torch.long,
                            )
                            for g in fill_groups
                        ]
                    )

                    prompt_ids_slice = prompt_ids[flat_indices]
                    prompt_mask_slice = prompt_mask[flat_indices]
                    completion_ids_slice = completion_ids[flat_indices]
                    completion_mask_slice = completion_mask[flat_indices]
                    rewards_slice = rewards[flat_indices]
                    rewards_per_func_slice = (
                        rewards_per_func[flat_indices]
                        if rewards_per_func is not None
                        else None
                    )

                    sampling_per_token_logps_slice = (
                        sampling_per_token_logps[flat_indices]
                        if sampling_per_token_logps is not None
                        else None
                    )
                    old_per_token_logps_slice = (
                        old_per_token_logps[flat_indices]
                        if old_per_token_logps is not None
                        else None
                    )
                    ref_per_token_logps_slice = (
                        ref_per_token_logps[flat_indices]
                        if ref_per_token_logps is not None
                        else None
                    )
                    importance_sampling_ratio_slice = (
                        importance_sampling_ratio[flat_indices]
                        if importance_sampling_ratio is not None
                        else None
                    )

                    # Slice python lists for logging/debug.
                    flat_idx_list = flat_indices.tolist()
                    prompts_text_slice = [
                        prompts_text[i] for i in flat_idx_list
                    ]
                    completions_text_slice = [
                        completions_text[i] for i in flat_idx_list
                    ]
                    completions_slice = [completions[i] for i in flat_idx_list]
                    images_slice = (
                        [images[i] for i in flat_idx_list]
                        if images is not None
                        else None
                    )
                    num_images_slice = (
                        [num_images[i] for i in flat_idx_list]
                        if num_images is not None
                        else None
                    )

                    if accum is None:
                        accum = {
                            "prompt_ids": [prompt_ids_slice],
                            "prompt_mask": [prompt_mask_slice],
                            "completion_ids": [completion_ids_slice],
                            "completion_mask": [completion_mask_slice],
                            "rewards": [rewards_slice],
                            "rewards_per_func": (
                                [rewards_per_func_slice]
                                if rewards_per_func_slice is not None
                                else None
                            ),
                            "sampling_per_token_logps": (
                                [sampling_per_token_logps_slice]
                                if sampling_per_token_logps_slice is not None
                                else None
                            ),
                            "old_per_token_logps": (
                                [old_per_token_logps_slice]
                                if old_per_token_logps_slice is not None
                                else None
                            ),
                            "ref_per_token_logps": (
                                [ref_per_token_logps_slice]
                                if ref_per_token_logps_slice is not None
                                else None
                            ),
                            "importance_sampling_ratio": (
                                [importance_sampling_ratio_slice]
                                if importance_sampling_ratio_slice is not None
                                else None
                            ),
                            "prompts_text": [prompts_text_slice],
                            "completions_text": [completions_text_slice],
                            "completions": [completions_slice],
                            "images": (
                                [images_slice]
                                if images_slice is not None
                                else None
                            ),
                            "num_images": (
                                [num_images_slice]
                                if num_images_slice is not None
                                else None
                            ),
                        }
                    else:
                        accum["prompt_ids"].append(prompt_ids_slice)
                        accum["prompt_mask"].append(prompt_mask_slice)
                        accum["completion_ids"].append(completion_ids_slice)
                        accum["completion_mask"].append(completion_mask_slice)
                        accum["rewards"].append(rewards_slice)
                        if (
                            accum.get("rewards_per_func") is not None
                            and rewards_per_func_slice is not None
                        ):
                            accum["rewards_per_func"].append(
                                rewards_per_func_slice
                            )
                        if (
                            accum.get("sampling_per_token_logps") is not None
                            and sampling_per_token_logps_slice is not None
                        ):
                            accum["sampling_per_token_logps"].append(
                                sampling_per_token_logps_slice
                            )
                        if (
                            accum.get("old_per_token_logps") is not None
                            and old_per_token_logps_slice is not None
                        ):
                            accum["old_per_token_logps"].append(
                                old_per_token_logps_slice
                            )
                        if (
                            accum.get("ref_per_token_logps") is not None
                            and ref_per_token_logps_slice is not None
                        ):
                            accum["ref_per_token_logps"].append(
                                ref_per_token_logps_slice
                            )
                        if (
                            accum.get("importance_sampling_ratio") is not None
                            and importance_sampling_ratio_slice is not None
                        ):
                            accum["importance_sampling_ratio"].append(
                                importance_sampling_ratio_slice
                            )
                        accum["prompts_text"].append(prompts_text_slice)
                        accum["completions_text"].append(completions_text_slice)
                        accum["completions"].append(completions_slice)
                        if (
                            accum.get("images") is not None
                            and images_slice is not None
                        ):
                            accum["images"].append(images_slice)
                        if (
                            accum.get("num_images") is not None
                            and num_images_slice is not None
                        ):
                            accum["num_images"].append(num_images_slice)

                    accumulated_groups += missing_groups

                # At this point, we should have exactly `target_groups` groups locally.
                if accum is None or not accum.get("rewards"):
                    # Absolute fallback: keep the last generated batch (should be rare).
                    prompt_ids = prompt_ids
                    prompt_mask = prompt_mask
                    completion_ids = completion_ids
                    completion_mask = completion_mask
                    rewards = rewards
                else:
                    prompt_ids = _pad_and_cat(
                        accum["prompt_ids"],
                        self.processing_class.pad_token_id,
                        left=True,
                    )
                    prompt_mask = _pad_and_cat(
                        accum["prompt_mask"], 0, left=True
                    )
                    completion_ids = _pad_and_cat(
                        accum["completion_ids"],
                        self.processing_class.pad_token_id,
                        left=False,
                    )
                    completion_mask = _pad_and_cat(
                        accum["completion_mask"], 0, left=False
                    )
                    rewards = torch.cat(accum["rewards"], dim=0)

                    if accum.get("rewards_per_func") is not None:
                        rewards_per_func = torch.cat(
                            accum["rewards_per_func"], dim=0
                        )

                    if accum.get("sampling_per_token_logps") is not None:
                        sampling_per_token_logps = _pad_and_cat(
                            accum["sampling_per_token_logps"], 0, left=False
                        )
                    if accum.get("old_per_token_logps") is not None:
                        old_per_token_logps = _pad_and_cat(
                            accum["old_per_token_logps"], 0, left=False
                        )
                    if accum.get("ref_per_token_logps") is not None:
                        ref_per_token_logps = _pad_and_cat(
                            accum["ref_per_token_logps"], 0, left=False
                        )
                    if accum.get("importance_sampling_ratio") is not None:
                        importance_sampling_ratio = _pad_and_cat(
                            accum["importance_sampling_ratio"], 0, left=False
                        )

                    # Restore python lists for logs.
                    prompts_text = [
                        txt
                        for chunk in accum.get("prompts_text", [])
                        for txt in chunk
                    ]
                    completions_text = [
                        txt
                        for chunk in accum.get("completions_text", [])
                        for txt in chunk
                    ]
                    completions = [
                        c
                        for chunk in accum.get("completions", [])
                        for c in chunk
                    ]

                    if accum.get("images") is not None:
                        images = [
                            img for chunk in accum["images"] for img in chunk
                        ]
                    if accum.get("num_images") is not None:
                        num_images = [
                            n
                            for chunk in accum["num_images"]
                            for n in (chunk or [])
                        ]

                # Enforce a fixed per-rank batch size (stable across steps).
                target_batch_size = target_groups * self.num_generations
                if rewards.shape[0] != target_batch_size:
                    # Truncate or pad (by repeating from the beginning) in whole-group increments.
                    if rewards.shape[0] > target_batch_size:
                        idx = slice(0, target_batch_size)
                        prompt_ids = prompt_ids[idx]
                        prompt_mask = prompt_mask[idx]
                        completion_ids = completion_ids[idx]
                        completion_mask = completion_mask[idx]
                        rewards = rewards[idx]
                        if rewards_per_func is not None:
                            rewards_per_func = rewards_per_func[idx]
                        if sampling_per_token_logps is not None:
                            sampling_per_token_logps = sampling_per_token_logps[
                                idx
                            ]
                        if old_per_token_logps is not None:
                            old_per_token_logps = old_per_token_logps[idx]
                        if ref_per_token_logps is not None:
                            ref_per_token_logps = ref_per_token_logps[idx]
                        if importance_sampling_ratio is not None:
                            importance_sampling_ratio = (
                                importance_sampling_ratio[idx]
                            )
                        prompts_text = prompts_text[:target_batch_size]
                        completions_text = completions_text[:target_batch_size]
                        completions = completions[:target_batch_size]
                        if images is not None:
                            images = images[:target_batch_size]
                        if num_images is not None:
                            num_images = num_images[:target_batch_size]
                    else:
                        # Pad by repeating groups from the start.
                        pad_needed = target_batch_size - rewards.shape[0]
                        if rewards.shape[0] > 0:
                            rep_idx = (
                                torch.arange(pad_needed, device=device)
                                % rewards.shape[0]
                            )
                            prompt_ids = torch.cat(
                                [prompt_ids, prompt_ids[rep_idx]], dim=0
                            )
                            prompt_mask = torch.cat(
                                [prompt_mask, prompt_mask[rep_idx]], dim=0
                            )
                            completion_ids = torch.cat(
                                [completion_ids, completion_ids[rep_idx]], dim=0
                            )
                            completion_mask = torch.cat(
                                [completion_mask, completion_mask[rep_idx]],
                                dim=0,
                            )
                            rewards = torch.cat(
                                [rewards, rewards[rep_idx]], dim=0
                            )
                            if rewards_per_func is not None:
                                rewards_per_func = torch.cat(
                                    [
                                        rewards_per_func,
                                        rewards_per_func[rep_idx],
                                    ],
                                    dim=0,
                                )
                            if sampling_per_token_logps is not None:
                                sampling_per_token_logps = torch.cat(
                                    [
                                        sampling_per_token_logps,
                                        sampling_per_token_logps[rep_idx],
                                    ],
                                    dim=0,
                                )
                            if old_per_token_logps is not None:
                                old_per_token_logps = torch.cat(
                                    [
                                        old_per_token_logps,
                                        old_per_token_logps[rep_idx],
                                    ],
                                    dim=0,
                                )
                            if ref_per_token_logps is not None:
                                ref_per_token_logps = torch.cat(
                                    [
                                        ref_per_token_logps,
                                        ref_per_token_logps[rep_idx],
                                    ],
                                    dim=0,
                                )
                            if importance_sampling_ratio is not None:
                                importance_sampling_ratio = torch.cat(
                                    [
                                        importance_sampling_ratio,
                                        importance_sampling_ratio[rep_idx],
                                    ],
                                    dim=0,
                                )
                            prompts_text = prompts_text + [
                                prompts_text[i] for i in rep_idx.tolist()
                            ]
                            completions_text = completions_text + [
                                completions_text[i] for i in rep_idx.tolist()
                            ]
                            completions = completions + [
                                completions[i] for i in rep_idx.tolist()
                            ]
                            if images is not None:
                                images = images + [
                                    images[i] for i in rep_idx.tolist()
                                ]
                            if num_images is not None:
                                num_images = num_images + [
                                    num_images[i] for i in rep_idx.tolist()
                                ]

                if self.accelerator.is_local_main_process:
                    kept_frac = valid_groups_accumulated / max(
                        1.0, target_groups
                    )
                    fallback_groups = max(
                        0, target_groups - valid_groups_accumulated
                    )
                    self.accelerator.print(
                        f"[filter] R{rank_id}: DONE -> valid_groups={valid_groups_accumulated}/{target_groups}, "
                        f"fallback_groups={fallback_groups}, kept_frac={kept_frac:.3f}, retries={attempt}"
                    )

                rewards_matrix = (
                    None  # force recreation for the finalized batch
                )
            else:
                # Filtering not enabled (eval mode) - keep all groups
                pass
            break

        # --- Grouped-wise rewards / advantages ---
        # GRPO/DAPO operate on prompt-groups of size `num_generations`. In practice
        # we expect the batch to be divisible by num_generations. However, a few
        # edge cases can break that (e.g., cross-rank sync truncation).
        #
        # To keep behavior well-defined and avoid `.view()` shape errors, we:
        #   1) truncate to full groups when possible, and
        #   2) otherwise fall back to group_size=1 (no group normalization).
        num_gen = max(1, int(self.num_generations))
        if (
            rewards_matrix is None
            and rewards.shape[0] > 0
            and rewards.shape[0] % num_gen != 0
        ):
            usable = (rewards.shape[0] // num_gen) * num_gen
            if usable > 0:
                # Truncate all per-sample tensors/lists to preserve alignment.
                prompt_ids = prompt_ids[:usable]
                prompt_mask = prompt_mask[:usable]
                completion_ids = completion_ids[:usable]
                completion_mask = completion_mask[:usable]
                rewards = rewards[:usable]
                rewards_per_func = rewards_per_func[:usable]
                acc_tensor = acc_tensor[:usable]
                if sampling_per_token_logps is not None:
                    sampling_per_token_logps = sampling_per_token_logps[:usable]
                if importance_sampling_ratio is not None:
                    importance_sampling_ratio = importance_sampling_ratio[
                        :usable
                    ]
                if old_per_token_logps is not None:
                    old_per_token_logps = old_per_token_logps[:usable]
                if ref_per_token_logps is not None:
                    ref_per_token_logps = ref_per_token_logps[:usable]
                if images is not None:
                    images = images[:usable]
                    if num_images is not None:
                        num_images = num_images[:usable]
                prompts_text = prompts_text[:usable]
                completions_text = completions_text[:usable]
                if isinstance(completions, list):
                    completions = completions[:usable]

        if rewards_matrix is None:
            if rewards.shape[0] > 0 and rewards.shape[0] % num_gen == 0:
                group_size = num_gen
            else:
                group_size = 1
            rewards_matrix = (
                rewards.view(-1, group_size)
                if rewards.shape[0] > 0
                else rewards.view(0, group_size)
            )
        else:
            group_size = rewards_matrix.size(1)

        total = rewards.shape[0]

        # Handle empty batch case (rank has no valid data after filtering)
        if total == 0:
            # Create empty tensors with correct shapes for this rank
            # Other ranks with data will contribute to gradient; this rank contributes nothing
            group_count = 0
            mean_grouped_rewards = torch.zeros(0, device=device)
            std_grouped_rewards = torch.ones(0, device=device)
            advantages = torch.zeros(0, device=device)
            is_std_zero = torch.zeros(0, dtype=torch.bool, device=device)
            all_process_advantages = torch.zeros(0, device=device)
            group_ids = torch.zeros(0, dtype=torch.long, device=device)
            # Define these for logging below (avoid UnboundLocalError).
            mean_per_group = torch.zeros(1, device=device)
            std_per_group = torch.zeros(1, device=device)
        else:
            group_ids = torch.div(
                torch.arange(total, device=device, dtype=torch.long),
                group_size,
                rounding_mode="floor",
            )
            group_count = group_ids.max().item() + 1

            counts = torch.bincount(group_ids, minlength=group_count).float()
            counts = counts.clamp(min=1.0)  # avoid divide-by-zero
            sums = torch.bincount(
                group_ids, weights=rewards, minlength=group_count
            )
            mean_per_group = sums / counts

            # Variance with Bessel correction avoided (match verl: std over group, singletons -> 1)
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

            mean_grouped_rewards = mean_per_group[group_ids]
            std_grouped_rewards = std_per_group[group_ids]

            advantages = rewards - mean_grouped_rewards
            if self.norm_adv_by_std_in_grpo:
                advantages = advantages / (std_grouped_rewards + 1e-6)

            is_std_zero = torch.isclose(
                std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
            )

            all_process_advantages = (
                advantages.clone()
            )  # keep the aggregated advantages for logging

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            if rewards_per_func.shape[0] == 0:
                mean_rewards = torch.tensor(0.0, device=device)
                std_func_rewards = torch.tensor(0.0, device=device)
            else:
                mean_rewards = torch.nanmean(rewards_per_func[:, i])
                std_func_rewards = nanstd(rewards_per_func[:, i])
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                self.accelerator.gather(mean_rewards).nanmean().item()
            )
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(
                self.accelerator.gather(std_func_rewards).nanmean().item()
            )
        self._metrics[mode]["reward"].append(
            self.accelerator.gather(mean_per_group.mean()).nanmean().item()
        )
        self._metrics[mode]["reward_std"].append(
            self.accelerator.gather(std_per_group.mean()).nanmean().item()
        )
        frac_zero_std = (
            is_std_zero.float().mean()
            if is_std_zero.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        self._metrics[mode]["frac_reward_zero_std"].append(
            self.accelerator.gather(frac_zero_std).nanmean().item()
        )
        if mode == "eval":
            try:
                acc_matrix = acc_tensor.view(-1, group_size)
                pass_rate = (acc_matrix.sum(dim=1) > 0).float().mean()
                gathered_pass = (
                    self.accelerator.gather(pass_rate.to(device))
                    .float()
                    .mean()
                    .item()
                )
                self._metrics[mode]["pass_rate"].append(gathered_pass)
            except Exception:
                pass  # avoid crashing eval if shapes are unexpected

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and (old_per_token_logps is not None)
            and (sampling_per_token_logps is not None)
            and (importance_sampling_ratio is not None)
        ):
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode][
                "sampling/sampling_logp_difference/mean"
            ].append(self.accelerator.gather(mean_delta).mean().item())
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode][
                "sampling/importance_sampling_ratio/min"
            ].append(
                nanmin(
                    self.accelerator.gather(min_importance_sampling_ratio)
                ).item()
            )
            self._metrics[mode][
                "sampling/importance_sampling_ratio/mean"
            ].append(
                self.accelerator.gather(mean_importance_sampling_ratio)
                .nanmean()
                .item()
            )
            self._metrics[mode][
                "sampling/importance_sampling_ratio/max"
            ].append(
                nanmax(
                    self.accelerator.gather(max_importance_sampling_ratio)
                ).item()
            )

        # Recompute token count after filtering for logging and loss scaling
        num_items_in_batch = completion_mask.sum()

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            # "rewards": rewards,   # ! disable rewards now and use GRPO style advantage (normalized reward) as rewards for online SFT
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "rewards_per_func": rewards_per_func,
        }
        if filtered:
            # Keep shuffle_sequence_dict happy by providing a tensor, not a raw bool
            output["filtered"] = torch.tensor(True, device=device)
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs[
                "pixel_attention_mask"
            ]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output

    def _compute_loss(self, model, inputs):
        """
        Copy of GRPOTrainer._compute_loss with optional entropy regularization.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
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

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
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
            log_importance_weights = (log_ratio * completion_mask).sum(
                -1
            ) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(
            coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
        )

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # Implement dual-clip PPO (matching verl's compute_policy_loss_vanilla)
        # See: verl/verl/trainer/ppo/core_algos.py:944-963
        advantages_expanded = advantages.unsqueeze(1)

        # Standard PPO clipping
        pg_losses1 = -advantages_expanded * coef_1
        pg_losses2 = -advantages_expanded * coef_2
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

        # Dual-clip: Apply lower bound for negative advantages
        pg_losses3 = -advantages_expanded * self.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

        # Use dual-clip for negative advantages, standard clip for positive
        per_token_loss = torch.where(
            advantages_expanded < 0, clip_pg_losses2, clip_pg_losses1
        )

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = (
                per_token_loss * inputs["importance_sampling_ratio"]
            )

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.entropy_coeff != 0.0:
            per_token_loss = per_token_loss - self.entropy_coeff * entropies

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            # Match TRL's original DAPO loss: global token-mean without extra grad-acc scaling.
            # Guard against empty / all-masked batches (normalizer==0) which
            # would otherwise yield inf/NaN losses.
            normalizer = (
                inputs["num_items_in_batch"].to(per_token_loss.dtype)
                / float(max(1, self.accelerator.num_processes))
            ).clamp(min=1.0)
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )
        if self.entropy_coeff != 0.0:
            self._metrics[mode]["entropy_coeff"].append(self.entropy_coeff)

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
            advantages.unsqueeze(1) < 0
        )
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        # Track dual-clip lower bound activation (when pg_losses3 < clip_pg_losses1 for negative advantages)
        is_dual_clip_lower = (advantages.unsqueeze(1) < 0) & (
            pg_losses3 < clip_pg_losses1
        )

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        dual_clip_lower_ratio = masked_batch_mean(is_dual_clip_lower.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )

        # Log dual-clip lower bound activation
        gathered_dual_clip_lower = self.accelerator.gather(
            dual_clip_lower_ratio
        )
        self._metrics[mode]["clip_ratio/dual_clip_lower"].append(
            gathered_dual_clip_lower.nanmean().item()
        )

        return loss
