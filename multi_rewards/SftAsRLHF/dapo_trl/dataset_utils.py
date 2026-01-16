"""
Dataset loading and preprocessing for DAPO training.
Loads the DAPO-Math-17k dataset and prepares it for training.
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset


def load_dapo_math_dataset(
    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Load the DAPO-Math-17k dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to load
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        Dataset object
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to {len(dataset)} samples")

    print(f"Loaded {len(dataset)} samples")
    return dataset


def format_prompt_for_chat(prompt_list: List[Dict[str, str]]) -> str:
    """
    Convert prompt list to a formatted string suitable for chat models.

    Args:
        prompt_list: List of message dictionaries with 'role' and 'content'

    Returns:
        Formatted prompt string
    """
    if isinstance(prompt_list, str):
        return prompt_list

    formatted_parts = []
    for message in prompt_list:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")

    return "\n\n".join(formatted_parts)


def extract_ground_truth(reward_model_info: Dict) -> str:
    """
    Extract ground truth answer from reward_model field.

    Args:
        reward_model_info: Dictionary containing reward model information

    Returns:
        Ground truth answer string
    """
    if isinstance(reward_model_info, dict):
        return reward_model_info.get("ground_truth", "")
    return ""


def preprocess_dataset_for_grpo(
    dataset,
    prompt_key: str = "prompt",
    max_prompt_length: int = 2048,
):
    """
    Preprocess the DAPO dataset for GRPO training.

    Args:
        dataset: Raw dataset from load_dapo_math_dataset
        prompt_key: Key for the prompt field
        max_prompt_length: Maximum length for prompts

    Returns:
        Processed dataset ready for GRPO training
    """

    def process_example(example):
        # Extract prompt
        prompt = example.get(prompt_key, [])
        # Preserve conversational format when available so TRL can apply chat templates
        if isinstance(prompt, list):
            formatted_prompt = prompt
        else:
            formatted_prompt = str(prompt)

        # Extract ground truth
        reward_model = example.get("reward_model", {})
        ground_truth = extract_ground_truth(reward_model)

        # Store data source for tracking
        data_source = example.get("data_source", "math_dapo")

        return {
            "prompt": formatted_prompt,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "ability": example.get("ability", ""),
        }

    processed_dataset = dataset.map(
        process_example,
        remove_columns=[
            col
            for col in dataset.column_names
            if col not in ["prompt", "ground_truth", "data_source", "ability"]
        ],
        desc="Processing dataset",
        num_proc=32,
    )

    return processed_dataset


def create_reward_lookup(dataset) -> Dict[str, str]:
    """
    Create a lookup dictionary mapping prompts to ground truth answers.

    This is needed for the reward function to access ground truths during training.

    Args:
        dataset: Processed dataset

    Returns:
        Dictionary mapping prompt to ground truth
    """
    reward_lookup = {}
    # Note: When prompts are conversational (list/dict), they are not hashable.
    # This lookup is only reliable for string prompts. For conversational prompts,
    # the reward function should read `ground_truth` directly from kwargs.
    for example in dataset:
        prompt = example["prompt"]
        ground_truth = example.get("ground_truth", "")
        if isinstance(prompt, str):
            reward_lookup[prompt] = ground_truth

    return reward_lookup


def prepare_dapo_dataset(
    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    train_split: str = "train",
    val_split: Optional[str] = None,
    eval_dataset_name: Optional[str] = None,
    eval_split: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    prompt_key: str = "prompt",
    max_prompt_length: int = 2048,
    val_fraction: float = 0.1,
    val_seed: int = 42,
):
    """
    Main function to prepare the DAPO dataset for training.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        train_split: Training split name
        val_split: Validation split name (optional)
        max_train_samples: Maximum training samples
        max_val_samples: Maximum validation samples
        prompt_key: Key for prompts in the dataset
        max_prompt_length: Maximum prompt length

    Returns:
        Tuple of (train_dataset, val_dataset, reward_lookup)
    """
    # Strategy:
    # - If an explicit val_split is provided and available, load it.
    # - If eval_dataset_name is provided, load that for validation (AIME-2024).
    # - Otherwise, create a deterministic train/val split from the requested train split.
    #   We split BEFORE sub-sampling so the held-out validation set is never used in training.

    # Load full dataset for splitting
    full_dataset = load_dapo_math_dataset(
        dataset_name=dataset_name,
        split=train_split,
        max_samples=None,
    )

    if val_split is None and eval_dataset_name is None:
        # Create a deterministic split
        try:
            split_dict = full_dataset.train_test_split(
                test_size=val_fraction, seed=val_seed, shuffle=True
            )
            raw_train_ds = split_dict["train"]
            raw_val_ds = split_dict["test"]
            print(
                f"Created validation split from '{train_split}': "
                f"train={len(raw_train_ds)}, val={len(raw_val_ds)} (fraction={val_fraction})"
            )
        except Exception as e:
            print(
                f"Warning: train_test_split failed ({e}); using entire dataset for training and no validation."
            )
            raw_train_ds = full_dataset
            raw_val_ds = None
    else:
        # Use provided val split or external eval dataset if available
        raw_train_ds = full_dataset
        if eval_dataset_name:
            try:
                # Load eval dataset - we'll apply max_val_samples later for consistency
                raw_val_ds = load_dapo_math_dataset(
                    dataset_name=eval_dataset_name,
                    split=eval_split or "train",
                    max_samples=max_val_samples,
                )
            except Exception as e:
                print(
                    f"Warning: Could not load eval dataset '{eval_dataset_name}': {e}"
                )
                raw_val_ds = None
        else:
            try:
                raw_val_ds = load_dapo_math_dataset(
                    dataset_name=dataset_name, split=val_split, max_samples=max_val_samples
                )
            except Exception as e:
                print(
                    f"Warning: Could not load validation split '{val_split}': {e}"
                )
                raw_val_ds = None

    # Apply optional sub-sampling AFTER splitting
    if max_train_samples is not None and raw_train_ds is not None:
        raw_train_ds = raw_train_ds.select(
            range(min(max_train_samples, len(raw_train_ds)))
        )
        print(
            f"Limited train split to {len(raw_train_ds)} samples after splitting"
        )
    # Note: max_val_samples is already applied during dataset loading above for eval_dataset_name/val_split cases
    # Only apply it here for the train_test_split case where we didn't apply it yet
    if max_val_samples is not None and raw_val_ds is not None:
        # Check if we need to apply max_val_samples (only for train_test_split case where it wasn't applied yet)
        if val_split is None and eval_dataset_name is None:
            raw_val_ds = raw_val_ds.select(
                range(min(max_val_samples, len(raw_val_ds)))
            )
            print(f"Limited val split to {len(raw_val_ds)} samples after splitting")

    # Preprocess both splits
    train_dataset = preprocess_dataset_for_grpo(
        raw_train_ds,
        prompt_key=prompt_key,
        max_prompt_length=max_prompt_length,
    )

    val_dataset = None
    if raw_val_ds is not None:
        val_dataset = preprocess_dataset_for_grpo(
            raw_val_ds,
            prompt_key=prompt_key,
            max_prompt_length=max_prompt_length,
        )

    # Create reward lookup for training
    reward_lookup = create_reward_lookup(train_dataset)

    print(f"Prepared {len(train_dataset)} training samples")
    if val_dataset:
        print(f"Prepared {len(val_dataset)} validation samples")

    return train_dataset, val_dataset, reward_lookup


if __name__ == "__main__":
    # Test the dataset loading
    train_dataset, val_dataset, reward_lookup = prepare_dapo_dataset(
        max_train_samples=10,
        max_val_samples=10,
    )

    print("\nExample training sample:")
    print(train_dataset[0])

    print(f"\nReward lookup size: {len(reward_lookup)}")
