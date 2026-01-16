# online_sft/dataset_utils.py
# Dataset loading and preprocessing utilities for Online SFT training
# Supports multiple task types: TLDR summarization, Math reasoning, etc.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset, load_dataset


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


# =============================================================================
# TLDR Dataset Utilities
# =============================================================================


def prepare_tldr_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_prompt_length: int = 512,
    num_proc: int = 32,
) -> Dataset:
    """
    Prepare the TLDR dataset for Online SFT training.

    Args:
        dataset: Raw TLDR dataset from HuggingFace
        tokenizer: Tokenizer for prompt length filtering
        max_prompt_length: Maximum prompt length in tokens
        num_proc: Number of processes for dataset mapping

    Returns:
        Processed dataset with prompt_text field
    """

    def tokenize(element):
        # Access the prompt string from the messages field
        prompt_string = element["messages"][0]["content"]

        tokenized_output = tokenizer(
            prompt_string,
            padding=False,
            add_special_tokens=True,
        )

        input_ids_list = tokenized_output["input_ids"]

        return {
            "input_ids": input_ids_list,
            "lengths": len(input_ids_list),
            "prompt_text": prompt_string,
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=[
            c for c in dataset.column_names if c not in ["prompt_text"]
        ],
        num_proc=num_proc,
    )

    # Filter out prompts that are too long
    dataset = dataset.filter(
        lambda x: x["lengths"] <= max_prompt_length,
        num_proc=num_proc,
    )

    return dataset


def load_tldr_dataset(
    dataset_name: str = "trl-internal-testing/tldr-preference-sft-trl-style",
    train_split: str = "train",
    val_split: str = "validation",
    tokenizer: Any = None,
    max_prompt_length: int = 512,
    num_proc: int = 4,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare the TLDR dataset.

    Args:
        dataset_name: HuggingFace dataset name
        train_split: Training split name
        val_split: Validation split name
        tokenizer: Tokenizer for processing
        max_prompt_length: Maximum prompt length
        num_proc: Number of processes

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    train_dataset = prepare_tldr_dataset(
        dataset[train_split],
        tokenizer,
        max_prompt_length=max_prompt_length,
        num_proc=num_proc,
    )

    val_dataset = None
    if val_split in dataset:
        val_dataset = prepare_tldr_dataset(
            dataset[val_split],
            tokenizer,
            max_prompt_length=max_prompt_length,
            num_proc=num_proc,
        )

    print(f"Loaded {len(train_dataset)} training samples")
    if val_dataset:
        print(f"Loaded {len(val_dataset)} validation samples")

    return train_dataset, val_dataset


# =============================================================================
# Math Dataset Utilities (adapted from DAPO)
# =============================================================================


def prepare_math_dataset(
    dataset: Dataset,
    tokenizer: Any,
    prompt_key: str = "prompt",
    max_prompt_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """
    Prepare the Math dataset for Online SFT training.

    Args:
        dataset: Raw math dataset from HuggingFace
        tokenizer: Tokenizer for prompt length filtering
        prompt_key: Key for the prompt field
        max_prompt_length: Maximum prompt length
        num_proc: Number of processes

    Returns:
        Processed dataset with prompt_text and ground_truth fields
    """

    def process_example(example):
        # Extract prompt
        prompt = example.get(prompt_key, [])

        # Convert conversational format to string for Online SFT
        if isinstance(prompt, list):
            prompt_text = format_prompt_for_chat(prompt)
        else:
            prompt_text = str(prompt)

        # Extract ground truth
        reward_model = example.get("reward_model", {})
        ground_truth = extract_ground_truth(reward_model)

        # Store data source for tracking
        data_source = example.get("data_source", "math_dapo")

        return {
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "ability": example.get("ability", ""),
            "prompt_raw": prompt,  # Keep original format
        }

    processed = dataset.map(
        process_example,
        remove_columns=[
            col
            for col in dataset.column_names
            if col
            not in [
                "prompt_text",
                "ground_truth",
                "data_source",
                "ability",
                "prompt_raw",
            ]
        ],
        desc="Processing math dataset",
        num_proc=num_proc,
    )

    # Tokenize and filter by length
    def add_lengths(example):
        tokenized = tokenizer(
            example["prompt_text"],
            padding=False,
            add_special_tokens=True,
        )
        return {"lengths": len(tokenized["input_ids"])}

    processed = processed.map(add_lengths, num_proc=num_proc)
    processed = processed.filter(
        lambda x: x["lengths"] <= max_prompt_length,
        num_proc=num_proc,
    )

    return processed


def load_math_dataset(
    dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    train_split: str = "train",
    eval_dataset_name: Optional[str] = None,
    eval_split: Optional[str] = None,
    tokenizer: Any = None,
    max_prompt_length: int = 2048,
    val_fraction: float = 0.1,
    val_seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_proc: int = 4,
) -> Tuple[Dataset, Optional[Dataset], Dict[str, str]]:
    """
    Load and prepare the Math dataset (DAPO-Math-17k or similar).

    Args:
        dataset_name: HuggingFace dataset name
        train_split: Training split name
        eval_dataset_name: Optional separate evaluation dataset
        eval_split: Evaluation split name
        tokenizer: Tokenizer for processing
        max_prompt_length: Maximum prompt length
        val_fraction: Fraction of data for validation if no explicit split
        val_seed: Seed for deterministic splitting
        max_train_samples: Optional max training samples
        max_val_samples: Optional max validation samples
        num_proc: Number of processes

    Returns:
        Tuple of (train_dataset, val_dataset, reward_lookup)
    """
    print(f"Loading dataset: {dataset_name}, split: {train_split}")
    full_dataset = load_dataset(dataset_name, split=train_split)
    print(f"Loaded {len(full_dataset)} samples")

    # Create train/val split
    if eval_dataset_name is None:
        try:
            split_dict = full_dataset.train_test_split(
                test_size=val_fraction,
                seed=val_seed,
                shuffle=True,
            )
            raw_train_ds = split_dict["train"]
            raw_val_ds = split_dict["test"]
            print(
                f"Created validation split: "
                f"train={len(raw_train_ds)}, val={len(raw_val_ds)} (fraction={val_fraction})"
            )
        except Exception as e:
            print(
                f"Warning: train_test_split failed ({e}); using entire dataset for training."
            )
            raw_train_ds = full_dataset
            raw_val_ds = None
    else:
        raw_train_ds = full_dataset
        try:
            raw_val_ds = load_dataset(
                eval_dataset_name,
                split=eval_split or "train",
            )
            print(
                f"Loaded {len(raw_val_ds)} evaluation samples from {eval_dataset_name}"
            )
        except Exception as e:
            print(
                f"Warning: Could not load eval dataset '{eval_dataset_name}': {e}"
            )
            raw_val_ds = None

    # Apply optional sub-sampling
    if max_train_samples is not None:
        raw_train_ds = raw_train_ds.select(
            range(min(max_train_samples, len(raw_train_ds)))
        )
        print(f"Limited train split to {len(raw_train_ds)} samples")
    if max_val_samples is not None and raw_val_ds is not None:
        raw_val_ds = raw_val_ds.select(
            range(min(max_val_samples, len(raw_val_ds)))
        )
        print(f"Limited val split to {len(raw_val_ds)} samples")

    # Process datasets
    train_dataset = prepare_math_dataset(
        raw_train_ds,
        tokenizer,
        max_prompt_length=max_prompt_length,
        num_proc=num_proc,
    )

    val_dataset = None
    if raw_val_ds is not None:
        val_dataset = prepare_math_dataset(
            raw_val_ds,
            tokenizer,
            max_prompt_length=max_prompt_length,
            num_proc=num_proc,
        )

    # Create reward lookup for prompt -> ground_truth mapping
    reward_lookup = {}
    for example in train_dataset:
        prompt = example["prompt_text"]
        ground_truth = example.get("ground_truth", "")
        if isinstance(prompt, str) and ground_truth:
            reward_lookup[prompt] = ground_truth

    print(f"Prepared {len(train_dataset)} training samples")
    if val_dataset:
        print(f"Prepared {len(val_dataset)} validation samples")
    print(f"Reward lookup size: {len(reward_lookup)}")

    return train_dataset, val_dataset, reward_lookup


# =============================================================================
# Generic Dataset Interface
# =============================================================================


def prepare_dataset(
    task_type: str,
    tokenizer: Any,
    **kwargs,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dict[str, str]]]:
    """
    Generic dataset preparation function.

    Args:
        task_type: Type of task ("tldr", "math", etc.)
        tokenizer: Tokenizer for processing
        **kwargs: Task-specific arguments

    Returns:
        Tuple of (train_dataset, val_dataset, reward_lookup or None)
    """
    if task_type == "tldr":
        train_ds, val_ds = load_tldr_dataset(tokenizer=tokenizer, **kwargs)
        return train_ds, val_ds, None
    elif task_type == "math":
        return load_math_dataset(tokenizer=tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def collate_fn(batch: List[Dict]) -> List[str]:
    """Simple collate function that returns a list of prompt texts."""
    return [item["prompt_text"] for item in batch]


def collate_fn_with_ground_truth(
    batch: List[Dict],
) -> Tuple[List[str], List[str]]:
    """Collate function that returns prompts and ground truths."""
    prompts = [item["prompt_text"] for item in batch]
    ground_truths = [item.get("ground_truth", "") for item in batch]
    return prompts, ground_truths


# =============================================================================
# Test
# =============================================================================


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Test with a simple tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test math dataset loading (will fail if dataset not available)
    try:
        train_ds, val_ds, reward_lookup = load_math_dataset(
            tokenizer=tokenizer,
            max_train_samples=10,
            max_val_samples=5,
        )
        print("\nExample training sample:")
        print(train_ds[0])
    except Exception as e:
        print(
            f"Math dataset test failed (expected if dataset not available): {e}"
        )
        print(
            f"Math dataset test failed (expected if dataset not available): {e}"
        )
