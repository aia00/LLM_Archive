"""Dataset utilities for math QA datasets (e.g., MATH-500)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from datasets import load_dataset


@dataclass
class MathDatasetConfig:
    dataset_name: str = "HuggingFaceH4/MATH-500"
    train_split: str = "train"
    eval_split: Optional[str] = None
    question_key: Optional[str] = None
    answer_key: Optional[str] = None
    system_prompt: str = "You are a helpful math assistant."
    prompt_template: str = "Problem:\n{question}\n\nAnswer:"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


def _pick_key(column_names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in column_names:
            return name
    return None


def _infer_keys(column_names: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    question_key = _pick_key(column_names, ["question", "problem", "prompt"])
    answer_key = _pick_key(column_names, ["final_answer", "answer", "solution", "ground_truth"])
    return question_key, answer_key


def _format_prompt(question: str, system_prompt: str, template: str) -> str:
    body = template.format(question=question)
    if system_prompt:
        return f"{system_prompt}\n\n{body}"
    return body


def prepare_math_dataset(config: MathDatasetConfig):
    """Load and preprocess a math QA dataset into GRPO-ready fields."""
    dataset = load_dataset(config.dataset_name, split=config.train_split)
    question_key = config.question_key
    answer_key = config.answer_key

    if question_key is None or answer_key is None:
        inferred_q, inferred_a = _infer_keys(dataset.column_names)
        question_key = question_key or inferred_q
        answer_key = answer_key or inferred_a

    if question_key is None or answer_key is None:
        raise ValueError(
            "Unable to infer question/answer keys. "
            "Please pass --question_key and --answer_key."
        )

    def process_example(example: Dict) -> Dict:
        question = str(example.get(question_key, ""))
        answer = str(example.get(answer_key, ""))
        prompt = _format_prompt(question, config.system_prompt, config.prompt_template)
        return {
            "prompt": prompt,
            "ground_truth": answer,
            "data_source": config.dataset_name,
        }

    processed = dataset.map(
        process_example,
        remove_columns=[col for col in dataset.column_names if col not in [question_key, answer_key]],
        desc="Preparing math dataset",
        num_proc=8,
    )

    if config.max_train_samples is not None:
        processed = processed.select(range(min(config.max_train_samples, len(processed))))

    eval_dataset = None
    if config.eval_split:
        eval_ds = load_dataset(config.dataset_name, split=config.eval_split)
        eval_dataset = eval_ds.map(
            process_example,
            remove_columns=[col for col in eval_ds.column_names if col not in [question_key, answer_key]],
            desc="Preparing eval dataset",
            num_proc=8,
        )
        if config.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(config.max_eval_samples, len(eval_dataset))))

    return processed, eval_dataset
