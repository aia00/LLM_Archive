"""Prepare math QA data in VERL's parquet format."""

from __future__ import annotations

import argparse
from typing import List

import sys
from pathlib import Path

from datasets import Dataset

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))

from mycode.dataset_utils import MathDatasetConfig, prepare_math_dataset


def build_prompt_messages(system_prompt: str, question: str) -> List[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VERL math dataset")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--question_key", type=str, default=None)
    parser.add_argument("--answer_key", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful math assistant.")
    parser.add_argument("--prompt_template", type=str, default="Problem:\n{question}\n\nAnswer:")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--eval_out", type=str, default=None)
    args = parser.parse_args()

    config = MathDatasetConfig(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        question_key=args.question_key,
        answer_key=args.answer_key,
        system_prompt="",
        prompt_template=args.prompt_template,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    train_dataset, eval_dataset = prepare_math_dataset(config)

    def convert(example):
        question = example.get("prompt", "")
        if isinstance(question, str):
            prompt_text = question
        else:
            prompt_text = str(question)
        messages = build_prompt_messages(args.system_prompt, prompt_text)
        return {
            "prompt": messages,
            "reward_model": {"ground_truth": example.get("ground_truth", ""), "style": "rule"},
            "data_source": "math_multiobj",
            "extra_info": {"index": example.get("index", 0)},
        }

    train_verl = train_dataset.map(convert, desc="Building VERL train parquet")
    train_verl.to_parquet(args.train_out)

    if args.eval_out and eval_dataset is not None:
        eval_verl = eval_dataset.map(convert, desc="Building VERL eval parquet")
        eval_verl.to_parquet(args.eval_out)


if __name__ == "__main__":
    main()
