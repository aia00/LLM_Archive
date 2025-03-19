#!/usr/bin/env python
import json
import sys
from datasets import load_dataset

def main():
    # 1) Load the APPS dataset
    apps_dataset = load_dataset("codeparrot/apps")
    train_dataset = apps_dataset["test"]

    # 2) Filter for introductory problems
    intro_train_dataset = train_dataset.filter(lambda example: example["difficulty"] == "introductory")

    # 3) Gather questions for indices 0..150 (or until dataset ends)
    questions_info = []
    limit = min(151, len(intro_train_dataset))
    for i in range(0, 0+limit):
        problem = intro_train_dataset[i]
        question_text = problem["question"]
        # Collect more info if needed, e.g. "solutions"
        questions_info.append({
            "index": i,
            "question": question_text
        })

    # 4) Save to a file
    output_filename = "all_questions_0_150.json"
    with open(output_filename, "w") as f:
        json.dump(questions_info, f, indent=4)

    print(f"[Done] Saved questions from index 0 to {limit-1} in '{output_filename}'.")

if __name__ == "__main__":
    main()
