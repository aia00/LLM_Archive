import os
from datasets import load_dataset

def main():
    # Load the APPS dataset and filter for introductory problems.
    apps_dataset = load_dataset("codeparrot/apps")
    train_dataset = apps_dataset["train"]
    intro_train_dataset = train_dataset.filter(lambda example: example["difficulty"] == "introductory")
    
    # Iterate over the introductory problems.
    for idx, problem in enumerate(intro_train_dataset):
        try:
            io_pair = eval(problem["input_output"])
        except Exception as e:
            print(f"Error evaluating input_output for index {idx}: {e}")
            continue
        
        # Ensure we have a dict and check if both inputs and outputs have more than 5 items.
        if isinstance(io_pair, dict):
            inputs_list = io_pair.get("inputs", [])
            outputs_list = io_pair.get("outputs", [])
            if len(inputs_list) > 1 and len(outputs_list) > 1:
                print(f"First problem with more than 3 test cases found at index {idx}.")
                print("Question:")
                print(problem["question"])
                break
    else:
        print("No problem with more than 3 test cases was found.")

if __name__ == "__main__":
    main()
