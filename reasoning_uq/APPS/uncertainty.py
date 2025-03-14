import re
import os
import subprocess
import tempfile
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random

# -------------------------------
# Load the APPS dataset and filter for introductory problems.
apps_dataset = load_dataset("codeparrot/apps")
train_dataset = apps_dataset["train"]
intro_train_dataset = train_dataset.filter(lambda example: example["difficulty"] == "introductory")

# -------------------------------
# Load the code generation model and tokenizer.
code_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
code_model = AutoModelForCausalLM.from_pretrained(code_model_name, device_map="auto")


# -------------------------------
# Function to extract example input and output, modify input as required.
def modify_example_io(question):
    example_io_pattern =  re.compile(r"-----Example-----\nInput\n(.*?)\nOutput\n(.*)", re.DOTALL)
    match = example_io_pattern.search(question)
    if not match:
        print("WARNING: RETURNS ORIGINAL QUESTION")
        return question
    
    example_input, example_output = match.groups()
    lines = example_input.strip().split("\n")
    output_lines = example_output.strip().split("\n")
    
    # Extract the number of test cases
    print(len(lines), len(output_lines))
    num_test_cases = int(lines[0])
    input_pairs = [(lines[2 * i + 1], lines[2 * i + 2], output_lines[i]) for i in range(num_test_cases)]
    
    # Select 1 or 2 random test cases
    selected_pairs = random.sample(input_pairs, random.randint(1, min(2, len(input_pairs))))
    
    # Modify selected input lines with random multiplication
    modified_pairs = []
    for first_line, second_line, output_line in selected_pairs:
        numbers = second_line.split()
        modified_numbers = " ".join(str(int(num) * random.randint(1, 20)) for num in numbers)
        modified_pairs.append((first_line, modified_numbers, output_line))
    
    # Construct new input and output with modified selections
    modified_input = "\n".join([f"{p[0]}\n{p[1]}" for p in modified_pairs])
    modified_output = "\n".join([p[2] for p in modified_pairs])
    
    modified_question = question[:match.start()] + f"-----Example-----\nInput\n{len(modified_pairs)}\n{modified_input}\n\nOutput\n{modified_output}\n" + question[match.end():]
    
    return modified_question

# Please generate a complete Python script that solves the following problem. Enclose your final code within ```python and ``` markers. Please ensure that your final code includes a main block to read input and print the result. Also, ensure that your entire output is within 1024 tokens.
# -------------------------------
# Build a prompt for code generation using a given question.
def build_code_prompt(question):
    return f"""
Please generate a complete Python script that solves the following problem.Enclose your final code within ```python and ``` markers.

Problem Description:
{question}

Generated Code:
"""

# -------------------------------
# Extract code from generated text (expects code to be between ```python and ```).
def extract_code(generated_text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None

# -------------------------------
# Generate code using the code generation model.
def generate_code(prompt):
    inputs = code_tokenizer(prompt, return_tensors="pt").to(code_model.device)
    outputs = code_model.generate(
        inputs.input_ids,
        max_length=1024,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    generated_text = code_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text[len(prompt):])
    return extract_code(generated_text[len(prompt):])

# -------------------------------
# Run the generated code and record output.
def run_generated_code(code, io_pair):
    inputs_list = io_pair.get("inputs", [])
    recorded_outputs = []
    
    for inp in inputs_list:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name
        try:
            result = subprocess.run(
                ["python", tmp_filename],
                input=inp,
                text=True,
                capture_output=True,
                timeout=10
            )
            recorded_outputs.append({"input": inp, "output": result.stdout.strip()})
        except Exception as e:
            recorded_outputs.append({"input": inp, "error": str(e)})
        finally:
            os.remove(tmp_filename)
    
    return recorded_outputs

# -------------------------------
# Main procedure:
NUM_SAMPLES_ORIGINAL = 3
NUM_REPHRASES = 10
NUM_SAMPLES_PER_REPHRASE = 10

problem = intro_train_dataset[5]
io_pair = eval(problem["input_output"])
original_question = problem['question']

final_results = []

# ---- Test on the original question ----
# print("\n=== Testing on Original Question ===")
# original_prompt = build_code_prompt(original_question)
# original_results = []
# for s in range(NUM_SAMPLES_ORIGINAL):
#     print(f"\n--- Original Sample {s+1} ---")
#     while True:
#         code = generate_code(original_prompt)
#         if code != None:
#             break
#     print("Generated Code:\n", code, "\n")
#     if code:
#         recorded_outputs = run_generated_code(code, io_pair)
#         original_results.append({"code": code, "recorded_outputs": recorded_outputs})
#     else:
#         original_results.append({"code": None, "recorded_outputs": []})
# final_results.append(original_results)

# ---- Test on rephrased questions ----
for r in range(NUM_REPHRASES):
    print(f"\n=== Rephrase {r+1} ===")
    rephrased_question = modify_example_io(original_question)
    print("Rephrased Question:", rephrased_question)
    
    code_prompt = build_code_prompt(rephrased_question)
    
    rephrase_results = []
    for s in range(NUM_SAMPLES_PER_REPHRASE):
        print(f"\n--- Sample {s+1} for Rephrase {r+1} ---")
        while True:
            code = generate_code(code_prompt)
            if code != None:
                break
        print("Generated Code:\n", code, "\n")
        if code:
            recorded_outputs = run_generated_code(code, io_pair)
            rephrase_results.append({"code": code, "recorded_outputs": recorded_outputs})
        else:
            rephrase_results.append({"code": None, "recorded_outputs": []})
    final_results.append(rephrase_results)

# Save the results.
with open("final_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("\nResults saved to final_results.json")
