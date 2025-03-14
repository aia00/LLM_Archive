import re
import os
import subprocess
import tempfile
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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

# Load a separate rephrase model and tokenizer (using a hypothetical model name).
rephrase_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_name)
rephrase_model = AutoModelForCausalLM.from_pretrained(rephrase_model_name, device_map="auto")

# -------------------------------
# Function to rephrase the problem description.
def rephrase_question(question):
    prompt = f"""
Please rephrase the following problem description in different words while preserving its meaning. Don't try to interpret the question, just do words rephrasing. The input and output conditions should remain unchanged. Don't output anything irrelevent, just the rephrased question and the example input and output.

Problem Description:
{question}

Rephrased Problem:
"""
    inputs = rephrase_tokenizer(prompt, return_tensors="pt").to(rephrase_model.device)
    outputs = rephrase_model.generate(
        inputs.input_ids,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    rephrased_text = rephrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rephrased_text[len(prompt):].strip()

# -------------------------------
# Build a prompt for code generation using a given question.
def build_code_prompt(question):
    return f"""
Please generate a complete Python script that solves the following problem. Enclose your final code within ```python and ``` markers. Please ensure that your final code includes a main block to read input and print the result. Also, ensure that your entire output does not exceed 2048 tokens.

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
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    generated_text = code_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_code(generated_text)

# -------------------------------
# Test the generated code using the provided input/output test cases.
# This function writes the code to a temporary file and runs it as a subprocess.
def test_generated_code(code: str, io_pair: dict) -> float:
    # Expect io_pair to be a dict with keys "inputs" and "outputs"
    inputs_list = io_pair.get("inputs", [])
    outputs_list = io_pair.get("outputs", [])
    
    if not inputs_list or not outputs_list:
        print("No input/output test cases provided.")
        return 0.0

    passed = 0
    total = len(inputs_list)
    
    for inp, expected in zip(inputs_list, outputs_list):
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
            output = result.stdout.strip()
            if output == expected.strip():
                passed += 1
            else:
                print(f"Failed:\nInput:\n{inp}\nExpected:\n{expected.strip()}\nGot:\n{output}")
        except Exception as e:
            print(f"Runtime Error for input:\n{inp}\nError: {e}")
        finally:
            os.remove(tmp_filename)
    
    return passed / total if total > 0 else 0.0

# -------------------------------
# Main procedure:
# For a given problem, first test on the original question, then perform multiple rephrases.
NUM_SAMPLES_ORIGINAL = 10
NUM_REPHRASES = 3
NUM_SAMPLES_PER_REPHRASE = 10

# Pick one problem (e.g., problem index 5) from the filtered introductory dataset.
problem = intro_train_dataset[5]
# Convert the test case string into a dictionary (adjust if needed)
io_pair = eval(problem["input_output"])
original_question = problem['question']

# This will hold results as a list of lists:
# The first element corresponds to original question samples,
# and each subsequent element corresponds to one rephrase iteration.
final_results = []

# ---- Test on the original question ----
print("\n=== Testing on Original Question ===")
original_prompt = build_code_prompt(original_question)
original_results = []
for s in range(NUM_SAMPLES_ORIGINAL):
    print(f"\n--- Original Sample {s+1} ---")
    code = generate_code(original_prompt)
    print("Generated Code:\n", code, "\n")
    if code:
        score = test_generated_code(code, io_pair)
        original_results.append({"code": code, "success_ratio": score})
        print(f"Pass Rate: {score:.2f}")
    else:
        print("Failed to extract valid code.")
        original_results.append({"code": None, "success_ratio": -1})
final_results.append(original_results)

# ---- Test on rephrased questions ----
for r in range(NUM_REPHRASES):
    print(f"\n=== Rephrase {r+1} ===")
    rephrased_question = rephrase_question(original_question)
    print("Rephrased Question:", rephrased_question)
    os._exit()
    
    code_prompt = build_code_prompt(rephrased_question)
    
    rephrase_results = []  # List to store samples for this rephrase iteration.
    for s in range(NUM_SAMPLES_PER_REPHRASE):
        print(f"\n--- Sample {s+1} for Rephrase {r+1} ---")
        code = generate_code(code_prompt)
        print("Generated Code:\n", code, "\n")
        if code:
            score = test_generated_code(code, io_pair)
            rephrase_results.append({"code": code, "success_ratio": score})
            print(f"Pass Rate: {score:.2f}")
        else:
            print("Failed to extract valid code.")
            rephrase_results.append({"code": None, "success_ratio": -1})
    final_results.append(rephrase_results)

# Save the nested results to a JSON file.
with open("final_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

# Optionally, print out the final nested results.
print("\nFinal Results:")
for i, group in enumerate(final_results):
    label = "Original" if i == 0 else f"Rephrase {i}"
    print(f"\n{label}:")
    for sample in group:
        print(sample)

print("\nResults saved to final_results.json")
