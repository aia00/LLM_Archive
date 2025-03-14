# %%
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import subprocess
import tempfile

# %%
# Load APPS dataset and filter for introductory difficulty
apps_dataset = load_dataset("codeparrot/apps")
train_dataset = apps_dataset["train"]

# Filter introductory problems
intro_train_dataset = train_dataset.filter(lambda example: example["difficulty"] == "introductory")

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# %%
def build_prompt(problem):
    return f"""
Please generate a complete Python script that solves the following problem. Enclose your final code within ```python and ``` markers. Also, ensure that your entire output (including this prompt) does not exceed 2048 tokens.

Problem Description:
{problem['question']}

Generated Code:
"""

# %%
# Extract code from generated text (code must be enclosed in ```python and ```)
def extract_code(generated_text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    if len(matches) >= 2:
        return matches[-1]
    else:
        # Fallback: if no second code block is found, return False or handle appropriately
        return False


# Generate code using the prompt
def generate_code(prompt):
    # print(prompt)
    # os._exit()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)
    # os._exit()
    return extract_code(generated_text).strip()


# %%
# Test the generated code using the provided input/output.
# This function writes the code to a temporary file and runs it as a subprocess.
def test_generated_code(code: str, io_pair: dict) -> float:
    # Expect io_pair to be a dictionary with keys "inputs" and "outputs"
    inputs_list = io_pair.get("inputs", [])
    outputs_list = io_pair.get("outputs", [])
    
    if not inputs_list or not outputs_list:
        print("No input/output test cases provided.")
        return 0.0

    passed = 0
    total = len(inputs_list)
    
    for inp, expected in zip(inputs_list, outputs_list):
        # Write the code to a temporary file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_filename = tmp.name
        try:
            # Run the temporary Python file with the test input.
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
                print(f"Failed: \nInput:\n{inp}\nExpected:\n{expected.strip()}\nGot:\n{output}")
        except Exception as e:
            print(f"Runtime Error for input:\n{inp}\nError: {e}")
        finally:
            os.remove(tmp_filename)
    
    return passed / total if total > 0 else 0.0

# %%
# Pick one problem (e.g., problem index 5) from the filtered introductory dataset.
problem = intro_train_dataset[5]
# Get the input/output test cases (this record stores them as a dict with "inputs" and "outputs")
io_pair = eval(problem["input_output"])

# For debugging, print the test case structure.
print("Input/Output test case:")
# print(io_pair)

prompt = build_prompt(problem)
print(prompt)

results = []

# Sample the answer 10 times for the same problem.
for i in range(10):
    print(f"\n=== Sample {i+1} ===")
    code = generate_code(prompt)
    print("Generated Code:\n", code, "\n")
    if code:
        score = test_generated_code(code, io_pair)
        results.append({"code": code, "success_ratio": score})
        print(f"Pass Rate: {score:.2f}")
    else:
        print("Failed to extract valid code.")
        results.append({"code": None, "success_ratio": -1})
    print("=" * 50)

# Optionally, print out all success ratios.
s_values = [result["success_ratio"] for result in results]
print("Success Ratios:", s_values)


