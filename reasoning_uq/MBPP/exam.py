# %%
import re
from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset
import contextlib
from io import StringIO
import os

# %%
# Load MBPP dataset
mbpp_dataset = load_dataset("mbpp")

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# %%
# Construct prompt
def build_prompt(problem):
    return f"""
Please generate a complete Python function based on the following problem description. Ensure the code passes the provided test cases. Always include the generated code between ```python and ```.

Problem Description:
{problem['text']}

Test Cases:
{problem['test_list']}

Generated Code:
"""

# %%
def extract_code(generated_text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    if len(matches) >= 2:
        return matches[1]
    else:
        # Fallback: if no second code block is found, return False or handle appropriately
        return False


# %%
# Generate code and extract the code block
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        temperature=1,
        top_p=0.8,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    code = extract_code(generated_text)
    return code

# %%
def test_generated_code(code: str, test_cases: list) -> float:
    # Use an isolated namespace for each sample
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Code execution error: {e}")
        return 0.0

    total = len(test_cases)
    passed = 0

    for test in test_cases:
        try:
            # Run the test code which contains an assert statement.
            # If the assert condition is true, no exception is thrown.
            # exec('print(reverse_words("python program"))', namespace)
            # exec('print(reverse_words("python program")=="program python")', namespace)
            exec(test, namespace)
            passed += 1
        except AssertionError:
            # The assert failed, so the test did not pass.
            print(f"Test failed (assertion error): {test}")
        except Exception as e:
            # Any other exception that may occur
            print(f"Test failed: {test}, error: {e}")

    return passed / total if total > 0 else 0.0


# %%
# Sample 10 times for one problem and record results
problem = mbpp_dataset["train"][13]
prompt = build_prompt(problem)
test_cases = problem["test_list"]

results = []

for i in range(10):
    print(f"Sample {i+1}:")
    generated_code = generate_code(prompt)
    if generated_code == False:
        print("Cannot extract the python code!")
        results.append({"code": generated_code, "success_ratio": -1})
        continue
    print(generated_code)
    # os._exit(0)
    ratio = test_generated_code(generated_code, test_cases)
    results.append({"code": generated_code, "success_ratio": ratio})
    print(f"Success Ratio: {ratio:.2f}\n")

print("All Sample Results:")
for idx, res in enumerate(results):
    print(f"Sample {idx+1}: Success Ratio: {res['success_ratio']:.2f}")
    print("Generated Code:")
    print(res['code'])
    print("-" * 40)


# %%
s_values = [result["success_ratio"] for result in results]
print(s_values)


