#!/usr/bin/env python
import re
import os
import subprocess
import tempfile
import json
import random
import torch
import sys
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# -----------------------------------------------------------------------------
# 1) PARSE QUESTION
# -----------------------------------------------------------------------------
def parse_question(text: str):
    """
    Returns a dictionary:
      {
        "question_desc": str,
        "question_input": str,
        "question_output": str,
        "examples": [
          {"example_input": str, "example_output": str},
          ...
        ],
        "note": str
      }
    """
    # 1) question_desc (before -----Input-----)
    qdesc_match = re.search(r"^(.*?)-----Input-----", text, re.DOTALL)
    question_desc = qdesc_match.group(1).strip() if qdesc_match else text.strip()

    # 2) question_input (between -----Input----- and -----Output-----)
    qinput_match = re.search(
        r"-----Input-----\s*(.*?)(?=-----Output-----|-----Example(?:s)?-----|-----Note-----|$)",
        text, re.DOTALL)
    question_input = qinput_match.group(1).strip() if qinput_match else ""

    # 3) question_output (between -----Output----- and -----Example----- or end)
    qoutput_match = re.search(
        r"-----Output-----\s*(.*?)(?=-----Example(?:s)?-----|-----Note-----|$)",
        text, re.DOTALL)
    question_output = qoutput_match.group(1).strip() if qoutput_match else ""

    # 4) note: after -----Note-----
    note_match = re.search(r"-----Note-----\s*(.*)", text, re.DOTALL)
    note_section = note_match.group(1).strip() if note_match else ""

    # 5) examples block: after -----Example----- or -----Examples-----
    ex_block_match = re.search(r"-----Example(?:s)?-----\s*(.*?)(?=-----Note-----|$)",
                               text, re.DOTALL)
    ex_block_text = ex_block_match.group(1).strip() if ex_block_match else ""

    # parse multiple Input...Output pairs from ex_block_text
    # We'll do a pattern for repeated pairs: Input\s*(.*?)(?=Output|$)Output\s*(.*?)(?=Input|$)
    pattern_io = r"Input\s*(.*?)(?=Output|$)Output\s*(.*?)(?=Input|$)"
    examples = []
    for match in re.finditer(pattern_io, ex_block_text, flags=re.DOTALL):
        ex_in = match.group(1).strip()
        ex_out = match.group(2).strip()
        examples.append({"example_input": ex_in, "example_output": ex_out})

    return {
        "question_desc": question_desc,
        "question_input": question_input,
        "question_output": question_output,
        "examples": examples,
        "note": note_section
    }

# -----------------------------------------------------------------------------
# 2) PROMPT BUILDERS
# -----------------------------------------------------------------------------
def build_code_prompt(q_desc, q_input, q_output):
    """
    Combine question_desc, question_input, question_output
    so the code sees the entire problem statement.
    """
    full_problem_text = q_desc
    if q_input.strip():
        full_problem_text += "\n\nInput Format:\n" + q_input
    if q_output.strip():
        full_problem_text += "\n\nOutput Format:\n" + q_output

    return f"""
Please generate a complete Python script that solves the following problem.
Enclose your final code within ```python and ``` markers. 
Make sure the Python code contains a main function.

Problem Description:
{full_problem_text}

Generated Code:
"""

def build_test_input_prompt(q_input, ex_input):
    return f"""
Please generate another input strictly following the format described.

Here is the input format:
{q_input}

Here is the example input:
{ex_input}

Do not include the example input exactly; make the data random. 
Output only the generated input, line by line, with no explanations.
"""

# -----------------------------------------------------------------------------
# 3) GENERIC TEXT GENERATION
# -----------------------------------------------------------------------------
def generate_text(prompt, tokenizer, model,
                  max_length=512, temperature=0.7, top_p=0.95, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------------------------------------------------------
# 4) EXTRACT CODE & BACKTICK-INPUT
# -----------------------------------------------------------------------------
def extract_code(generated_text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    return matches[-1].strip() if matches else None

def extract_input_from_backticks(generated_text: str) -> str:
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, generated_text, re.DOTALL)
    return matches[-1].strip() if matches else None

# -----------------------------------------------------------------------------
# 5) CODE/TEST GENERATION
# -----------------------------------------------------------------------------
def generate_code(q_desc, q_input, q_output, code_tokenizer, code_model):
    prompt = build_code_prompt(q_desc, q_input, q_output)
    raw_generation = generate_text(
        prompt,
        tokenizer=code_tokenizer,
        model=code_model,
        max_length=1024*4,
        temperature=0.5,
        top_p=0.9,
        do_sample=True
    )
    return extract_code(raw_generation[len(prompt):])

def generate_single_test_input(q_input, ex_input, test_input_tokenizer, test_input_model):
    prompt = build_test_input_prompt(q_input, ex_input)
    raw_generation = generate_text(
        prompt,
        tokenizer=test_input_tokenizer,
        model=test_input_model,
        max_length=512,
        temperature=0.4,
        top_p=0.9,
        do_sample=True
    )
    relevant_part = raw_generation[len(prompt):]
    return extract_input_from_backticks(relevant_part)

# -----------------------------------------------------------------------------
# 6) RUN CODE
# -----------------------------------------------------------------------------
def run_code_with_input(code_str, single_input):
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str)
        tmp_filename = tmp.name
    try:
        result = subprocess.run(
            ["python", tmp_filename],
            input=single_input,
            text=True,
            capture_output=True,
            timeout=10
        )
        return (result.stdout.strip(), result.stderr)
    except Exception as e:
        return (None, str(e))
    finally:
        os.remove(tmp_filename)

# -----------------------------------------------------------------------------
# 7) MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=5,
                        help="Index of the problem in the intro_train_dataset.")
    args = parser.parse_args()

    # 1) Load APPS dataset
    apps_dataset = load_dataset("codeparrot/apps")
    train_dataset = apps_dataset["test"]
    intro_train_dataset = train_dataset.filter(lambda ex: ex["difficulty"] == "introductory")

    if args.index < 0 or args.index >= len(intro_train_dataset):
        print(f"Question index {args.index} out of range. Found {len(intro_train_dataset)} items.")
        sys.exit(1)

    # Grab problem & parse
    problem = intro_train_dataset[args.index]
    text = problem["question"]
    solutions = eval(problem.get("solutions", "[]"))
    if not solutions:
        print(f"No standard solution found for question index {args.index}")
        sys.exit(0)
    standard_solution = solutions[0]

    parsed = parse_question(text)
    q_desc = parsed["question_desc"]
    q_input = parsed["question_input"]
    q_output = parsed["question_output"]
    examples_list = parsed["examples"]
    note_section = parsed["note"]

    # Display them
    print("-----QUESTION DESCRIPTION-----")
    print(q_desc)
    print("\n-----QUESTION INPUT FORMAT-----")
    print(q_input)
    print("\n-----QUESTION OUTPUT FORMAT-----")
    print(q_output)
    print("\n-----MULTIPLE EXAMPLES-----")
    for i, e in enumerate(examples_list, 1):
        print(f"Example {i}:\nInput:\n{e['example_input']}\nOutput:\n{e['example_output']}")
    print("\n-----NOTE-----")
    print(note_section)

    # 2) Load code & test input models
    code_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
    code_model = AutoModelForCausalLM.from_pretrained(code_model_name, device_map="auto")

    test_input_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    test_input_tokenizer = AutoTokenizer.from_pretrained(
        test_input_model_name, use_auth_token=True
    )
    test_input_model = AutoModelForCausalLM.from_pretrained(
        test_input_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=True
    )

    # Setup
    NUM_CODE_SAMPLES = 10
    NUM_TEST_INPUTS = 10

    code_fail_count = 0
    input_fail_count = 0

    # We'll use the *first* example input for verifying code, if it exists.
    # If no examples, fallback to an empty string or skip code checking logic
    example_input_text = examples_list[0]["example_input"] if examples_list else ""

    # 3) Generate code solutions that pass the first example input
    code_variants = []
    while len(code_variants) < NUM_CODE_SAMPLES:
        candidate_code = generate_code(q_desc, q_input, q_output, code_tokenizer, code_model)
        if not candidate_code:
            code_fail_count += 1
            print("Empty snippet, re-generating code.")
            if code_fail_count > 2 * NUM_CODE_SAMPLES:
                print("[FATAL] Exceeded maximum code generation failures.")
                sys.exit(1)
            continue

        out_ex, err_ex = run_code_with_input(candidate_code, example_input_text)
        if out_ex=='' or len(err_ex)>5:
            code_fail_count += 1
            print("Code snippet error or empty output on example input, re-generating.")
            if code_fail_count > 2 * NUM_CODE_SAMPLES:
                print("[FATAL] Exceeded maximum code generation failures.")
                sys.exit(1)
            continue

        code_variants.append(candidate_code)
        print(f"[Accepted code snippet #{len(code_variants)} that passes the first example input.]")

    # 4) Generate test inputs that pass the standard solution
    accepted_inputs = []
    standard_outputs = []
    while len(accepted_inputs) < NUM_TEST_INPUTS:
        new_inp = generate_single_test_input(q_input, example_input_text,
                                             test_input_tokenizer, test_input_model)
        if not new_inp:
            input_fail_count += 1
            print("No triple-backtick input found, discarding.")
            if input_fail_count > 4 * NUM_TEST_INPUTS:
                print("[FATAL] Exceeded maximum input generation failures.")
                sys.exit(1)
            continue

        out_std, err_std = run_code_with_input(standard_solution, new_inp)
        if out_std=='' or len(err_std)>5:
            input_fail_count += 1
            print("Standard solution error or empty output with new test input, discarding.")
            if input_fail_count > 4 * NUM_TEST_INPUTS:
                print("[FATAL] Exceeded maximum input generation failures.")
                sys.exit(1)
            continue

        accepted_inputs.append(new_inp)
        standard_outputs.append(out_std)
        print(f"[Accepted test input #{len(accepted_inputs)}]\n{new_inp}")
        print("std output =>", out_std)

    # 5) Evaluate final code solutions
    all_results = []
    for idx, code_str in enumerate(code_variants, start=1):
        code_run_results = []
        for inp_str, out_sol in zip(accepted_inputs, standard_outputs):
            out_m, err_m = run_code_with_input(code_str, inp_str)
            code_run_results.append({
                "input": inp_str,
                "standard_solution_output": out_sol,
                "model_output": out_m or "",
                "error": err_m or None
            })
        all_results.append({
            "code_index": idx,
            "code": code_str,
            "test_results": code_run_results
        })

    # 6) Save
    os.makedirs("results_test", exist_ok=True)
    result_filename = f"results_test/final_results_q{args.index}.json"
    with open(result_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n[Done] {result_filename} saved. Code uses (desc + input + output), multiple examples are parsed, "
          f"and code generation / input generation have failure caps set as requested.")

if __name__ == "__main__":
    main()
