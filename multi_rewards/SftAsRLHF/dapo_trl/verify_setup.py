"""
Verification script for DAPO implementation.
Tests all components before running full training.
"""

import sys
sys.path.insert(0, '..')

print("=" * 80)
print("DAPO Implementation Verification")
print("=" * 80)

# Test 1: Reward function
print("\n[Test 1/3] Testing reward function (math_dapo alignment)...")
from reward_function import compute_score, compute_dapo_reward

test_cases = [
    # Valid "Answer:" pattern matches (case-insensitive)
    ("Answer: 42", "42", True),
    ("answer: 3.14", "3.14", True),
    ("ANSWER: 100", "100", True),
    # Invalid patterns (no "Answer:" prefix)
    ("42", "24", False),
    ("The solution is: -5.5", "-5.5", False),
    # Boxed answers no longer extracted (matching verl)
    ("\\boxed{100}", "100", False),
    # Numeric fallback removed (matching verl)
    ("Answer: 1/2", "0.5", False),
    # But exact string match works
    ("Answer: 0.5", "0.5", True),
]

all_passed = True
for solution, ground_truth, expected in test_cases:
    result = compute_score(solution, ground_truth)["acc"]
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"  {status} compute_score('{solution[:30]}...', '{ground_truth}') = {result} (expected {expected})")

# Quick check of overlong penalty application path
rewards = compute_dapo_reward(
    completions=[{"content": "short"}, {"content": "x" * 10_000}],
    prompts=["", ""],
    ground_truths=["[INVALID]", "[INVALID]"],
    tokenizer=None,
    max_response_length=20,
    overlong_buffer_len=5,
    overlong_penalty_factor=1.0,
    apply_overlong_penalty=True,
)
if rewards[0] == -1.0 and rewards[1] < -1.0:
    print("  ✓ Overlong penalty path exercised")
else:
    print("  ✗ Overlong penalty path failed!")
    all_passed = False

if all_passed:
    print("  ✓ All reward function tests passed!")
else:
    print("  ✗ Some reward function tests failed!")
    sys.exit(1)

# Test 2: Dataset loading
print("\n[Test 2/3] Testing dataset loading...")
from dataset_utils import prepare_dapo_dataset, format_prompt_for_chat

try:
    train_ds, val_ds, reward_lookup = prepare_dapo_dataset(max_train_samples=5)
    print(f"  ✓ Loaded {len(train_ds)} training samples")
    print(f"  ✓ Reward lookup size: {len(reward_lookup)}")

    example = train_ds[0]
    # Format prompt for display regardless of type
    prompt_display = (
        example["prompt"] if isinstance(example["prompt"], str) else format_prompt_for_chat(example["prompt"])  # type: ignore
    )
    print(f"  ✓ Example prompt: {prompt_display[:80]}...")
    print(f"  ✓ Example ground truth: {example['ground_truth']}")

    # Verify reward lookup works
    prompt_value = example["prompt"]
    if isinstance(prompt_value, str):
        if prompt_value in reward_lookup:
            print("  ✓ Reward lookup working correctly for string prompts")
        else:
            print("  ✗ Reward lookup missing ground truth for string prompt!")
            sys.exit(1)
    else:
        # Conversational prompts (list/dict) are not hashable; lookup is intentionally empty
        print("  ✓ Conversational prompt detected; reward will read ground_truth from kwargs")

except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("  ✓ Dataset loading tests passed!")

# Test 3: TRL imports
print("\n[Test 3/3] Testing TRL imports...")
try:
    from trl import GRPOTrainer, GRPOConfig
    print("  ✓ GRPOTrainer imported successfully")
    print("  ✓ GRPOConfig imported successfully")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("  ✓ AutoTokenizer imported successfully")
    print("  ✓ AutoModelForCausalLM imported successfully")

    print("  ✓ All required imports successful!")

except Exception as e:
    print(f"  ✗ Import failed: {e}")
    print("\nPlease make sure you have installed:")
    print("  - trl")
    print("  - transformers")
    print("  - datasets")
    print("  - accelerate")
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("✓ All verification tests passed!")
print("=" * 80)
print("\nYou can now run training with:")
print("  ./test_dapo.sh      # Quick test with small model")
print("  ./run_dapo.sh       # Full training")
print("=" * 80)
