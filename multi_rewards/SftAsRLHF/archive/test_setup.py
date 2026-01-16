#!/usr/bin/env python3
# test_setup.py
# Quick test to verify models load and trainer initializes correctly

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from online_sft_trainer import OnlineSFTTrainer, OnlineSFTTrainerConfig

def test_model_loading():
    """Test that models can be loaded from HuggingFace."""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)

    model_path = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    reward_path = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

    print(f"\n1. Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped", padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"   ✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

    print(f"\n2. Loading policy model from {model_path}...")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    policy.resize_token_embeddings(len(tokenizer))
    print(f"   ✓ Policy loaded. Params: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

    print(f"\n3. Loading reference model from {model_path}...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    ref_policy.resize_token_embeddings(len(tokenizer))
    print(f"   ✓ Reference loaded. Params: {sum(p.numel() for p in ref_policy.parameters()) / 1e6:.1f}M")

    print(f"\n4. Loading reward model from {reward_path}...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        device_map="cpu",
    )
    print(f"   ✓ Reward model loaded. Params: {sum(p.numel() for p in reward_model.parameters()) / 1e6:.1f}M")

    return tokenizer, policy, ref_policy, reward_model


def test_trainer_init(tokenizer, policy, ref_policy, reward_model):
    """Test that trainer initializes correctly."""
    print("\n" + "=" * 60)
    print("Testing Trainer Initialization")
    print("=" * 60)

    cfg = OnlineSFTTrainerConfig(
        beta=0.1,
        max_new_tokens=53,
        temperature=1.0,
        top_p=0.9,
        lr=3e-6,
        use_amp=False,
    )

    print("\n1. Initializing OnlineSFTTrainer...")
    trainer = OnlineSFTTrainer(
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    print("   ✓ Trainer initialized successfully")

    return trainer


def test_forward_pass(trainer):
    """Test a single forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    prompt_batch = [
        "Summarize this: A simple test prompt.",
        "TLDR of: Another test prompt for the model.",
    ]

    print(f"\n1. Running trainer.step() with {len(prompt_batch)} prompts...")
    print("   (This will generate responses and compute loss)")

    try:
        with torch.no_grad():
            metrics = trainer.step(prompt_batch)

        print("   ✓ Forward pass completed successfully!")
        print("\n2. Metrics returned:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")

        # Check for warnings
        warnings = [k for k in metrics.keys() if k.startswith("warn/")]
        if warnings:
            print(f"\n   ⚠ Warnings detected: {warnings}")
        else:
            print("\n   ✓ No warnings detected")

        return True
    except Exception as e:
        print(f"   ✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    try:
        from datasets import load_dataset

        print("\n1. Loading TLDR dataset (validation split, first 10 samples)...")
        dataset = load_dataset(
            "trl-internal-testing/tldr-preference-sft-trl-style",
            split="validation[:10]",
        )
        print(f"   ✓ Dataset loaded. {len(dataset)} samples")

        print("\n2. Inspecting first sample:")
        sample = dataset[0]
        print(f"   Keys: {list(sample.keys())}")
        if "messages" in sample:
            print(f"   Messages: {len(sample['messages'])} turns")
            print(f"   First message role: {sample['messages'][0].get('role', 'N/A')}")
            print(f"   Prompt length: {len(sample['messages'][0]['content'])} chars")

        return True
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("ONLINE SFT TRAINER - SETUP VERIFICATION")
    print("=" * 60)
    print("\nThis script verifies that all components can be loaded and")
    print("the trainer can execute a forward pass correctly.")
    print("\nNote: This test runs on CPU and may be slow.")

    # Test 1: Dataset
    print("\n" + "─" * 60)
    dataset_ok = test_dataset_loading()

    # Test 2: Models
    print("\n" + "─" * 60)
    try:
        tokenizer, policy, ref_policy, reward_model = test_model_loading()
        models_ok = True
    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        models_ok = False

    if not models_ok:
        print("\n" + "=" * 60)
        print("RESULT: FAILED - Could not load models")
        print("=" * 60)
        return

    # Test 3: Trainer
    print("\n" + "─" * 60)
    try:
        trainer = test_trainer_init(tokenizer, policy, ref_policy, reward_model)
        trainer_ok = True
    except Exception as e:
        print(f"\n✗ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        trainer_ok = False

    if not trainer_ok:
        print("\n" + "=" * 60)
        print("RESULT: FAILED - Could not initialize trainer")
        print("=" * 60)
        return

    # Test 4: Forward pass
    print("\n" + "─" * 60)
    forward_ok = test_forward_pass(trainer)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Dataset loading:    {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"  Model loading:      {'✓ PASS' if models_ok else '✗ FAIL'}")
    print(f"  Trainer init:       {'✓ PASS' if trainer_ok else '✗ FAIL'}")
    print(f"  Forward pass:       {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print("=" * 60)

    if all([dataset_ok, models_ok, trainer_ok, forward_ok]):
        print("\n✓ ALL TESTS PASSED - Ready to start training!")
        print("\nRun training with:")
        print("  python train_online_sft.py [args...]")
        print("  or")
        print("  bash run_online_sft.sh")
    else:
        print("\n✗ SOME TESTS FAILED - Please fix issues before training")

    print()


if __name__ == "__main__":
    main()
