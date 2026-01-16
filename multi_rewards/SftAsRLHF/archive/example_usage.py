# example_usage.py
# Minimal usage example for OnlineSFTTrainer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from online_sft_trainer import OnlineSFTTrainer, OnlineSFTTrainerConfig

# Models (replace with your checkpoints)
policy_name = "gpt2"
ref_name = (
    policy_name  # often initialize ref as a frozen copy of the starting policy
)
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"  # example; pick a reward model you have

tokenizer = AutoTokenizer.from_pretrained(policy_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

policy = AutoModelForCausalLM.from_pretrained(policy_name)
ref_policy = AutoModelForCausalLM.from_pretrained(ref_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name)

cfg = OnlineSFTTrainerConfig(
    beta=0.02,
    max_new_tokens=128,
    temperature=1.0,
    top_p=0.95,
    do_sample=True,
    num_return_sequences=1,
    lr=1e-5,
    grad_clip_norm=1.0,
    use_amp=False,
    denom_eps=1e-8,
)

trainer = OnlineSFTTrainer(
    policy=policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    tokenizer=tokenizer,
    cfg=cfg,
)

# Dummy prompt batch
prompt_batch = [
    "Explain Newton's second law in simple terms.",
    "Write a friendly email to thank a colleague for their help.",
    "Give me a few dinner ideas for a vegetarian diet.",
]

metrics = trainer.step(prompt_batch)
print(metrics)
