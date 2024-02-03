from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'

# Load our model from local
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME+'_model')

# And its associated tokenizer from local
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME+'_tokenizer')

input_ids = tokenizer.encode("Hello, world!", add_special_tokens=True) # encode the text
input_ids = torch.tensor([input_ids]) # convert to tensor

with torch.no_grad():
    outputs = model(input_ids)

# `outputs` is a tuple, we only care about the first element, which is the logits
logits = outputs[0]

print(logits)