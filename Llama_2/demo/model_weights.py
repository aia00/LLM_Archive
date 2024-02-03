from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'

# Load our model from local
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME+'_model')

# And its associated tokenizer from local
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME+'_tokenizer')

print(model)