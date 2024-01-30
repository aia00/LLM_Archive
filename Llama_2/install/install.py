from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'

# aquire your token here: https://huggingface.co/settings/tokens
access_token = "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"

# Download the model and its weights
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=access_token)

# Save the model and its weights in a directory with the same name as the model
model.save_pretrained(MODEL_NAME+'_model', token=access_token)

# Download the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)

# Save the tokenizer
tokenizer.save_pretrained(MODEL_NAME+'_tokenizer', token=access_token)



