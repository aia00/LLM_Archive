from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 需替换为实际模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto")  # 自动分配设备)

# 将模型设置为推理模式
model.eval()


prompt = "你好，DeepSeek-R1-Distill-Llama-8B！"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


# 生成文本（示例参数）
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)