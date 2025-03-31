import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 加载模型和分词器
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
    )
    return tokenizer, model

# 使用模型进行推理
def infer_from_model(model, tokenizer, input_text, use_config=True, config_path="/root/users/jusjus/Self/LLaMA-Factory/jusjus/default_config.json"):
    # 从 JSON 文件中加载生成参数
    if use_config == True:
        with open(config_path, 'r') as f:
            gen_kwargs_all = json.load(f)
    else:
        gen_kwargs_all = {
            "max_length": 512,
            "top_p": 0.38,
            "temperature": 0.85,
        }
    
    # 编码输入文本
    input_text_len = len(input_text)
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    # 创建注意力掩码
    attention_mask = torch.ones(input_ids['input_ids'].size(), dtype=torch.long).to("cuda")
    
    # 添加必要的参数
    gen_kwargs_all["attention_mask"] = attention_mask

    # 使用模型生成文本
    outputs = model.generate(input_ids=input_ids['input_ids'], **gen_kwargs_all)
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[input_text_len:]
    return generated_text

'''
# 使用示例
model_dir = "/root/users/jusjus/Self/LLaMA-Factory/jusjus/Gemma_9B_QA3000"
input_text = "What is the significance of audit committees in corporate governance?"
#config_path = "generation_config.json"

# 加载模型和分词器
tokenizer, model = load_model(model_dir)

# 使用模型进行推理
generated_text = infer_from_model(model, tokenizer, input_text, use_config=False)
print("Generated Text:", generated_text)
'''
