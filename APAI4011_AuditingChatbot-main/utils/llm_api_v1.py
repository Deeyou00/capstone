from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

app = FastAPI()

class InputText(BaseModel):
    text: str

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
    if use_config:
        with open(config_path, 'r') as f:
            gen_kwargs_all = json.load(f)
    else:
        gen_kwargs_all = {
            "max_length": 512,
            "top_p": 0.38,
            "temperature": 0.85,
        }
    
    # 编码输入文本
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    # 创建注意力掩码
    attention_mask = torch.ones(input_ids['input_ids'].size(), dtype=torch.long).to("cuda")
    
    # 添加必要的参数
    gen_kwargs_all["attention_mask"] = attention_mask

    # 使用模型生成文本
    outputs = model.generate(input_ids=input_ids['input_ids'], **gen_kwargs_all)
    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 在全局范围内初始化模型和分词器
model_dir = "/root/users/jusjus/Self/LLaMA-Factory/jusjus/Gemma_9B_QA3000"
tokenizer, model = load_model(model_dir)

@app.post("/generate/")
async def generate_text(input_text: InputText):
    try:
        generated_text = infer_from_model(model, tokenizer, input_text.text, use_config=False)
        generated_text = generated_text[len(input_text.text):].replace("\n", " ")
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)
