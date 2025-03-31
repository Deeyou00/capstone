import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def apply_lora(base_model_path, lora_path, output_path):
    # 加载基础模型的tokenizer和模型
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    
    # 加载LoRA模型
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )
    
    # 应用LoRA并卸载LoRA模型
    model = lora_model.merge_and_unload()
    
    # 保存合并后的模型
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)
    return model

'''
base_model_path = "google/gemma-2-9b-it"
# LoRA模型路径
lora_path = "/root/users/jusjus/Self/LLaMA-Factory/saves/Gemma-2-9B-Instruct/lora/train_2024-11-23-04-13-58"
# 输出路径
output_path = "jusjus/Gemma-2-9B-Instruct-Lora"

lora_model = apply_lora(base_model_path, lora_path, output_path)
'''
