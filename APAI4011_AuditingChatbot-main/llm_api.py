<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, File, UploadFile, Body, Query
from fastapi import UploadFile, File, HTTPException
import os
=======
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
>>>>>>> 2ffd424 (Initial commit - reset history)
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from starlette.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
<<<<<<< HEAD
from QARAG.qa_rag import qa_rag_run  # Import the RAG function
=======
>>>>>>> 2ffd424 (Initial commit - reset history)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/models", StaticFiles(directory="models"), name="models")

# Global variables to store the model and tokenizer
global_tokenizer = None
global_model = None

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


@app.post("/generate/")
async def generate_text(input_text: InputText = Body(...)):
    try:
        generated_text = infer_from_model(global_model, global_tokenizer, input_text.text, use_config=False)
        generated_text = generated_text[len(input_text.text):].replace("\n", " ")
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), model_dir: str = "/root/users/jusjus/Self/LLaMA-Factory/jusjus/Gemma_9B_QA3000"):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")

        return {"generated_text": "True"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/init-model/{model_path}")
async def init_model(model_path: str):
    try:
        global global_tokenizer, global_model
        model_path = f"/root/users/jusjus/Self/LLaMA-Factory/jusjus/{model_path}"
        global_tokenizer, global_model = load_model(model_path)
        return {"message": "Model and tokenizer initialized successfully", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
<<<<<<< HEAD

@app.post("/QARAG/")
async def qarag_endpoint(question: str = Query(...), pdf_path: str = Query(...)):
    """
    Endpoint to handle question and PDF path for QARAG.
    Calls `qa_rag_run(pdf_path, question)` and returns the result.
    """
    try:
        # Call the RAG function
        print(pdf_path.strip("/"), question)
        result = qa_rag_run(pdf_path.strip("/"), question)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")

@app.post("/upload")
async def post_upload_file(file: UploadFile):
    file_pth = file.filename
    print(file_pth)
    with open(f"Data/uploaded.pdf", "wb") as F:
        F.write(await file.read())

    return {"message": "File upload successfully"}

=======
    
>>>>>>> 2ffd424 (Initial commit - reset history)
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)