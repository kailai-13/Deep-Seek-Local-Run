from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = "Qwen/Qwen1.5-1.8B"  # Smaller model for GTX 1650

# Auto-detect device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # Automatically handles GPU/CPU offloading
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    print(f"✅ Model loaded successfully on {device}!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Define request model
class PromptRequest(BaseModel):
    text: str

# Optimized generation function
def generate_response(prompt: str, max_new_tokens=256):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Limit input length
        ).to(model.device)  # Use the model's device
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=30,  # Reduced for memory savings
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # Prevent repetition
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise HTTPException(status_code=500, detail="CUDA out of memory - try shorter prompt")
        raise

# API Endpoint
@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    try:
        response = generate_response(prompt_request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))