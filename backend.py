from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
MODEL_NAME = "Qwen/Qwen2.5-Math-7B"

# Quantization configuration for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically handles GPU/CPU offloading
        trust_remote_code=True
    )
    
    # For better memory management
    model.config.use_cache = True
    model.eval()
    print("✅ Model loaded successfully with 4-bit quantization!")

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
        ).to(model.device)
        
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