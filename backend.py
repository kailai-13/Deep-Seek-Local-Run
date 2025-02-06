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
MODEL_NAME = "Qwen/Qwen2.5-Math-7B"

# Auto-detect device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer with optimized settings
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",   # Automatically allocate to CPU/GPU
        torch_dtype=torch.float16,  # Reduce memory usage
        low_cpu_mem_usage=True  # Optimize CPU memory usage
    ).to(device)
    
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define request model
class PromptRequest(BaseModel):
    text: str

# Generate response function
def generate_response(prompt: str, max_new_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # ✅ Fix attention mask warning
            max_new_tokens=max_new_tokens,  # ✅ Use max_new_tokens instead of max_length
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id  # ✅ Set pad_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API Endpoint
@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    try:
        response = generate_response(prompt_request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))