import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pydantic import BaseModel, Field
import nest_asyncio
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import uvicorn

# -----------------------------
# Request schema
# -----------------------------
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Bangla text to be translated")
    target_language: str = Field(..., description="Target language for translation")

# -----------------------------
# Prompts
# -----------------------------
system_prompt = """You are a professional translator.
Translate the Bengali text provided by the user into English.
Requirements:
- Provide only the translation, no extra explanations.
When the user provides Bengali text, respond only with the English translation.
"""

# -----------------------------
# Model setup
# -----------------------------
model_name = "unsloth/gemma-2-9b-bnb-4bit"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    offload_folder="offload"
)


model.eval()

device = next(model.parameters()).device

# -----------------------------
# Translation function
# -----------------------------
def translate_bangla_to_english(text: str) -> str:
    prompt = f"System: {system_prompt}\nUser: {text}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)

    # Extract only assistant's reply
    translation = output.split("Assistant:")[-1].strip()
    if "\n" in translation:
        translation = translation.split("\n")[0]

    return translation

# -----------------------------
# Test run
# -----------------------------
# if __name__ == "__main__":
#     test_text = "জমিয়তে উলামায়ে ইসলাম বাংলাদেশ হলো বাংলাদেশের একটি ইসলামপন্থী রাজনৈতিক দল"
#     print(translate_bangla_to_english(test_text))



load_dotenv()

api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Set OpenAI API key
openai.api_key = api_key

logger.info("API key set")  


app = FastAPI(
    title="Bangla to English Translator API",
    description="API for translating Bangla text to English with accurate output",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Wellcome to the translator",
            "status": "active",
            "endpoints": {
                "/translate": "POST - Translate Bangla text to English",
                "/docs": "GET - API documentation"
            }
            
            }


@app.post("/translate", response_model=Dict[str, Any])
async def text_translate(req: TranslationRequest):
    try:
        if not req.text.strip():
            translated_text = translate_bangla_to_english(req.text)
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        
        return {
            "translation": translated_text,
            # "original_text": req.text,
            "status": "success"
        }
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    # print("\n" + "="*50)
    print("Starting Bangla to English Translator API")
    print("="*50)
    print("Server will run on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)







