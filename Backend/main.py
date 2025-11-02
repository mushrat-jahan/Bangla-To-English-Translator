import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# import translator 
from loguru import logger
import nest_asyncio
from fastapi import FastAPI
from typing import Dict, Any
from groq import Groq
import uvicorn
from my_translate import TranslationRequest, translate_bangla_to_english


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables. Please check your .env file.")

# Set OpenAI API key
# openai.api_key = api_key
print(api_key)
logger.info("API key set")  

app = FastAPI(
    title="Bangla to English Translator API",
    description="API for translating Bangla text to English with accurate output",
    version="1.0.0"
)

# CORS middleware - allows frontend to connect
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
                # "/health": "GET - Check API health",
                "/docs": "GET - API documentation"
            }
            
            }


@app.post("/translate", response_model=Dict[str, Any])

async def text_translate(req: TranslationRequest): 
    print("Before sending to translation module", req)        
    translated_text = translate_bangla_to_english(req.text)
    # print("Translation done", translated_text) 
    return {
        "translation": translated_text,
        # "original_text": req.text,
        "status": "success"
    }
   
# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    print("Starting Bangla to English Translator API")
    print("="*50)
    print("Server will run on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
