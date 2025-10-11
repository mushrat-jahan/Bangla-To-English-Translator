import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# import translator 
from loguru import logger
import nest_asyncio
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from unsloth import FastLanguageModel
import uvicorn
from Backend.my_translate import TranslationRequest, translate_bangla_to_english


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



    # translated_text = translate_bangla_to_english(req.text)
    # return {"translation":translated_text}

# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy",
#         "model": model_name,
#         "device": str(device)
#     }


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
