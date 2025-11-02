from groq import Groq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# -----------------------------
# Request schema
# -----------------------------
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Bangla text to be translated")
    target_language: str = Field(..., description="Target language for translation")

# -----------------------------
# Groq client setup
# -----------------------------
client = Groq(api_key=api_key)

# -----------------------------
# System prompt
# -----------------------------
system_prompt = """You are a professional translator.
Translate the Bengali text provided by the user into English.
Requirements:
- Provide only the translation, no extra explanations.
When the user provides Bengali text, respond only with the English translation.
"""


model_name = "openai/gpt-oss-20b"
# -----------------------------
# Translation function
# -----------------------------
def translate_bangla_to_english(text: str) -> str:
    try:
        prompt = f"System: {system_prompt}\nUser: {text}\nAssistant:"

        # Call Groq API
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=512
        )

        # Extract translation result
        if completion.choices and completion.choices[0].message:
            translation = completion.choices[0].message.content.strip()
        else:
            translation = "Error: No translation returned."

        # Clean translation if it contains extra newlines
        if "\n" in translation:
            translation = translation.split("\n")[0]

        return translation

    except Exception as e:
        return f"Translation failed: {str(e)}"



# if __name__ == "__main__":
#     test_text = "জমিয়তে উলামায়ে ইসলাম বাংলাদেশ হলো বাংলাদেশের একটি ইসলামপন্থী রাজনৈতিক দল"
#     print(translate_bangla_to_english(test_text))




























