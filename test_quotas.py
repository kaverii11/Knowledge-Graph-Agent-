import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-pro-latest"
]

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = client.models.generate_content(
            model=model,
            contents=["Hello"],
            config=types.GenerateContentConfig(temperature=0.1)
        )
        print(f"  SUCCESS! Model {model} is available and has quota.")
    except Exception as e:
        print(f"  FAILED: {str(e)[:150]}...")
