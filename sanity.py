from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Missing GEMINI_API_KEY in .env file")
    raise SystemExit

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="What is the density of cooked rice in g/mL? Return only a number."
    )

    print("\n--- RAW RESPONSE ---")
    print(response)
    print("\n--- TEXT OUTPUT ---")
    if hasattr(response, "text"):
        print(response.text)
    else:
        print(str(response))

except Exception as e:
    print("Error calling Gemini API:", e)
