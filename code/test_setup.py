import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)
c = genai.Client(api_key=os.environ["GEMINI_API_KEY_3"])

for model in ["gemini-2.5-flash-lite", "gemini-flash-latest",
              "gemini-2.0-flash", "gemini-3.5-flash"]:
    try:
        r = c.models.generate_content(model=model, contents="say OK")
        print(f"{model:24} -> WORKS: {r.text.strip()[:20]}")
    except Exception as e:
        print(f"{model:24} -> ERROR: {str(e)[:70]}")
    time.sleep(2)