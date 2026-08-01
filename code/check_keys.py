import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

print("== .env file contents (redacted) ==")
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, val = line.split("=", 1)
        print(f"  {name.strip()}: length={len(val.strip())} starts={val.strip()[:4]}")
else:
    print("  .env NOT FOUND in current directory!")

load_dotenv(override=True)
print("\n== per-key live test (gemini-2.0-flash) ==")
for n in ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]:
    k = os.environ.get(n)
    if not k:
        print(f"  {n}: (not set)")
        continue
    try:
        client = genai.Client(api_key=k)          # hold a reference so it isn't closed
        client.models.generate_content(model="gemini-2.0-flash", contents="say OK")
        print(f"  {n}: {k[:4]}...(len {len(k)}) -> WORKS")
    except Exception as e:
        print(f"  {n}: {k[:4]}...(len {len(k)}) -> ERROR {str(e)[:80]}")