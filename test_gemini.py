import os
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

headers = {
    "Authorization": f"Bearer {GEMINI_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "gemini-pro",
    "messages": [
        {"role": "user", "content": "Hello Gemini, are you online?"}
    ]
}

response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    headers=headers,
    json=data
)
print(response.status_code)
print(response.text) 