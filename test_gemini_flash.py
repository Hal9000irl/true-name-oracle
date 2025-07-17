import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAQR6yckwlWt3zdVHDntYHIhiB6Q4dyfFo"
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": API_KEY
}
data = {
    "contents": [
        {
            "parts": [
                {"text": "Explain how AI works in a few words"}
            ]
        }
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text) 