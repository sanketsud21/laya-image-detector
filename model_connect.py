import os

import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENROUTER_API_KEY in your environment or .env file.")

api_url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
model = "nvidia/nemotron-3-ultra-550b-a55b:free"


def ask_model(messages):
    response = requests.post(
        url=api_url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "reasoning": {"enabled": True},
        },
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        try:
            details = response.json()
        except ValueError:
            details = response.text
        raise RuntimeError(
            f"OpenRouter request failed ({response.status_code}): {details}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("OpenRouter returned invalid JSON.") from error

    choices = payload.get("choices")
    if not choices:
        raise RuntimeError(f"OpenRouter response did not include choices: {payload}")
    return choices[0]["message"]


question = "How many r's are in the word 'strawberry'?"
response = ask_model([{"role": "user", "content": question}])

# Extract the assistant message with reasoning_details
print("first response " + response.get('content'))
# Preserve the assistant message with reasoning_details
messages = [
    {"role": "user", "content": question},
    {
        "role": "assistant",
        "content": response.get("content"),
        "reasoning_details": response.get("reasoning_details"),
    },
    {"role": "user", "content": "Are you sure? Think carefully."},
]

# Second API call - model continues reasoning from where it left off
response2 = ask_model(messages)
print("second response " + response2.get('content'))