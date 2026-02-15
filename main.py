from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# Load API keys from .env
load_dotenv()

# Initialize Cerebras client
# Using the key name found in the user's .env file (CEBERAS_API_KEY)
client = Cerebras(api_key=os.getenv("CEBERAS_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Data format coming FROM the frontend -----
class DebateRequest(BaseModel):
    topic: str
    userMessage: str

# ----- API endpoint -----
@app.post("/debate")
def debate(req: DebateRequest):
    prompt = f"""
You are a skilled debate opponent.
Debate topic: {req.topic}

The user argues:
"{req.userMessage}"

Respond with a logical rebuttal.
Keep it under 4 sentences.
Be respectful but firm.
"""

    response = client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": "You are an expert debater."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    ai_reply = response.choices[0].message.content

    return {
        "reply": ai_reply
    }