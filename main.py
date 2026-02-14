from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ----- Data format coming FROM the frontend -----
class DebateRequest(BaseModel):
    topic: str
    user_argument: str

# ----- API endpoint -----
@app.post("/debate")
def debate(req: DebateRequest):
    prompt = f"""
You are a skilled debate opponent.
Debate topic: {req.topic}

The user argues:
"{req.user_argument}"

Respond with a logical rebuttal.
Keep it under 4 sentences.
Be respectful but firm.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert debater."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    ai_reply = response.choices[0].message.content

    return {
        "ai_reply": ai_reply
    }