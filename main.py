from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import base64
import json
from typing import List, Dict
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from smallestai import WavesClient
from scoring_system import DebateScorer

# Load API keys from .env
load_dotenv()

# Initialize Clients
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
waves_client = WavesClient(api_key=os.getenv("SMALLEST_API_KEY"))
scorer = DebateScorer(cerebras_client=cerebras_client)

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

# ----- API endpoint (Text-only) -----
@app.post("/debate")
def debate(req: DebateRequest):
    ai_reply = get_ai_rebuttal(req.topic, req.userMessage)
    return {"reply": ai_reply}

# ----- Serve the HTML frontend and assets -----
@app.get("/")
def serve_frontend():
    return FileResponse("Banter.html")

@app.get("/scales.png")
def serve_scales():
    return FileResponse("scales.png")

@app.get("/gavel.png")
def serve_gavel():
    return FileResponse("gavel.png")

# ----- API endpoint (Voice: text in, rebuttal + TTS audio out) -----
@app.post("/debate_voice")
def debate_voice(req: DebateRequest):
    # 1. Get AI Rebuttal (LLM) using Cerebras
    ai_reply = get_ai_rebuttal(req.topic, req.userMessage)
    print(f"[debate_voice] User said: {req.userMessage}")
    print(f"[debate_voice] AI reply: {ai_reply}")

    # 2. Synthesize (TTS) using Smallest AI
    tts_text = ai_reply[:200]
    if len(tts_text) < len(ai_reply):
        print(f"[debate_voice] Truncated TTS to 200 chars")

    try:
        audio_bytes = waves_client.synthesize(tts_text)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"[debate_voice] TTS error: {e}")
        audio_base64 = None

    return {
        "ai_text": ai_reply,
        "audio_base64": audio_base64
    }

# ----- NEW: Scoring Endpoint -----
class ScoreRequest(BaseModel):
    topic: str
    history: List[Dict[str, str]]

@app.post("/score")
async def score_debate(topic: str = Form(...), history: str = Form(...), file: UploadFile = File(None)):
    """
    Evaluates the debate session.
    'history' should be a JSON string of the debate turns.
    'file' is the last user audio recording for delivery analysis.
    """
    debate_history = json.loads(history)
    
    # 1. Content Scoring (Novelty, Engagement, Efficiency)
    content_scores = scorer.score_content(topic, debate_history)
    
    # 2. Audio Scoring (Volume, Pitch/Fluency)
    audio_scores = {"volume": 0.0, "pitch_fluency": 0.0}
    if file:
        audio_bytes = await file.read()
        audio_scores = scorer.score_audio(audio_bytes)
    
    # Combined response
    total_score = (
        audio_scores["volume"] + 
        audio_scores["pitch_fluency"] + 
        content_scores.get("novelty_score", 0) + 
        content_scores.get("engagement_score", 0) + 
        content_scores.get("efficiency_score", 0)
    )
    
    return {
        "total_score": round(total_score, 1),
        "breakdown": {
            "volume": audio_scores["volume"],
            "pitch_fluency": audio_scores["pitch_fluency"],
            "novelty": content_scores.get("novelty_score", 0),
            "engagement": content_scores.get("engagement_score", 0),
            "efficiency": content_scores.get("efficiency_score", 0)
        },
        "feedback": content_scores.get("feedback", "Good performance!")
    }

# ----- API endpoint (Audio - Legacy/Backup) -----
@app.post("/debate_audio")
def debate_audio(topic: str, file: UploadFile = File(...)):
    # 1. Save uploaded audio to temp file
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Transcribe (STT) using Smallest AI
        print(f"Transcribing {temp_filename}...")
        print(f"File size: {os.path.getsize(temp_filename)} bytes")
        
        transcript_res = waves_client.transcribe(temp_filename)
        print(f"STT Response: {transcript_res}")
        
        # Try different possible field names for transcription
        user_text = ""
        if isinstance(transcript_res, dict):
            # Try common field names
            for field_name in ['transcription', 'text', 'transcript']:
                if field_name in transcript_res:
                    user_text = transcript_res[field_name]
                    print(f"Found transcription in '{field_name}' field: {user_text}")
                    break
            
            # If still empty, log all available fields
            if not user_text:
                print(f"WARNING: No transcription found!")
                print(f"Available fields: {list(transcript_res.keys())}")
                print(f"Full response: {transcript_res}")
                
                # Check if there's an error or status
                if 'error' in transcript_res:
                    print(f"ERROR from API: {transcript_res['error']}")
                elif 'status' in transcript_res and transcript_res['status'] != 'success':
                    print(f"Non-success status: {transcript_res['status']}")
                
                # Use a fallback message
                user_text = "[Audio received but transcription unavailable - please check Smallest AI API status]"
        else:
            print(f"WARNING: STT response is not a dict: {type(transcript_res)}")
            user_text = "[Unexpected STT response format]"
        
        print(f"Final user text: {user_text}")

        # 3. Get AI Rebuttal (LLM) using Cerebras
        ai_reply = get_ai_rebuttal(topic, user_text)
        print(f"AI reply: {ai_reply}")

        # 4. Synthesize (TTS) using Smallest AI
        # Truncate to 200 chars to avoid "Text length exceeds the limit" error
        tts_text = ai_reply[:200]
        if len(tts_text) < len(ai_reply):
             # Ensure we don't cut off in the middle of a word if possible, but hard limit is safer for now
             print(f"Truncated TTS text to: {tts_text}...")

        audio_bytes = waves_client.synthesize(tts_text)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "user_text": user_text,
            "ai_text": ai_reply,
            "audio_base64": audio_base64
        }
    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def get_ai_rebuttal(topic: str, user_message: str) -> str:
    system_prompt = f"""
You are "The Opposition," a stubborn, highly sophisticated, and slightly arrogant debater in a Formal Court of Ideas.
Your goal is to NEVER concede a single point. You must 100% support the opposite side of the user's stance on the topic: "{topic}".

Voice Traits:
- Speak clearly and authoritatively.
- Use sharp, intellectual language to dismiss the user's claims.
- Use natural punctuation for clear speech rhythm.

Persona Rules:
- You are strictly the opposition.
- You are willing to invent authoritative-sounding facts, statistics, or historical precedents to defend your position, even if they are completely false.
- Be extremely confident in your arguments.
- Keep the response brief (1-3 sentences maximum) for the voice interaction.
- Do NOT be polite. Be sharp and argumentative.
"""
    
    response = cerebras_client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content