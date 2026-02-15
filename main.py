from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import base64
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from smallestai import WavesClient

# Load API keys from .env
load_dotenv()

# Initialize Clients
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
waves_client = WavesClient(api_key=os.getenv("SMALLEST_API_KEY"))

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

# ----- Serve the HTML frontend -----
@app.get("/")
def serve_frontend():
    return FileResponse("Banter.html")

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

# ----- API endpoint (Audio) -----
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
    prompt = f"""
You are a skilled debate opponent.
Debate topic: {topic}

The user argues:
"{user_message}"

Respond with a logical rebuttal.
Keep it extremely brief (1 short sentence).
Be respectful but firm.
"""
    response = cerebras_client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "system", "content": "You are a concise expert debater."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60
    )
    return response.choices[0].message.content