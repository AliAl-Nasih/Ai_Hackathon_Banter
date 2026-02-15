import pytest
import os
import json
import base64
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pydub import AudioSegment

# Import the FastAPI app and DebateScorer
from main import app
from scoring_system import DebateScorer

client = TestClient(app)

@pytest.fixture
def mock_cerebras():
    mock = MagicMock()
    # Mock the LLM response for scoring
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "novelty_score": 30,
            "engagement_score": 15,
            "efficiency_score": 8,
            "feedback": "Great job, very concise."
        })))
    ]
    mock.chat.completions.create.return_value = mock_response
    return mock

# --- 1. Unit Tests for DebateScorer ---

@patch("scoring_system.AudioSegment")
def test_score_audio_volume_ideal(mock_audio_class):
    scorer = DebateScorer()
    
    # Mock the AudioSegment instance
    mock_audio = MagicMock()
    mock_audio.dBFS = -15.0
    mock_audio.__len__.return_value = 1000
    mock_audio.strip_silence.return_value = MagicMock(__len__=lambda x: 750) # 0.75 ratio
    
    # Mock segment slicing for pitch variance
    mock_slice = MagicMock()
    mock_slice.dBFS = -12.0
    mock_audio.__getitem__.return_value = mock_slice
    
    mock_audio_class.from_file.return_value = mock_audio
    
    scores = scorer.score_audio(b"fake_audio_bytes")
    assert scores["volume"] == 15.0
    # fluency_ratio 0.75 => fluency_score 10
    # variance 0 => pitch_score 5
    assert scores["pitch_fluency"] == 15.0

@patch("scoring_system.AudioSegment")
def test_score_audio_volume_loud(mock_audio_class):
    scorer = DebateScorer()
    
    mock_audio = MagicMock()
    mock_audio.dBFS = -5.0 # Too loud
    mock_audio.__len__.return_value = 1000
    mock_audio.strip_silence.return_value = MagicMock(__len__=lambda x: 750)
    
    # Pitch components
    s1 = MagicMock(dBFS=-10.0)
    s2 = MagicMock(dBFS=-20.0)
    mock_audio.__getitem__.side_effect = [s1, s2]
    
    mock_audio_class.from_file.return_value = mock_audio
    
    scores = scorer.score_audio(b"fake_audio_bytes")
    # Expected volume: 15 - (-5 + 10)*2 = 5
    assert scores["volume"] == 5.0
    # pitch_score: (max(-10, -20) - min(-10, -20)) / 2 = 10 / 2 = 5
    # total pitch_fluency: 10 + 5 = 15
    assert scores["pitch_fluency"] == 15.0

def test_score_content(mock_cerebras):
    scorer = DebateScorer(cerebras_client=mock_cerebras)
    topic = "AI Safety"
    history = [
        {"role": "user", "content": "AI is dangerous."},
        {"role": "ai", "content": "I disagree."}
    ]
    
    scores = scorer.score_content(topic, history)
    assert scores["novelty_score"] == 30
    assert scores["engagement_score"] == 15
    assert scores["efficiency_score"] == 8
    assert "feedback" in scores

# --- 2. Integration Tests for FastAPI Endpoints ---

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Banter" in response.text

@patch("main.cerebras_client")
def test_debate_endpoint(mock_llm):
    # Mock LLM response
    mock_res = MagicMock()
    mock_res.choices = [MagicMock(message=MagicMock(content="Mocked AI Reply"))]
    mock_llm.chat.completions.create.return_value = mock_res
    
    payload = {"topic": "AI", "userMessage": "Tell me more."}
    response = client.post("/debate", json=payload)
    
    assert response.status_code == 200
    assert response.json()["reply"] == "Mocked AI Reply"

@patch("main.waves_client")
@patch("main.get_ai_rebuttal")
def test_debate_voice_endpoint(mock_rebuttal, mock_waves):
    mock_rebuttal.return_value = "AI Rebuttal Text"
    mock_waves.synthesize.return_value = b"fakeaudiobytes"
    
    payload = {"topic": "AI", "userMessage": "Hello"}
    response = client.post("/debate_voice", json=payload)
    
    assert response.status_code == 200
    assert response.json()["ai_text"] == "AI Rebuttal Text"
    assert "audio_base64" in response.json()
    assert response.json()["audio_base64"] == base64.b64encode(b"fakeaudiobytes").decode("utf-8")

@patch("main.scorer")
def test_score_endpoint(mock_scorer):
    # Mock Content and Audio scores
    mock_scorer.score_content.return_value = {
        "novelty_score": 30, "engagement_score": 15, "efficiency_score": 8, "feedback": "Feedback"
    }
    mock_scorer.score_audio.return_value = {
        "volume": 12.0, "pitch_fluency": 18.0
    }
    
    history = json.dumps([{"role": "user", "content": "Test"}])
    data = {
        "topic": "AI",
        "history": history
    }
    
    # Create a dummy audio file
    audio_file = ("test.webm", b"fake audio data", "audio/webm")
    
    response = client.post("/score", data=data, files={"file": audio_file})
    
    assert response.status_code == 200
    json_res = response.json()
    assert json_res["total_score"] == 12.0 + 18.0 + 30 + 15 + 8
    assert json_res["breakdown"]["volume"] == 12.0
    assert json_res["breakdown"]["novelty"] == 30
