# AI Debate Banter — The Court of Ideas

AI Debate Banter is a real-time, voice-interactive debate platform where you face off against "The Opposition" — a stubborn, sophisticated, and opinionated AI in a premium courtroom setting.

## 🌟 Features

- **🎙️ Real-time Voice Interaction**: Uses the browser's Web Speech API for instant transcription and Smallest.ai's Lightning engine for hyper-realistic TTS responses.
- **⚖️ The Stubborn Opposition**: An AI persona designed to never concede, use intellectual nuance, and even invent authoritative "facts" to defend its stance.
- **📊 Performance Scoring**: A comprehensive 100-point rubric that evaluates your debate skills:
  - **Delivery (35p)**: Volume consistency and Pitch/Fluency.
  - **Content (65p)**: Novelty of contributions, Engagement with AI points, and Efficiency/Conciseness.
- **🎨 Premium UI**: A polished, judicial-themed interface with high-resolution assets and a dynamic "Ember" quote background.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A modern browser (Chrome or Edge recommended for Web Speech API support).

### 2. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*(Ensure `ffmpeg` is installed on your system if you want to run the local test suite audio analysis)*

### 3. API Keys
Create a `.env` file in the root directory and add your keys:
```env
CEREBRAS_API_KEY=your_cerebras_key_here
SMALLEST_API_KEY=your_smallest_ai_key_here
```

### 4. Run the Server
```bash
.venv/bin/uvicorn main:app --port 8001
```

## 🚀 How It Works

1. **Open the App**: Navigate to [http://localhost:8001](http://localhost:8001).
2. **Start a Case**: Enter a topic (e.g., "AI Safety") and your opening stance.
3. **Debate**: 
   - Click the **🎤 Speak Your Argument** button.
   - Your speech will appear live. Click **Stop** to send it.
   - The AI will rebut using its sophisticated, stubborn persona.
4. **The Verdict**: Click **End Session** to receive your **Court Evaluation Report** with a detailed score breakdown and judge's feedback.

## 🧪 Testing
Run the backend test suite to verify endpoints and scoring logic:
```bash
.venv/bin/pytest test_backend.py
```

## 🧰 Tech Stack
- **Backend**: FastAPI (Python)
- **LLM**: Cerebras (Llama 3.1 8B)
- **TTS**: Smallest.ai Waves Lightning
- **Audio Analysis**: Pydub
- **Frontend**: Vanilla JS, HTML5, CSS3, Web Speech API
