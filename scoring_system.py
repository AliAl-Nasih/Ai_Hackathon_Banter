import os
import io
import math
from pydub import AudioSegment
from typing import List, Dict

class DebateScorer:
    def __init__(self, cerebras_client=None):
        self.cerebras_client = cerebras_client

    def score_audio(self, audio_bytes: bytes) -> Dict[str, float]:
        """
        Scores Volume (15p), Pitch and Fluency (20p) using pydub.
        """
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # --- 1. Volume Score (15p) ---
            # Ideal range: -20 dBFS to -10 dBFS
            dbfs = audio.dBFS
            if -20 <= dbfs <= -10:
                volume_score = 15
            elif dbfs > -10:
                # Too loud (yelling)
                volume_score = max(0, 15 - (dbfs + 10) * 2)
            else:
                # Too quiet (whispering)
                volume_score = max(0, 15 - abs(dbfs + 20))
            
            # --- 2. Pitch and Fluency (20p) ---
            # Fluency: Speed and consistency. Check for silence ratio.
            chunks = audio.split_to_mono() # Use first channel if stereo
            # (Simplification: detect non-silent parts)
            non_silent = audio.strip_silence(silence_thresh=-50, chunk_size=10)
            fluency_ratio = len(non_silent) / len(audio) if len(audio) > 0 else 0
            # Ideal fluency ratio is high (>0.7), but not 1.0 (need some pauses)
            if 0.6 <= fluency_ratio <= 0.9:
                fluency_score = 10
            else:
                fluency_score = max(0, 10 - abs(fluency_ratio - 0.75) * 20)
                
            # Pitch dynamic range (estimated by amplitude variance)
            # We look at the max vs min volume of sub-segments
            segment_dbfs = [audio[i:i+500].dBFS for i in range(0, len(audio), 500) if len(audio[i:i+500]) > 0]
            if len(segment_dbfs) > 1:
                variance = max(segment_dbfs) - min(segment_dbfs)
                # Ideal variance is roughly 10-20 dB
                pitch_score = min(10, variance / 2)
            else:
                pitch_score = 5

            return {
                "volume": round(volume_score, 1),
                "pitch_fluency": round(fluency_score + pitch_score, 1)
            }
        except Exception as e:
            print(f"Audio scoring error: {e}")
            return {"volume": 0.0, "pitch_fluency": 0.0}

    def score_content(self, topic: str, history: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Scores Novelty (35p), Engagement (20p), and Efficiency (10p) using Cerebras.
        """
        if not self.cerebras_client:
            return {"novelty": 0, "engagement": 0, "efficiency": 0, "feedback": "Scoring unavailable (No LLM)"}

        history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])
        
        prompt = f"""
Evaluate the USER's performance in this debate on the topic: "{topic}".
Rubric (Total 65 points for content):
1. Novel Contributions (35p): Did they bring new evidence, analogies, or perspectives to shift the topic?
2. Engagement (20p): Did they respond to the AI's questions, recognize valid claims, and expand on them?
3. Efficiency (10p): Were they concise and clear?

Debate History:
{history_str}

Return a JSON object with:
- novelty_score (0-35)
- engagement_score (0-20)
- efficiency_score (0-10)
- feedback (String, 2-3 sentences max summarizing pros/cons)

ONLY RETURN THE JSON.
"""
        try:
            response = self.cerebras_client.chat.completions.create(
                model="llama3.1-8b",
                messages=[{"role": "system", "content": "You are a professional debate judge."}, {"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            # Basic parsing if LLM wraps in code blocks
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except Exception as e:
            print(f"Content scoring error: {e}")
            return {"novelty_score": 0, "engagement_score": 0, "efficiency_score": 0, "feedback": "Error analyzing content."}
