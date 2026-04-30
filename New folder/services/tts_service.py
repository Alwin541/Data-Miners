import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

from services.ml_service import BASE_DIR


STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"


def build_parent_audio_script(analysis: dict, prediction: dict, student_input: dict) -> str:
    recommendations = analysis.get("recommendations", [])
    suggestions_text = " ".join(
        f"Suggestion {index + 1}: {item}" for index, item in enumerate(recommendations)
    )
    return (
        f"Student risk report. The predicted addiction risk is {prediction.get('addiction')}. "
        f"The predicted marks are English {prediction.get('english')}, Maths {prediction.get('maths')}, "
        f"and Science {prediction.get('science')}. "
        f"The daily usage is {student_input.get('Daily_Usage_Hours')} hours, social media is "
        f"{student_input.get('Time_on_Social_Media')} hours, gaming is {student_input.get('Time_on_Gaming')} hours, "
        f"sleep is {student_input.get('Sleep_Hours')} hours, apps used daily is {student_input.get('Apps_Used_Daily')}, "
        f"and phone checks per day is {student_input.get('Phone_Checks_Per_Day')}. "
        f"{analysis.get('parent_summary', '')} "
        f"The biggest risk factor is {analysis.get('biggest_risk_factor')}, and the most affected subject is "
        f"{analysis.get('most_affected_subject')}. {suggestions_text}"
    )


def _elevenlabs_tts(text: str, output: Path) -> bool:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return False

    try:
        import requests

        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
                "voice_settings": {"stability": 0.55, "similarity_boost": 0.75},
            },
            timeout=30,
        )
        response.raise_for_status()
        output.write_bytes(response.content)
        return output.exists() and output.stat().st_size > 0
    except Exception:
        return False


def _gtts_tts(text: str, output: Path) -> bool:
    try:
        from gtts import gTTS

        gTTS(text=text, lang="en").save(str(output))
        return output.exists() and output.stat().st_size > 0
    except Exception:
        return False


def generate_parent_audio(parent_summary: str) -> Optional[str]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"parent_summary_{uuid4().hex[:10]}.mp3"
    output = AUDIO_DIR / filename

    if _elevenlabs_tts(parent_summary, output) or _gtts_tts(parent_summary, output):
        return f"audio/{filename}"
    if output.exists() and output.stat().st_size == 0:
        try:
            output.unlink()
        except OSError:
            pass
    return None
