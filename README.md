# Smart Student Analyzer

A Flask-based student risk dashboard that analyzes screen-time habits, predicts addiction risk, estimates subject performance, generates visual reports, and creates parent-friendly recommendations with optional audio narration.

## Features

- Student input form for screen-time and routine habits
- Addiction risk prediction using trained ML models
- Subject mark prediction for English, Maths, and Science
- Counselor dashboard with dark analytics UI
- Parent report with summary, key problems, recommendations, and audio narration
- Subject-wise analysis with predicted marks, affected factors, and suggested fixes
- Charts for:
  - screen usage split
  - risk indicator
  - study performance vs screen time
  - current input impact by subject
- Optional Grok/XAI-powered wording through the OpenAI-compatible client
- Text-to-speech support through ElevenLabs or gTTS fallback

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- models/
|   |-- random_forest_addiction_model.joblib
|   |-- random_forest_addiction_preprocessor.joblib
|   |-- xgboost_addiction_model.joblib
|   |-- english_marks_model (1) (1).joblib
|   |-- maths_marks_model (1).joblib
|   `-- science_marks_model (1).joblib
|-- services/
|   |-- ai_service.py
|   |-- graph_service.py
|   |-- ml_service.py
|   `-- tts_service.py
|-- static/
|   |-- style.css
|   |-- audio/
|   `-- charts/
`-- templates/
    |-- index.html
    |-- dashboard.html
    |-- parent_report.html
    `-- subject_analysis.html
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Place the trained `.joblib` files in the `models/` folder.

4. Run the app.

```bash
python app.py
```

5. Open the local Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Model Files

The app currently expects these subject model filenames:

```text
english_marks_model (1) (1).joblib
maths_marks_model (1).joblib
science_marks_model (1).joblib
```

For addiction risk, the app first tries:

```text
xgboost_addiction_model.joblib
```

If XGBoost is not installed or the file cannot load, it falls back to:

```text
random_forest_addiction_model.joblib
random_forest_addiction_preprocessor.joblib
```

The final addiction label is calibrated with submitted screen-risk habits so clearly healthy inputs are not shown as high risk due to a raw model mismatch.

## Optional Environment Variables

Create a `.env` file if you want external AI wording or ElevenLabs audio.

```env
XAI_API_KEY=your_xai_or_grok_key
GROK_MODEL=grok-3
XAI_BASE_URL=https://api.x.ai/v1

ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

If no AI key is provided, the app uses local model-driven analysis. If no ElevenLabs key is provided, it tries `gTTS`.

## Input Fields

The student form uses:

- Daily usage hours
- Time on social media
- Sleep hours
- Apps used daily
- Time on gaming
- Phone checks per day

Academic performance is not entered by the user. Internal model features are estimated from the submitted habits where needed.

## Notes

- Generated charts are saved under `static/charts/`.
- Generated audio files are saved under `static/audio/`.
- The Flask secret key in `app.py` should be changed before production use.
- This is a decision-support dashboard, not a medical or clinical diagnosis tool.
