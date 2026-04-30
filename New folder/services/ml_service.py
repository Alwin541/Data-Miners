from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "addiction": "logistica_regression_addiction_model.joblib",
    "addiction_fallback": "random_forest_addiction_model.joblib",
    "marks": {
        "english": "english_marks_model (1) (1).joblib",
        "maths": "maths_marks_model (1).joblib",
        "science": "science_marks_model (1).joblib",
    },
}
MARKS_MODEL_PATTERNS = {
    "english": ["*english*marks*.joblib", "*english*model*.joblib"],
    "maths": ["*maths*marks*.joblib", "*math*marks*.joblib", "*maths*model*.joblib", "*math*model*.joblib"],
    "science": ["*science*marks*.joblib", "*science*model*.joblib"],
}
ADDICTION_PREPROCESSOR_FILE = "xgboost_addiction_preprocessor.joblib"
ADDICTION_FALLBACK_PREPROCESSOR_FILE = "random_forest_addiction_preprocessor.joblib"

FEATURE_COLUMNS = [
    "Daily_Usage_Hours",
    "Time_on_Social_Media",
    "Sleep_Hours",
    "Apps_Used_Daily",
    "Time_on_Gaming",
    "Phone_Checks_Per_Day",
    "Academic_Performance",
]

MARKS_FEATURE_COLUMNS = [
    "Daily_Usage_Hours",
    "Sleep_Hours",
    "Apps_Used_Daily",
    "Time_on_Social_Media",
    "Time_on_Gaming",
    "Phone_Checks_Per_Day",
    "Family_Communication",
    "Weekend_Usage_Hours",
    "Screen_Time_Before_Bed",
    "Time_on_Education",
    "Exercise_Hours",
    "Academic_Performance",
]

MARKS_FEATURE_DEFAULTS = {
    "Family_Communication": 6.0,
    "Weekend_Usage_Hours": 4.0,
    "Screen_Time_Before_Bed": 1.0,
    "Time_on_Education": 2.0,
    "Exercise_Hours": 1.0,
}


@dataclass
class FeatureInput:
    Daily_Usage_Hours: float
    Time_on_Social_Media: float
    Sleep_Hours: float
    Apps_Used_Daily: int
    Time_on_Gaming: float
    Phone_Checks_Per_Day: int

    @classmethod
    def from_form(cls, form: Dict[str, Any]) -> "FeatureInput":
        return cls(
            Daily_Usage_Hours=float(form.get("Daily_Usage_Hours", 0) or 0),
            Time_on_Social_Media=float(form.get("Time_on_Social_Media", 0) or 0),
            Sleep_Hours=float(form.get("Sleep_Hours", 0) or 0),
            Apps_Used_Daily=int(float(form.get("Apps_Used_Daily", 0) or 0)),
            Time_on_Gaming=float(form.get("Time_on_Gaming", 0) or 0),
            Phone_Checks_Per_Day=int(float(form.get("Phone_Checks_Per_Day", 0) or 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "Daily_Usage_Hours": self.Daily_Usage_Hours,
            "Time_on_Social_Media": self.Time_on_Social_Media,
            "Sleep_Hours": self.Sleep_Hours,
            "Apps_Used_Daily": self.Apps_Used_Daily,
            "Time_on_Gaming": self.Time_on_Gaming,
            "Phone_Checks_Per_Day": self.Phone_Checks_Per_Day,
        }


def _load_joblib(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model file {path.name}: {exc}") from exc


def _try_load_joblib(path: Path) -> Optional[Any]:
    try:
        return _load_joblib(path)
    except RuntimeError:
        return None


def _resolve_marks_model_file(subject: str) -> Optional[Path]:
    configured = MODEL_DIR / MODEL_FILES["marks"][subject]
    if configured.exists():
        return configured

    for pattern in MARKS_MODEL_PATTERNS[subject]:
        matches = sorted(MODEL_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def _build_dataframe(feature_input: FeatureInput):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for ML inference. Install requirements.txt.") from exc

    values = {
        **feature_input.to_dict(),
        "Academic_Performance": _estimate_academic_performance(feature_input),
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def _build_marks_dataframe(feature_input: FeatureInput):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for ML inference. Install requirements.txt.") from exc

    values = {
        **MARKS_FEATURE_DEFAULTS,
        "Daily_Usage_Hours": feature_input.Daily_Usage_Hours,
        "Sleep_Hours": feature_input.Sleep_Hours,
        "Apps_Used_Daily": feature_input.Apps_Used_Daily,
        "Time_on_Social_Media": feature_input.Time_on_Social_Media,
        "Time_on_Gaming": feature_input.Time_on_Gaming,
        "Phone_Checks_Per_Day": feature_input.Phone_Checks_Per_Day,
        "Academic_Performance": _estimate_academic_performance(feature_input),
    }
    return pd.DataFrame([values], columns=MARKS_FEATURE_COLUMNS)


def _normalize_addiction_label(raw_value: Any) -> str:
    value = raw_value.item() if hasattr(raw_value, "item") else raw_value
    if isinstance(value, str):
        label = value.strip().title()
        if label in {"Low", "Medium", "High"}:
            return label
        if label in {"0", "1", "2"}:
            value = int(label)

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Medium"

    if numeric <= 0.5:
        return "Low"
    if numeric <= 1.5:
        return "Medium"
    return "High"


def _rule_based_addiction_label(feature_input: FeatureInput) -> str:
    risk_points = 0
    risk_points += feature_input.Daily_Usage_Hours >= 6
    risk_points += feature_input.Time_on_Social_Media >= 3
    risk_points += feature_input.Time_on_Gaming >= 2
    risk_points += feature_input.Sleep_Hours < 6.5
    risk_points += feature_input.Apps_Used_Daily >= 15
    risk_points += feature_input.Phone_Checks_Per_Day >= 90

    if risk_points <= 1:
        return "Low"
    if risk_points <= 3:
        return "Medium"
    return "High"


def _calibrate_addiction_label(model_label: str, feature_input: FeatureInput) -> str:
    rule_label = _rule_based_addiction_label(feature_input)
    rank = {"Low": 0, "Medium": 1, "High": 2}
    model_rank = rank.get(model_label, 1)
    rule_rank = rank[rule_label]

    if abs(model_rank - rule_rank) >= 2:
        return rule_label
    if rule_label == "Low" and model_label == "Medium":
        return "Low"
    return model_label


def _bounded_mark(value: float) -> float:
    return round(max(0, min(100, float(value))), 1)


def _estimate_academic_performance(feature_input: FeatureInput) -> float:
    score = 92.0
    score -= max(0, feature_input.Daily_Usage_Hours - 2.5) * 2.0
    score -= max(0, feature_input.Time_on_Social_Media - 1.0) * 1.2
    score -= max(0, feature_input.Time_on_Gaming - 0.75) * 1.5
    score -= max(0, 7.5 - feature_input.Sleep_Hours) * 2.5
    score -= max(0, feature_input.Apps_Used_Daily - 8) * 0.15
    score -= max(0, feature_input.Phone_Checks_Per_Day - 40) * 0.03
    return _bounded_mark(score)


def _predict_addiction(model: Any, addiction_preprocessor: Optional[Any], input_df):
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features != len(FEATURE_COLUMNS):
        raise RuntimeError(
            f"Addiction model expects {expected_features} features, but this app provides {len(FEATURE_COLUMNS)}."
        )

    transformed = addiction_preprocessor.transform(input_df) if addiction_preprocessor is not None else input_df.to_numpy()
    return model.predict(transformed)


def _predict_subject_mark(feature_input: FeatureInput, subject: str) -> Optional[float]:
    marks_model_path = _resolve_marks_model_file(subject)
    if marks_model_path is None:
        return None
    marks_model = _load_joblib(marks_model_path)
    if marks_model is None:
        return None
    if hasattr(marks_model, "n_jobs"):
        marks_model.n_jobs = 1

    marks_df = _build_marks_dataframe(feature_input)
    expected_features = getattr(marks_model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(MARKS_FEATURE_COLUMNS):
        raise RuntimeError(
            f"Marks model expects {expected_features} features, but this app provides {len(MARKS_FEATURE_COLUMNS)}."
        )

    raw_prediction = marks_model.predict(marks_df)[0]
    return _bounded_mark(raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction)


def _fallback_subject_marks(feature_input: FeatureInput) -> Dict[str, float]:
    baseline = _estimate_academic_performance(feature_input)
    usage_penalty = max(0, feature_input.Daily_Usage_Hours - 4) * 1.8
    social_penalty = max(0, feature_input.Time_on_Social_Media - 1.5) * 1.4
    gaming_penalty = max(0, feature_input.Time_on_Gaming - 1) * 1.8
    sleep_penalty = max(0, 7 - feature_input.Sleep_Hours) * 2.0
    app_penalty = max(0, feature_input.Apps_Used_Daily - 10) * 0.25
    check_penalty = max(0, feature_input.Phone_Checks_Per_Day - 60) * 0.04

    return {
        "english": _bounded_mark(baseline - usage_penalty - social_penalty - sleep_penalty - app_penalty),
        "maths": _bounded_mark(baseline - usage_penalty - gaming_penalty - sleep_penalty - check_penalty),
        "science": _bounded_mark(baseline - usage_penalty - social_penalty / 2 - gaming_penalty / 2 - sleep_penalty),
    }


def _predict_subject_marks(feature_input: FeatureInput) -> tuple[Dict[str, float], str]:
    fallback_marks = _fallback_subject_marks(feature_input)
    marks = {}
    loaded_subjects = []
    missing_subjects = []

    for subject in ("english", "maths", "science"):
        try:
            predicted_mark = _predict_subject_mark(feature_input, subject)
        except Exception:
            predicted_mark = None

        if predicted_mark is None:
            marks[subject] = fallback_marks[subject]
            missing_subjects.append(subject.title())
        else:
            marks[subject] = predicted_mark
            loaded_subjects.append(subject.title())

    if not missing_subjects:
        status = "Subject marks use trained English, Maths, and Science marks models."
    elif loaded_subjects:
        status = (
            f"Subject marks use trained models for {', '.join(loaded_subjects)}; "
            f"{', '.join(missing_subjects)} use input-based fallback estimates because those model files are missing."
        )
    else:
        status = "Subject marks use input-based fallback estimates because trained subject marks models could not load."
    return marks, status


def _fallback_prediction(feature_input: FeatureInput) -> Dict[str, Any]:
    marks = _fallback_subject_marks(feature_input)
    return {
        "addiction": _rule_based_addiction_label(feature_input),
        **marks,
        "model_status": "Fallback rule set used because trained addiction model/preprocessor loading failed.",
    }


def predict_student_risk(feature_input: FeatureInput) -> Dict[str, Any]:
    try:
        marks, marks_model_status = _predict_subject_marks(feature_input)

        input_df = _build_dataframe(feature_input)
        addiction_preprocessor = _try_load_joblib(MODEL_DIR / ADDICTION_PREPROCESSOR_FILE)
        addiction_model = _try_load_joblib(MODEL_DIR / MODEL_FILES["addiction"])
        addiction_model_file = MODEL_FILES["addiction"]
        if addiction_model is None:
            addiction_preprocessor = _try_load_joblib(MODEL_DIR / ADDICTION_FALLBACK_PREPROCESSOR_FILE)
            addiction_model = _try_load_joblib(MODEL_DIR / MODEL_FILES["addiction_fallback"])
            addiction_model_file = MODEL_FILES["addiction_fallback"]
        if addiction_model is None:
            raise RuntimeError(f"Missing model file: {MODEL_FILES['addiction_fallback']}")

        model_addiction = _normalize_addiction_label(_predict_addiction(addiction_model, addiction_preprocessor, input_df)[0])
        addiction = _calibrate_addiction_label(model_addiction, feature_input)

        return {
            "addiction": addiction,
            **marks,
            "model_status": (
                f"Addiction prediction uses {addiction_model_file}. "
                f"Raw addiction model label was {model_addiction}; final risk is calibrated with submitted screen-risk habits. "
                f"{marks_model_status}"
            ),
        }
    except Exception:
        return _fallback_prediction(feature_input)
