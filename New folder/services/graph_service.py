from pathlib import Path
from typing import Any, Dict, List
import os
from uuid import uuid4

from services.ml_service import BASE_DIR, FeatureInput


CHART_DIR = BASE_DIR / "static" / "charts"
MPL_CONFIG_DIR = BASE_DIR / "static" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def _risk_color(addiction: str) -> str:
    return {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}.get(addiction, "#546e7a")


def _save_pie(feature_input: FeatureInput, run_id: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    other_screen = max(
        0,
        feature_input.Daily_Usage_Hours - feature_input.Time_on_Social_Media - feature_input.Time_on_Gaming,
    )
    labels = ["Social Media", "Gaming", "Other Screen"]
    values = [
        feature_input.Time_on_Social_Media,
        feature_input.Time_on_Gaming,
        other_screen,
    ]
    if sum(values) <= 0:
        values = [1, 1, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Screen Usage Split")
    output = CHART_DIR / f"screen_usage_pie_{run_id}.png"
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return f"charts/screen_usage_pie_{run_id}.png"


def _save_risk_indicator(addiction: str, run_id: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 1.4))
    ax.axis("off")
    color = _risk_color(addiction)
    ax.add_patch(plt.Rectangle((0.02, 0.25), 0.96, 0.5, color=color, transform=ax.transAxes))
    ax.text(
        0.5,
        0.5,
        f"{addiction.upper()} ADDICTION RISK",
        color="white",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )
    output = CHART_DIR / f"risk_indicator_{run_id}.png"
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return f"charts/risk_indicator_{run_id}.png"


def _bounded_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _average_screen_penalty(feature_input: FeatureInput, daily_usage_hours: float) -> float:
    usage_penalty = max(0, daily_usage_hours - 4) * 1.8
    social_penalty = max(0, feature_input.Time_on_Social_Media - 1.5) * 1.4
    gaming_penalty = max(0, feature_input.Time_on_Gaming - 1) * 1.8
    sleep_penalty = max(0, 7 - feature_input.Sleep_Hours) * 2.0
    app_penalty = max(0, feature_input.Apps_Used_Daily - 10) * 0.25
    check_penalty = max(0, feature_input.Phone_Checks_Per_Day - 60) * 0.04

    english_penalty = usage_penalty + social_penalty + sleep_penalty + app_penalty
    maths_penalty = usage_penalty + gaming_penalty + sleep_penalty + check_penalty
    science_penalty = usage_penalty + social_penalty / 2 + gaming_penalty / 2 + sleep_penalty
    return (english_penalty + maths_penalty + science_penalty) / 3


def _performance_for_screen_time(feature_input: FeatureInput, daily_usage_hours: float, baseline: float) -> float:
    return round(_bounded_score(baseline - _average_screen_penalty(feature_input, daily_usage_hours)), 1)


def _save_performance_vs_screen_time(feature_input: FeatureInput, prediction: Dict[str, Any], run_id: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usage_values = [hour * 0.5 for hour in range(0, 25)]
    current_usage = feature_input.Daily_Usage_Hours
    current_performance = round(
        (prediction["english"] + prediction["maths"] + prediction["science"]) / 3,
        1,
    )
    baseline = current_performance + _average_screen_penalty(feature_input, current_usage)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    performance_values = [
        _performance_for_screen_time(feature_input, usage, baseline)
        for usage in usage_values
    ]
    ax.plot(usage_values, performance_values, color="#1565c0", linewidth=2.4)
    ax.scatter([current_usage], [current_performance], color="#c62828", s=70, zorder=3)
    ax.axvline(current_usage, color="#c62828", linestyle="--", linewidth=1)
    ax.annotate(
        f"Current: {current_usage:g}h, {current_performance:g}",
        xy=(current_usage, current_performance),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        color="#263238",
    )
    ax.set_title("Study Performance vs Screen Time")
    ax.set_xlabel("Daily screen time (hours)")
    ax.set_ylabel("Predicted study performance")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output = CHART_DIR / f"study_performance_screen_time_{run_id}.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return f"charts/study_performance_screen_time_{run_id}.png"


def _bounded(value: float, upper: float = 2.0) -> float:
    return max(0.0, min(upper, value))


def _negative_impact(weight: float, severity: float) -> float:
    impact = round(weight * severity, 3)
    return 0.0 if impact == 0 else -impact


def _feature_label(feature: str) -> str:
    labels = {
        "Daily_Usage_Hours": "Daily usage",
        "Time_on_Social_Media": "Social media",
        "Sleep_Hours": "Sleep deficit",
        "Apps_Used_Daily": "Apps used",
        "Time_on_Gaming": "Gaming",
        "Phone_Checks_Per_Day": "Phone checks",
    }
    return labels.get(feature, feature.replace("_", " "))


def _feature_value(feature_input: FeatureInput, feature: str) -> str:
    value = getattr(feature_input, feature)
    if feature in {"Daily_Usage_Hours", "Time_on_Social_Media", "Time_on_Gaming", "Sleep_Hours"}:
        return f"{value:g}h"
    return f"{value:g}"


def _input_severity(feature_input: FeatureInput) -> Dict[str, float]:
    return {
        "Daily_Usage_Hours": _bounded((feature_input.Daily_Usage_Hours - 3.5) / 4.5),
        "Time_on_Social_Media": _bounded((feature_input.Time_on_Social_Media - 1.0) / 3.0),
        "Time_on_Gaming": _bounded((feature_input.Time_on_Gaming - 0.75) / 3.0),
        "Sleep_Hours": _bounded((7.5 - feature_input.Sleep_Hours) / 3.0),
        "Apps_Used_Daily": _bounded((feature_input.Apps_Used_Daily - 10) / 25.0),
        "Phone_Checks_Per_Day": _bounded((feature_input.Phone_Checks_Per_Day - 60) / 140.0),
    }


def _model_coefficients(model_name: str, feature_input: FeatureInput) -> Dict[str, List[Dict[str, Any]]]:
    severity = _input_severity(feature_input)
    weights = {
        "english": {
            "Time_on_Social_Media": 0.42,
            "Daily_Usage_Hours": 0.34,
            "Sleep_Hours": 0.28,
            "Apps_Used_Daily": 0.18,
            "Phone_Checks_Per_Day": 0.12,
            "Time_on_Gaming": 0.08,
        },
        "maths": {
            "Time_on_Gaming": 0.48,
            "Phone_Checks_Per_Day": 0.31,
            "Daily_Usage_Hours": 0.26,
            "Sleep_Hours": 0.22,
            "Apps_Used_Daily": 0.12,
            "Time_on_Social_Media": 0.08,
        },
        "science": {
            "Daily_Usage_Hours": 0.37,
            "Time_on_Social_Media": 0.33,
            "Time_on_Gaming": 0.24,
            "Sleep_Hours": 0.29,
            "Phone_Checks_Per_Day": 0.12,
            "Apps_Used_Daily": 0.10,
        },
    }

    negative = sorted(
        (
            (feature, _negative_impact(weight, severity.get(feature, 0)))
            for feature, weight in weights[model_name].items()
        ),
        key=lambda item: item[1],
    )[:3]
    sleep_support = _bounded(feature_input.Sleep_Hours / 8.0, upper=1.25)
    positive = [("Sleep_Hours", round(weights[model_name]["Sleep_Hours"] * sleep_support, 3))]
    return {
        "negative": [{"feature": name, "impact": value} for name, value in negative],
        "positive": [{"feature": name, "impact": value} for name, value in positive],
    }


def _save_feature_impact_chart(
    feature_input: FeatureInput,
    feature_impacts: Dict[str, Dict[str, List[Dict[str, Any]]]],
    run_id: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for subject, impacts in feature_impacts.items():
        for item in impacts["negative"]:
            if item["impact"] < 0:
                feature = item["feature"]
                rows.append(
                    {
                        "label": f"{subject.title()}: {_feature_label(feature)} ({_feature_value(feature_input, feature)})",
                        "impact": item["impact"],
                    }
                )

    if not rows:
        rows = [
            {
                "label": f"Sleep support ({_feature_value(feature_input, 'Sleep_Hours')})",
                "impact": max(
                    item["impact"]
                    for impacts in feature_impacts.values()
                    for item in impacts["positive"]
                ),
            }
        ]

    rows = sorted(rows, key=lambda item: item["impact"])[:8]
    labels = [item["label"] for item in rows]
    values = [item["impact"] for item in rows]
    colors = ["#c62828" if value < 0 else "#2e7d32" for value in values]

    fig_height = max(3.4, 0.55 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#263238", linewidth=1)
    ax.set_title("Current Input Impact by Subject")
    ax.set_xlabel("Input-adjusted impact score")
    fig.tight_layout()
    output = CHART_DIR / f"feature_impact_bar_{run_id}.png"
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return f"charts/feature_impact_bar_{run_id}.png"


def generate_graphs(feature_input: FeatureInput, prediction: Dict[str, Any]) -> Dict[str, Any]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:10]
    feature_impacts = {
        "english": _model_coefficients("english", feature_input),
        "maths": _model_coefficients("maths", feature_input),
        "science": _model_coefficients("science", feature_input),
    }

    return {
        "screen_usage_pie": _save_pie(feature_input, run_id),
        "study_performance_screen_time": _save_performance_vs_screen_time(feature_input, prediction, run_id),
        "feature_impact_bar": _save_feature_impact_chart(feature_input, feature_impacts, run_id),
        "risk_indicator": _save_risk_indicator(prediction["addiction"], run_id),
        "risk_color": _risk_color(prediction["addiction"]),
        "feature_impacts": feature_impacts,
        "run_id": run_id,
    }
