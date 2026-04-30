import json
import os
import re
from typing import Any, Dict, List

from services.ml_service import FeatureInput


SUBJECT_CLOSE_THRESHOLD = 3.0
NO_AFFECTED_MARK_THRESHOLD = 80.0


def _most_affected_subject(marks: Dict[str, float]) -> str:
    if all(mark > NO_AFFECTED_MARK_THRESHOLD for mark in marks.values()):
        return "No affected subject"
    if max(marks.values()) - min(marks.values()) <= SUBJECT_CLOSE_THRESHOLD:
        return "All subjects"
    lowest = min(marks.values())
    tied = [subject for subject, mark in marks.items() if abs(mark - lowest) <= SUBJECT_CLOSE_THRESHOLD]
    if len(tied) == len(marks):
        return "All subjects"
    if len(tied) > 1:
        return " and ".join(tied)
    return tied[0]


def _subject_mark(marks: Dict[str, float], subject: str) -> float:
    if subject in marks:
        return marks[subject]
    return min(marks.values())


def _subject_phrase(subject: str) -> str:
    if subject == "All subjects":
        return "all three subjects"
    if subject == "No affected subject":
        return "no affected subject"
    return subject


def _aggregate_subject_impacts(feature_impacts: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    negative_by_feature: Dict[str, float] = {}
    positive_by_feature: Dict[str, float] = {}
    for impacts in feature_impacts.values():
        for item in impacts.get("negative", []):
            feature = item.get("feature", "")
            impact = abs(float(item.get("impact", 0)))
            negative_by_feature[feature] = max(negative_by_feature.get(feature, 0), impact)
        for item in impacts.get("positive", []):
            feature = item.get("feature", "")
            impact = float(item.get("impact", 0))
            positive_by_feature[feature] = max(positive_by_feature.get(feature, 0), impact)

    return {
        "negative": [
            {"feature": feature, "impact": -impact}
            for feature, impact in sorted(negative_by_feature.items(), key=lambda item: item[1], reverse=True)
        ],
        "positive": [
            {"feature": feature, "impact": impact}
            for feature, impact in sorted(positive_by_feature.items(), key=lambda item: item[1], reverse=True)
        ],
    }


def _feature_value(feature_input: FeatureInput, feature: str) -> float:
    return float(getattr(feature_input, feature, 0) or 0)


def _human_feature(feature: str) -> str:
    names = {
        "Daily_Usage_Hours": "daily phone usage",
        "Sleep_Hours": "sleep",
        "Time_on_Social_Media": "social media time",
        "Time_on_Gaming": "gaming time",
        "Apps_Used_Daily": "number of apps used daily",
        "Phone_Checks_Per_Day": "phone checking frequency",
        "Academic_Performance": "internal academic baseline",
        "Healthy_Routine": "stable routine",
    }
    return names.get(feature, feature.replace("_", " ").lower())


def _habit_risk_scores(feature_input: FeatureInput, subject_impacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    usage_penalties = {
        "Daily_Usage_Hours": max(0, feature_input.Daily_Usage_Hours - 3.5),
        "Time_on_Social_Media": max(0, feature_input.Time_on_Social_Media - 1),
        "Time_on_Gaming": max(0, feature_input.Time_on_Gaming - 0.75),
        "Sleep_Hours": max(0, 7.5 - feature_input.Sleep_Hours),
        "Apps_Used_Daily": max(0, feature_input.Apps_Used_Daily - 10),
        "Phone_Checks_Per_Day": max(0, feature_input.Phone_Checks_Per_Day - 60) / 20,
    }

    model_penalties = {feature: 0.0 for feature in usage_penalties}
    for item in subject_impacts.get("negative", []):
        feature = item.get("feature", "")
        if feature in model_penalties:
            model_penalties[feature] += abs(float(item.get("impact", 0)))

    risks = []
    for feature, usage_penalty in usage_penalties.items():
        if usage_penalty <= 0:
            continue
        score = usage_penalty * (1 + model_penalties.get(feature, 0))
        risks.append(
            {
                "feature": feature,
                "label": _human_feature(feature),
                "value": _feature_value(feature_input, feature),
                "score": round(score, 3),
                "model_penalty": round(model_penalties.get(feature, 0), 3),
            }
        )

    return sorted(risks, key=lambda item: item["score"], reverse=True)


def _strongest_positive_factor(subject_impacts: Dict[str, Any]) -> str:
    positives = subject_impacts.get("positive", [])
    if not positives:
        return "Sleep_Hours"
    return max(positives, key=lambda item: float(item.get("impact", 0))).get("feature", "Sleep_Hours")


def _elaborate_recommendation(
    title: str,
    action: str,
    reason: str,
    follow_up: str,
) -> str:
    return f"{title}: {action} Reason: {reason} Follow-up: {follow_up}"


def _effect_reason(subject: str, feature: str, value: float, mark: float) -> str:
    subject_name = subject.title()
    if feature == "Healthy_Routine":
        return f"{subject_name} does not have a major screen-habit pressure in the submitted values; predicted mark is {mark}."
    if feature == "Daily_Usage_Hours":
        return f"{subject_name} is affected because total phone use is {value:g}h, leaving less uninterrupted study time; predicted mark is {mark}."
    if feature == "Time_on_Social_Media":
        return f"{subject_name} is affected because social media is {value:g}h, which can break reading, recall, and writing focus; predicted mark is {mark}."
    if feature == "Time_on_Gaming":
        return f"{subject_name} is affected because gaming is {value:g}h, which competes with practice-heavy study time; predicted mark is {mark}."
    if feature == "Sleep_Hours":
        if value >= 7.5:
            return f"{subject_name} is supported by {value:g}h sleep; no sleep-related weakness is flagged. Predicted mark is {mark}."
        return f"{subject_name} is affected because sleep is only {value:g}h, reducing attention and memory consolidation; predicted mark is {mark}."
    if feature == "Apps_Used_Daily":
        return f"{subject_name} is affected because {value:g} apps are used daily, increasing context switching during study; predicted mark is {mark}."
    if feature == "Phone_Checks_Per_Day":
        return f"{subject_name} is affected because the phone is checked {value:g} times per day, fragmenting concentration; predicted mark is {mark}."
    return f"{subject_name} is affected by {_human_feature(feature)} at {value:g}; predicted mark is {mark}."


def _dashboard_subject_message(subject: str, feature: str, value: float) -> str:
    if feature == "Healthy_Routine":
        return "No major subject pressure is detected from the submitted habits."
    if subject == "english":
        if feature == "Time_on_Social_Media":
            return f"Reading and writing focus is most affected by {value:g}h of social media."
        if feature == "Daily_Usage_Hours":
            return f"Long total phone use ({value:g}h) reduces uninterrupted reading time."
        if feature == "Apps_Used_Daily":
            return f"Using {value:g} apps increases switching while reading or writing."
        if feature == "Sleep_Hours":
            if value >= 7.5:
                return f"{value:g}h sleep is supporting recall and language focus."
            return f"Only {value:g}h sleep can weaken recall and language focus."
        return f"{_human_feature(feature).title()} is the main pressure on English."

    if subject == "maths":
        if feature == "Time_on_Gaming":
            return f"Gaming time ({value:g}h) is competing with maths practice."
        if feature == "Phone_Checks_Per_Day":
            return f"{value:g} phone checks can break problem-solving concentration."
        if feature == "Daily_Usage_Hours":
            return f"{value:g}h total phone use leaves less time for maths drills."
        if feature == "Sleep_Hours":
            if value >= 7.5:
                return f"{value:g}h sleep is supporting calculation accuracy."
            return f"Only {value:g}h sleep can reduce accuracy in calculations."
        if feature == "Apps_Used_Daily":
            return f"Using {value:g} apps can interrupt multi-step maths practice."
        if feature == "Time_on_Social_Media":
            return f"{value:g}h social media can break concentration during maths work."
        return f"{_human_feature(feature).title()} is the main pressure on Maths."

    if feature == "Daily_Usage_Hours":
        return f"{value:g}h total phone use is the strongest pressure on science study."
    if feature == "Time_on_Social_Media":
        return f"{value:g}h social media can interrupt concept review and retention."
    if feature == "Time_on_Gaming":
        return f"{value:g}h gaming reduces time for science practice and revision."
    if feature == "Sleep_Hours":
        if value >= 7.5:
            return f"{value:g}h sleep is supporting memory for science concepts."
        return f"Only {value:g}h sleep can weaken memory for science concepts."
    if feature == "Apps_Used_Daily":
        return f"Using {value:g} apps can fragment science reading and revision."
    if feature == "Phone_Checks_Per_Day":
        return f"{value:g} phone checks can interrupt science observation and recall."
    return f"{_human_feature(feature).title()} is the main pressure on Science."


def _dashboard_subject_title(subject: str, feature: str) -> str:
    if feature == "Healthy_Routine":
        return "Stable Routine"
    titles = {
        "english": {
            "Time_on_Social_Media": "Reading Focus",
            "Daily_Usage_Hours": "Study Time",
            "Apps_Used_Daily": "Writing Flow",
            "Sleep_Hours": "Language Recall",
            "Phone_Checks_Per_Day": "Attention Span",
            "Time_on_Gaming": "Reading Routine",
        },
        "maths": {
            "Time_on_Gaming": "Practice Time",
            "Phone_Checks_Per_Day": "Problem Focus",
            "Daily_Usage_Hours": "Drill Time",
            "Sleep_Hours": "Calculation Accuracy",
            "Apps_Used_Daily": "Step-by-Step Focus",
            "Time_on_Social_Media": "Concentration",
        },
        "science": {
            "Daily_Usage_Hours": "Concept Review",
            "Time_on_Social_Media": "Memory Retention",
            "Time_on_Gaming": "Revision Time",
            "Sleep_Hours": "Concept Memory",
            "Apps_Used_Daily": "Learning Continuity",
            "Phone_Checks_Per_Day": "Observation Focus",
        },
    }
    return titles.get(subject, {}).get(feature, "Subject Impact")


def _subject_fix(subject: str, feature: str, feature_input: FeatureInput) -> str:
    subject_name = subject.title()
    if feature == "Healthy_Routine":
        return f"Maintain the current routine and review {subject_name} again if screen time increases."
    if feature == "Daily_Usage_Hours":
        target = max(2.5, min(4.0, feature_input.Daily_Usage_Hours - 1.5))
        return f"Reduce total phone use toward {target:g}h and reserve one fixed block for {subject_name} practice."
    if feature == "Time_on_Social_Media":
        target = max(0.75, feature_input.Time_on_Social_Media - 1)
        return f"Limit social media to about {target:g}h and do {subject_name} before opening social apps."
    if feature == "Time_on_Gaming":
        target = max(0.5, feature_input.Time_on_Gaming - 1)
        return f"Limit gaming to about {target:g}h and use the saved time for {subject_name} problem practice."
    if feature == "Sleep_Hours":
        return f"Move toward 7.5h sleep and place {subject_name} revision earlier in the evening."
    if feature == "Apps_Used_Daily":
        return f"Keep study-time apps to 8 or fewer and close nonessential apps while studying {subject_name}."
    if feature == "Phone_Checks_Per_Day":
        return f"Check the phone only after 25-minute {subject_name} study blocks, aiming for fewer than 60 checks."
    return f"Use a focused {subject_name} revision block and review progress after one week."


def _build_subject_effects(
    feature_input: FeatureInput,
    prediction: Dict[str, Any],
    feature_impacts: Dict[str, Any],
) -> Dict[str, Any]:
    subject_marks = {
        "english": prediction["english"],
        "maths": prediction["maths"],
        "science": prediction["science"],
    }
    effects = {}
    for subject, mark in subject_marks.items():
        risks = _habit_risk_scores(feature_input, feature_impacts.get(subject, {}))
        top_risks = risks[:2]
        if not top_risks:
            top_risks = [
                {
                    "feature": "Healthy_Routine",
                    "label": _human_feature("Healthy_Routine"),
                    "value": 0,
                    "score": 0,
                    "model_penalty": 0,
                }
            ]

        effects[subject] = {
            "predicted_mark": mark,
            "affected_by": [
                {
                    **risk,
                    "dashboard_title": _dashboard_subject_title(subject, risk["feature"]),
                    "dashboard_message": _dashboard_subject_message(subject, risk["feature"], risk["value"]),
                    "reason": _effect_reason(subject, risk["feature"], risk["value"], mark),
                    "fix": _subject_fix(subject, risk["feature"], feature_input),
                }
                for risk in top_risks
            ],
        }
    return effects


def _build_action_plan(
    feature_input: FeatureInput,
    prediction: Dict[str, Any],
    feature_impacts: Dict[str, Any],
) -> Dict[str, Any]:
    marks = {
        "English": prediction["english"],
        "Maths": prediction["maths"],
        "Science": prediction["science"],
    }
    subject = _most_affected_subject(marks)
    no_affected_subject = subject == "No affected subject"
    subject_key = subject.lower() if subject in marks else None
    subject_impacts = feature_impacts.get(subject_key, {}) if subject_key else _aggregate_subject_impacts(feature_impacts)
    ranked_risks = _habit_risk_scores(feature_input, subject_impacts)
    has_actionable_risk = bool(ranked_risks) and not no_affected_subject
    biggest = ranked_risks[0] if has_actionable_risk else {
        "feature": "Daily_Usage_Hours",
        "label": "no major subject risk" if no_affected_subject else "screen routine",
        "value": feature_input.Daily_Usage_Hours,
        "score": 0,
    }
    positive_feature = _strongest_positive_factor(subject_impacts)
    addiction = prediction["addiction"].lower()
    subject_text = _subject_phrase(subject)
    subject_mark = _subject_mark(marks, subject)
    close_subjects = subject == "All subjects"

    recommendations = []
    if no_affected_subject:
        recommendations.append(
            _elaborate_recommendation(
                "Maintain strong performance",
                "Keep the current study routine steady and avoid adding extra entertainment screen time.",
                "All three predicted subject marks are above 80, so the model does not flag an affected subject.",
                "Review the dashboard again only if screen time increases or sleep drops.",
            )
        )
    elif not has_actionable_risk:
        recommendations.append(
            _elaborate_recommendation(
                "Maintain the current screen routine",
                f"Keep daily phone usage close to {feature_input.Daily_Usage_Hours:g}h and do not add new entertainment apps.",
                f"The model did not find a major screen habit above the action threshold, but {subject_text} should be supported together.",
                f"Review {subject_text} after one week and only change the plan if the mark does not improve.",
            )
        )
    elif biggest["feature"] == "Daily_Usage_Hours":
        target = max(2.5, min(4.0, feature_input.Daily_Usage_Hours - 1.5))
        recommendations.append(
            _elaborate_recommendation(
                "Reduce total daily phone time",
                f"Bring usage down from {feature_input.Daily_Usage_Hours:g}h to about {target:g}h for the next 7 days.",
                f"The model ranked daily phone usage as the strongest risk for {subject_text}, with the lowest predicted mark at {subject_mark}.",
                "Track the daily total each evening and compare the next predicted report after one week.",
            )
        )
    elif biggest["feature"] == "Time_on_Social_Media":
        target = max(0.75, feature_input.Time_on_Social_Media - 1)
        recommendations.append(
            _elaborate_recommendation(
                "Control social media exposure",
                f"Cut social media from {feature_input.Time_on_Social_Media:g}h to about {target:g}h and move the saved time to {subject_text}.",
                f"Social media is a negative factor for the current prediction, and {subject_text} need the most support.",
                "Use an app timer and keep social media after the main study block.",
            )
        )
    elif biggest["feature"] == "Time_on_Gaming":
        target = max(0.5, feature_input.Time_on_Gaming - 1)
        recommendations.append(
            _elaborate_recommendation(
                "Limit gaming before study is complete",
                f"Reduce gaming from {feature_input.Time_on_Gaming:g}h to about {target:g}h until {subject_text} improve.",
                f"Gaming is pulling risk upward in the model, and the lowest predicted mark is {subject_mark}.",
                "Allow gaming only after the planned subject revision is finished.",
            )
        )
    elif biggest["feature"] == "Sleep_Hours":
        recommendations.append(
            _elaborate_recommendation(
                "Prioritize sleep recovery",
                f"Increase sleep from {feature_input.Sleep_Hours:g}h toward 7.5h for the next week.",
                f"Low sleep is the strongest changeable risk, and the lowest predicted mark is {subject_text} at {subject_mark}.",
                "Move entertainment phone use earlier and set a fixed sleep time.",
            )
        )
    elif biggest["feature"] == "Apps_Used_Daily":
        recommendations.append(
            _elaborate_recommendation(
                "Reduce app switching",
                f"Cut app usage from {feature_input.Apps_Used_Daily} apps to 8 or fewer on study days.",
                (
                    "Frequent app switching increases distraction risk, and all three subject predictions are close enough to improve together."
                    if close_subjects
                    else f"Frequent app switching increases distraction risk, especially when {subject_text} already have the weakest predicted mark."
                ),
                "Keep only study, communication, and essential apps available during homework time.",
            )
        )
    elif biggest["feature"] == "Phone_Checks_Per_Day":
        recommendations.append(
            _elaborate_recommendation(
                "Reduce phone checking frequency",
                f"Lower checks from {feature_input.Phone_Checks_Per_Day} to under 60 per day using fixed check-in times.",
                f"Repeated checking fragments attention, and the model links the current pattern with weaker {subject_text} performance.",
                "Check the phone only after completing 25-minute study blocks.",
            )
        )
    else:
        recommendations.append(
            _elaborate_recommendation(
                "Strengthen subject practice",
                f"Add one focused {subject_text} revision block daily.",
                f"The submitted screen routine is used to predict that {subject_text} currently have the lowest predicted area.",
                "Use a short daily quiz or worksheet to measure improvement.",
            )
        )

    if no_affected_subject:
        recommendations.append(
            _elaborate_recommendation(
                "Protect the current routine",
                "Keep sleep consistent and keep study time before entertainment apps.",
                "No subject is currently flagged as affected, so the best action is preserving the healthy pattern.",
                "Watch for rising screen time or lower sleep in the next report.",
            )
        )
    elif positive_feature in {"Sleep_Hours", "Academic_Performance"}:
        recommendations.append(
            _elaborate_recommendation(
                "Use sleep as the support factor",
                f"Do {subject_text} revision earlier in the evening and protect a consistent sleep target.",
                "Sleep is one of the strongest positive factors in the subject-impact analysis.",
                "Record sleep hours next to study time so the counselor can compare both in the next report.",
            )
        )
    else:
        recommendations.append(
            _elaborate_recommendation(
                "Add focused subject practice",
                f"Schedule 25 minutes of {subject_text} practice before entertainment screen time.",
                f"This directly addresses the weakest predicted subject instead of only reducing phone use.",
                "Use the same time slot every day so the habit is easy to repeat.",
            )
        )

    if no_affected_subject:
        recommendations.append(
            _elaborate_recommendation(
                "Keep balanced screen use",
                "Keep entertainment apps after homework and maintain the current sleep schedule.",
                "The subject predictions are healthy, so the goal is prevention rather than correction.",
                "Recheck after a week if any input value changes sharply.",
            )
        )
    elif not has_actionable_risk:
        recommendations.append(
            _elaborate_recommendation(
                "Monitor without overcorrecting",
                "Keep entertainment time stable rather than making a major restriction.",
                f"The current pattern does not show a large screen-risk spike, even though the predicted addiction label is {addiction}.",
                "Re-run the dashboard after seven days with updated values.",
            )
        )
    elif prediction["addiction"] == "High":
        recommendations.append(
            _elaborate_recommendation(
                "Use short-term app limits",
                "Set app limits on the largest entertainment category for the next 3 days.",
                "The addiction classifier returned High risk, so the first goal is to reduce exposure quickly and visibly.",
                "If the risk remains High, extend the limit for another week and involve a parent or counselor.",
            )
        )
    elif prediction["addiction"] == "Medium":
        recommendations.append(
            _elaborate_recommendation(
                "Move entertainment after homework",
                "Use entertainment apps only after the main homework block is complete.",
                "The model returned Medium risk, so structure may be enough without a strict ban.",
                "Compare the predicted marks next week to see whether this routine helped.",
            )
        )
    else:
        recommendations.append(
            _elaborate_recommendation(
                "Maintain low-risk habits",
                "Keep the current screen level and avoid adding extra entertainment time.",
                "The model returned Low risk, so the priority is preserving the useful routine.",
                f"Use any extra time for {subject_text}, since it is still the lowest predicted area.",
            )
        )

    return {
        "marks": marks,
        "most_affected_subject": subject,
        "most_affected_mark": _subject_mark(marks, subject),
        "biggest_risk": biggest,
        "ranked_risks": ranked_risks[:3],
        "positive_feature": positive_feature,
        "has_actionable_risk": has_actionable_risk,
        "recommendations": recommendations[:3],
    }


def _fallback_analysis(
    feature_input: FeatureInput,
    prediction: Dict[str, Any],
    feature_impacts: Dict[str, Any],
) -> Dict[str, Any]:
    plan = _build_action_plan(feature_input, prediction, feature_impacts)
    subject_effects = _build_subject_effects(feature_input, prediction, feature_impacts)
    subject = plan["most_affected_subject"]
    risk = plan["biggest_risk"]
    risk_name = risk.get("label") or _human_feature(risk["feature"])
    subject_text = _subject_phrase(subject)
    tied_all = subject == "All subjects"
    no_affected_subject = subject == "No affected subject"
    ranked_text = "; ".join(
        f"{item['label']}={item['value']:g} (risk score {item['score']})"
        for item in plan["ranked_risks"]
    ) or "no major screen habit crossed the action threshold"
    subject_parts = []
    for subject_key, effect in subject_effects.items():
        drivers = ", ".join(item["label"] for item in effect["affected_by"])
        subject_parts.append(f"{subject_key.title()} {effect['predicted_mark']} affected mainly by {drivers}")
    subject_summary = "; ".join(subject_parts)
    audio_narration = (
        f"Student risk report. The predicted addiction risk is {prediction['addiction']}. "
        f"English is predicted at {prediction['english']}, Maths at {prediction['maths']}, and Science at {prediction['science']}. "
        f"The submitted values show {feature_input.Daily_Usage_Hours:g} hours of daily phone use, "
        f"{feature_input.Time_on_Social_Media:g} hours on social media, {feature_input.Time_on_Gaming:g} hours on gaming, "
        f"{feature_input.Sleep_Hours:g} hours of sleep, {feature_input.Apps_Used_Daily} apps used daily, "
        f"and {feature_input.Phone_Checks_Per_Day} phone checks per day. "
        + (
            "All three subject predictions are above 80, so there is no affected subject. "
            if no_affected_subject
            else
            f"All subjects are tied, so there is no single most affected subject; the main habit to watch is {risk_name}. "
            if tied_all
            else f"The most affected subject is {subject}, mainly because of {risk_name}. "
        )
        + f"The first recommendation is: {plan['recommendations'][0]}"
    )

    return {
        "counselor_insight": (
            f"The addiction classifier places this student in the {prediction['addiction']} risk band. "
            "The academic layer predicts subject marks from the submitted screen exposure, sleep, app switching, and phone-check frequency. "
            + (
                "After those adjustments, all three subjects are above 80, so there is no affected subject. "
                if no_affected_subject
                else f"After those adjustments, all three subjects are close enough to treat together at about {plan['most_affected_mark']}; improve all three rather than singling out one subject. "
                if tied_all
                else f"After those adjustments, {subject_text} is the weakest predicted area at {plan['most_affected_mark']}. "
            )
            + f"The ranked risk drivers for this profile are: {ranked_text}. "
            + f"Subject-wise interpretation: {subject_summary}. "
            + (
                "No subject is flagged as affected, so the priority is maintaining the current routine."
                if no_affected_subject
                else f"The priority intervention is {risk_name} because it is high in the submitted data and aligns with the weakest subject's impact pattern."
                if plan["has_actionable_risk"]
                else "No screen habit crossed the action threshold, so the counselor focus should be maintaining the routine while improving subject practice."
            )
        ),
        "parent_summary": (
            (
                f"Your child is in the {prediction['addiction'].lower()} risk group. All predicted subject marks are above 80, so there is no affected subject, and "
                if no_affected_subject
                else f"Your child is in the {prediction['addiction'].lower()} risk group. The three subject marks are close enough, so all three subjects should be improved together, and "
                if tied_all
                else f"Your child is in the {prediction['addiction'].lower()} risk group. {subject} is the lowest predicted mark "
                f"({plan['most_affected_mark']}), and "
            )
            + (
                "the current routine should be maintained."
                if no_affected_subject
                else f"the main habit to change is {risk_name}."
                if plan["has_actionable_risk"]
                else "the current screen routine should be maintained while adding subject practice."
            )
        ),
        "recommendations": plan["recommendations"],
        "biggest_risk_factor": risk_name,
        "most_affected_subject": subject,
        "subject_effects": subject_effects,
        "audio_narration": audio_narration,
        "ai_source": "Model-driven local analysis; create a .env file with XAI_API_KEY to use Grok wording.",
    }


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_analysis(
    feature_input: FeatureInput,
    prediction: Dict[str, Any],
    feature_impacts: Dict[str, Any],
) -> Dict[str, Any]:
    plan = _build_action_plan(feature_input, prediction, feature_impacts)
    subject_effects = _build_subject_effects(feature_input, prediction, feature_impacts)
    payload = {
        "addiction": prediction["addiction"],
        "marks": {
            "eng": prediction["english"],
            "math": prediction["maths"],
            "sci": prediction["science"],
        },
        "usage": feature_input.to_public_dict(),
        "top_negative": {
            subject: values["negative"] for subject, values in feature_impacts.items()
        },
        "top_positive": {
            subject: values["positive"] for subject, values in feature_impacts.items()
        },
        "model_driven_action_plan": plan,
        "subject_effects": subject_effects,
    }

    xai_api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not xai_api_key:
        return _fallback_analysis(feature_input, prediction, feature_impacts)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=xai_api_key,
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        )
        prompt = (
            "Return strict JSON with counselor_insight, parent_summary, recommendations, biggest_risk_factor, most_affected_subject, audio_narration.\n"
            "Also include subject_effects explaining which submitted habits affect English, Maths, and Science.\n"
            "Use the model_driven_action_plan exactly; do not give generic advice or introduce new factors.\n"
            f"If all subject marks are above {NO_AFFECTED_MARK_THRESHOLD:g}, most_affected_subject must be 'No affected subject' and the message must say there is no affected subject.\n"
            f"If the subject marks differ by {SUBJECT_CLOSE_THRESHOLD:g} points or less, most_affected_subject must be 'All subjects' and the message must recommend improving all three subjects together.\n"
            "Make counselor_insight detailed: explain model reasoning, risk drivers, weakest subject, and subject-by-subject effects.\n"
            "Make each recommendation 2-3 practical sentences tied to predicted addiction, weakest subject, marks, and numeric usage.\n"
            "Make audio_narration a parent-friendly spoken script using the risk, input values, affected subject, and recommendations.\n"
            f"Analyze this student decision-support payload: {json.dumps(payload)}"
        )
        response = client.chat.completions.create(
            model=os.getenv("GROK_MODEL", "grok-3"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise student risk analyst. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        parsed = _parse_json_response(response.choices[0].message.content.strip())
        if len(parsed.get("recommendations", [])) != 3:
            parsed["recommendations"] = parsed.get("recommendations", [])[:3]
        if len(parsed.get("recommendations", [])) < 3:
            parsed["recommendations"] = plan["recommendations"]
        parsed["biggest_risk_factor"] = plan["biggest_risk"].get("label") or _human_feature(plan["biggest_risk"]["feature"])
        parsed["most_affected_subject"] = plan["most_affected_subject"]
        parsed.setdefault("subject_effects", subject_effects)
        parsed.setdefault(
            "audio_narration",
            _fallback_analysis(feature_input, prediction, feature_impacts)["audio_narration"],
        )
        parsed["ai_source"] = f"Grok API ({os.getenv('GROK_MODEL', 'grok-3')})"
        return parsed
    except Exception as exc:
        analysis = _fallback_analysis(feature_input, prediction, feature_impacts)
        analysis["ai_source"] = f"Local fallback; Grok call failed: {exc}"
        return analysis
