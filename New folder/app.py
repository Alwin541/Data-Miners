from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for

from services.ai_service import generate_analysis
from services.graph_service import generate_graphs
from services.ml_service import FeatureInput, predict_student_risk
from services.tts_service import build_parent_audio_script, generate_parent_audio


load_dotenv()

app = Flask(__name__)
app.secret_key = "replace-this-secret-for-production"


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    feature_input = FeatureInput.from_form(request.form)
    prediction = predict_student_risk(feature_input)
    graphs = generate_graphs(feature_input, prediction)
    analysis = generate_analysis(feature_input, prediction, graphs["feature_impacts"])
    audio_script = analysis.get("audio_narration") or build_parent_audio_script(
        analysis, prediction, feature_input.to_public_dict()
    )
    audio_path = generate_parent_audio(audio_script)

    session["student_input"] = feature_input.to_public_dict()
    session["prediction"] = prediction
    session["graphs"] = graphs
    session["analysis"] = analysis
    session["audio_path"] = audio_path
    session["audio_script"] = audio_script

    return redirect(url_for("dashboard"))


def _result_context():
    if "prediction" not in session:
        return None

    return {
        "student_input": session.get("student_input", {}),
        "prediction": session.get("prediction", {}),
        "graphs": session.get("graphs", {}),
        "analysis": session.get("analysis", {}),
        "audio_path": session.get("audio_path"),
        "audio_script": session.get("audio_script"),
    }


@app.route("/dashboard")
def dashboard():
    context = _result_context()
    if context is None:
        return redirect(url_for("index"))
    return render_template("dashboard.html", **context)


@app.route("/parent-report")
def parent_report():
    context = _result_context()
    if context is None:
        return redirect(url_for("index"))
    return render_template("parent_report.html", **context)


@app.route("/subject-analysis")
def subject_analysis():
    context = _result_context()
    if context is None:
        return redirect(url_for("index"))
    return render_template("subject_analysis.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
