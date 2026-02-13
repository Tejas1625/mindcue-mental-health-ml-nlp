from __future__ import annotations
import json
import os
from datetime import datetime
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
# --- Add imports for SHAP and NumPy ---
import shap
import numpy as np
# --------------------------------------

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "models", "mindcue_textclf.joblib")
LOG_PATH = os.path.join(APP_ROOT, "logs", "predictions.csv")

app = Flask(__name__)
CORS(app)

# Lazy load model and explainer
_model = None
_explainer = None


def get_model_and_explainer():
    """
    Loads the model and initializes the SHAP explainer on the first call.
    """
    global _model, _explainer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python train.py` first."
            )
        _model = joblib.load(MODEL_PATH)

        # --- CORRECTED SHAP INITIALIZATION ---
        # 1. Create a wrapper function that SHAP can reliably call.
        #    This function takes a NumPy array of strings and returns the model's probabilities.
        def f(x):
            return _model.predict_proba(x)

        # 2. Create a Text masker that splits text into words, similar to TF-IDF.
        masker = shap.maskers.Text(r"\W+")

        # 3. Initialize the Explainer with the wrapper function and the text masker.
        #    This is a more robust pattern for text-based scikit-learn pipelines.
        _explainer = shap.Explainer(f, masker, output_names=_model.classes_.tolist())
        # ----------------------------------------

    return _model, _explainer


def safe_predict_and_explain(text: str):
    """
    Generates a prediction, probability map, and SHAP explanation for the input text.
    """
    model, explainer = get_model_and_explainer()

    # Get standard prediction and probabilities
    label = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    proba_map = {c: float(p) for c, p in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)}

    # --- Generate the SHAP explanation ---
    shap_values = explainer([text])
    instance_explanation = shap_values[0]
    class_index = np.where(model.classes_ == label)[0][0]
    words = instance_explanation.data
    sv_for_class = instance_explanation.values[:, class_index]

    # Combine words and their SHAP values, sort by importance, and take the top 7
    explanation_list = sorted(
        list(zip(words, sv_for_class)),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:7]
    explanation_list = [item for item in explanation_list if item[1] != 0]
    # -----------------------

    return label, proba_map, explanation_list


def log_prediction(text: str, label: str, proba_map: dict | None):
    """
    Logs prediction metadata to a CSV file.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "text_len": len(text or ""),
        "label": label,
        "top1_prob": max(proba_map.values()) if proba_map else None,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", index=False, header=header)


@app.get("/")
def index():
    """
    Serves the main HTML page.
    """
    return render_template("index.html")


@app.post("/predict")
def predict():
    """
    Handles prediction requests, now including SHAP explanations.
    """
    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Empty text."}), 400

        label, proba_map, explanation = safe_predict_and_explain(text)
        response = {"label": label, "proba": proba_map, "explanation": explanation}

        if label.lower() == "suicidal":
            response["safety"] = ("If you or someone you know is in immediate danger, "
                                  "please seek local emergency help or talk to someone you trust. "
                                  "Call 112 [Emergencies] urgently.")

        # Now this call will work correctly
        log_prediction(text, label, proba_map)
        return jsonify(response)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


