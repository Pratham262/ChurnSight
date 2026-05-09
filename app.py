from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "adaboost_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

# ── Feature order must match training ───────────────────────────────────────
FEATURES = [
    "age", "gender", "tenure", "usage_frequency",
    "support_calls", "payment_delay", "subscription_type",
    "contract_length", "total_spend", "last_interaction"
]

# ── Retention logic ──────────────────────────────────────────────────────────
def retention_action(prob: float) -> dict:
    if prob > 0.70:
        return {
            "level": "High",
            "action": "Offer Discount or Promotional Incentive",
            "color": "red",
            "icon": "🚨"
        }
    elif prob >= 0.50:
        return {
            "level": "Medium",
            "action": "Proactive Customer Support Intervention",
            "color": "amber",
            "icon": "⚠️"
        }
    else:
        return {
            "level": "Low",
            "action": "Continue Normal Engagement",
            "color": "green",
            "icon": "✅"
        }

def predict_single(features: list) -> dict:
    arr    = np.array(features).reshape(1, -1)
    scaled = scaler.transform(arr)
    prob   = float(model.predict_proba(scaled)[0][1])
    rec    = retention_action(prob)
    return {
        "churn_probability": round(prob * 100, 2),
        "verdict": "Likely to Churn" if prob >= 0.50 else "Likely to Stay",
        "risk_level": rec["level"],
        "action": rec["action"],
        "color": rec["color"],
        "icon": rec["icon"]
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result = predict_single(features)
    return jsonify(result)

@app.route("/api/scenario", methods=["POST"])
def scenario():
    data = request.get_json()
    try:
        features = [float(data[f]) for f in FEATURES]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    payment_idx = FEATURES.index("payment_delay")
    multipliers = [1.0, 1.2, 1.5, 2.0]
    results = []

    for m in multipliers:
        f = features.copy()
        f[payment_idx] = f[payment_idx] * m
        r = predict_single(f)
        r["multiplier"] = m
        r["label"] = f"×{m} Payment Delay"
        results.append(r)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
