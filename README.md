# ChurnSight — Flask Web App

AI-powered customer churn prediction and scenario simulator.

## Project Structure

```
churnsight_app/
├── app.py                  # Flask backend (API + routing)
├── requirements.txt        # Python dependencies
├── adaboost_model.pkl      # ← Place your model pickle here
├── scaler.pkl              # ← Place your scaler pickle here
└── templates/
    └── index.html          # Tailwind CSS dashboard
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your pickle files
Copy `adaboost_model.pkl` and `scaler.pkl` into the `churnsight_app/` folder.

> If your files have different names, set environment variables:
> ```bash
> export MODEL_PATH=my_model.pkl
> export SCALER_PATH=my_scaler.pkl
> ```

### 3. Run the app
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## API Endpoints

### `POST /api/predict`
Single customer churn prediction.

**Request body:**
```json
{
  "age": 35,
  "gender": 1,
  "tenure": 12,
  "usage_frequency": 15,
  "support_calls": 2,
  "payment_delay": 5,
  "subscription_type": 1,
  "contract_length": 1,
  "total_spend": 500,
  "last_interaction": 30
}
```

**Response:**
```json
{
  "churn_probability": 73.4,
  "verdict": "Likely to Churn",
  "risk_level": "High",
  "action": "Offer Discount or Promotional Incentive",
  "color": "red",
  "icon": "🚨"
}
```

### `POST /api/scenario`
Simulates churn probability across 4 payment-delay multipliers (×1.0, ×1.2, ×1.5, ×2.0).

Same request body as `/api/predict`. Returns an array of 4 results.

---

## Feature Encoding

| Feature           | Values                                      |
|-------------------|---------------------------------------------|
| gender            | 1 = Male, 0 = Female                        |
| subscription_type | 0 = Basic, 1 = Standard, 2 = Premium        |
| contract_length   | 0 = Quarterly, 1 = Monthly, 2 = Annual      |

---

## Risk Thresholds

| Probability    | Risk   | Action                              |
|----------------|--------|-------------------------------------|
| > 70%          | High   | Offer Discount / Incentive          |
| 50% – 70%      | Medium | Customer Support Intervention       |
| < 50%          | Low    | Continue Normal Engagement          |
