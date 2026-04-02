"""
Credit Scoring Model - Flask Backend API
Supports the Kaggle-trained model with 32 features.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load Model & Artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

def load_artifacts():
    model        = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    scaler       = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'r') as f:
        metrics  = json.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    return model, scaler, metrics, feature_names

try:
    model, scaler, metrics, feature_names = load_artifacts()
    MODEL_LOADED = True
    print(" Model loaded successfully")
    print(f"   Features expected ({len(feature_names)}): {feature_names}")
except Exception as e:
    MODEL_LOADED = False
    print(f"  Model not loaded: {e}. Run train_kaggle.py first.")


# Helper Functions

def compute_credit_score(probability):
    score = int(300 + (1 - probability) * 550)
    return min(850, max(300, score))

def get_credit_tier(score):
    if score >= 750: return ("Excellent", "#00C896", "AAA")
    elif score >= 700: return ("Good",      "#4CAF50", "AA")
    elif score >= 650: return ("Fair",      "#FFC107", "BB")
    elif score >= 600: return ("Poor",      "#FF7043", "CC")
    else:              return ("Very Poor", "#F44336", "D")

def get_recommendations(data, score):
    recs = []
    if data.get('missed_payments', 0) > 0:
        recs.append({"icon": "💳", "title": "Improve Payment History",
                     "desc": "Set up auto-pay to eliminate missed payments — it's 35% of your score."})
    if data.get('debt_to_income', 0) > 0.4:
        recs.append({"icon": "📉", "title": "Reduce Debt-to-Income",
                     "desc": "Your DTI is high. Paying down existing debts can significantly improve your profile."})
    if data.get('credit_utilization', 0) > 0.3:
        recs.append({"icon": "🔢", "title": "Lower Credit Utilization",
                     "desc": "Try to keep credit utilization under 30%. Consider requesting a credit limit increase."})
    if data.get('credit_history_years', 5) < 3:
        recs.append({"icon": "⏳", "title": "Build Credit History",
                     "desc": "Longer credit history improves your score. Avoid closing old accounts."})
    if data.get('num_credit_inquiries', 0) > 3:
        recs.append({"icon": "🔍", "title": "Limit Hard Inquiries",
                     "desc": "Too many recent inquiries signal risk. Space out credit applications."})
    if data.get('loan_int_rate', 0) > 15:
        recs.append({"icon": "📊", "title": "High Interest Rate",
                     "desc": "Your loan carries a high interest rate. Improving your credit score can help you refinance at better rates."})
    if not recs:
        recs.append({"icon": "⭐", "title": "Maintain Your Profile",
                     "desc": "Your credit profile looks strong! Keep up the good habits."})
    return recs


# Build Full Feature Vector 

def build_features(data: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with ALL features the trained model expects.
    The model was trained with train_kaggle.py which may include extra columns
    (int_rate_norm, grade_util_interaction, prev_default, has_rent,
     loan_grade_num, intent_* one-hots).
    We reconstruct all of them from the API payload.
    """

    # Core inputs
    age                  = float(data.get('age', 30))
    income               = float(data.get('income', 50000))
    employment_years     = float(data.get('employment_years', 3))
    debt_to_income       = float(data.get('debt_to_income', 0.3))
    credit_utilization   = float(data.get('credit_utilization', 0.3))
    missed_payments      = float(data.get('missed_payments', 0))
    num_open_accounts    = float(data.get('num_open_accounts', 3))
    num_credit_inquiries = float(data.get('num_credit_inquiries', 1))
    total_debt           = float(data.get('total_debt', income * debt_to_income))
    credit_history_years = float(data.get('credit_history_years', 5))
    savings_balance      = float(data.get('savings_balance', 5000))
    monthly_expenses     = float(data.get('monthly_expenses', 2000))
    loan_amount          = float(data.get('loan_amount', 15000))
    num_dependents       = float(data.get('num_dependents', 0))
    has_mortgage         = int(bool(data.get('has_mortgage', False)))
    has_auto_loan        = int(bool(data.get('has_auto_loan', False)))

    # Extra Kaggle-dataset fields 
    loan_int_rate        = float(data.get('loan_int_rate', 11.0))   # typical market rate
    loan_grade           = str(data.get('loan_grade', 'C'))          # A–G
    loan_intent          = str(data.get('loan_intent', 'PERSONAL')).upper()
    prev_default         = int(bool(data.get('prev_default', False)))
    has_rent             = int(not has_mortgage and not bool(data.get('has_own', False)))

    # Loan grade → ordinal  A=6 … G=0
    grade_map    = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
    loan_grade_num = float(grade_map.get(loan_grade.upper(), 3))

    #  Engineered features 
    payment_stress          = missed_payments * debt_to_income
    income_per_dependent    = income / (num_dependents + 1)
    savings_ratio           = savings_balance / (income + 1)
    monthly_debt_burden     = (total_debt / 12) / (income / 12 + 1)
    credit_experience_score = credit_history_years * (1 - credit_utilization)
    loan_to_income          = loan_amount / (income + 1)
    int_rate_norm           = loan_int_rate / 30.0
    grade_util_interaction  = loan_grade_num * credit_utilization

    # Intent one-hot columns (top 5 from training)
    # Training used: EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION
    known_intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION']
    intent_flags  = {f'intent_{i.lower()}': int(loan_intent == i) for i in known_intents}

    # Assemble base row 
    row = {
        'age'                   : age,
        'income'                : income,
        'employment_years'      : employment_years,
        'debt_to_income'        : debt_to_income,
        'credit_utilization'    : credit_utilization,
        'missed_payments'       : missed_payments,
        'num_open_accounts'     : num_open_accounts,
        'num_credit_inquiries'  : num_credit_inquiries,
        'total_debt'            : total_debt,
        'credit_history_years'  : credit_history_years,
        'savings_balance'       : savings_balance,
        'monthly_expenses'      : monthly_expenses,
        'loan_amount'           : loan_amount,
        'num_dependents'        : num_dependents,
        'has_mortgage'          : has_mortgage,
        'has_auto_loan'         : has_auto_loan,
        # Engineered
        'payment_stress'        : payment_stress,
        'income_per_dependent'  : income_per_dependent,
        'savings_ratio'         : savings_ratio,
        'monthly_debt_burden'   : monthly_debt_burden,
        'credit_experience_score': credit_experience_score,
        'loan_to_income'        : loan_to_income,
        # Kaggle extras
        'int_rate_norm'         : int_rate_norm,
        'grade_util_interaction': grade_util_interaction,
        'prev_default'          : prev_default,
        'has_rent'              : has_rent,
        'loan_grade_num'        : loan_grade_num,
        **intent_flags,
    }

    df = pd.DataFrame([row])

    # Align to exact feature list saved during training 
    # Add any missing columns with 0, drop any extras, reorder
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]

    return df


# Routes
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file."""
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": MODEL_LOADED,
        "timestamp"   : datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not trained yet. Run: python train_kaggle.py"}), 503

    try:
        data = request.get_json()

        features  = build_features(data)
        X_scaled  = scaler.transform(features.values)  # Convert to numpy to suppress sklearn warning

        prob_default = float(model.predict_proba(X_scaled)[0][1])
        prediction   = int(model.predict(X_scaled)[0])

        credit_score            = compute_credit_score(prob_default)
        tier, color, grade      = get_credit_tier(credit_score)
        recommendations         = get_recommendations(data, credit_score)

        # Feature importances
        feature_importances = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fi_pairs    = sorted(zip(feature_names, importances),
                                 key=lambda x: x[1], reverse=True)[:8]
            feature_importances = [
                {"feature": k.replace('_', ' ').title(), "importance": round(float(v) * 100, 1)}
                for k, v in fi_pairs
            ]

        return jsonify({
            "success": True,
            "prediction": {
                "default_risk": round(prob_default * 100, 1),
                "approved"    : prediction == 0,
                "credit_score": credit_score,
                "grade"       : grade,
                "tier"        : tier,
                "tier_color"  : color,
            },
            "feature_importances": feature_importances,
            "recommendations"    : recommendations,
            "timestamp"          : datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not trained yet"}), 503
    return jsonify(metrics)


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not trained yet"}), 503
    try:
        records = request.get_json()
        results = []
        for rec in records:
            features = build_features(rec)
            X_scaled = scaler.transform(features)
            prob     = float(model.predict_proba(X_scaled)[0][1])
            score    = compute_credit_score(prob)
            tier, color, grade = get_credit_tier(score)
            results.append({
                "id"          : rec.get("id", "N/A"),
                "credit_score": score,
                "grade"       : grade,
                "tier"        : tier,
                "default_risk": round(prob * 100, 1),
                "approved"    : int(model.predict(X_scaled)[0]) == 0
            })
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')