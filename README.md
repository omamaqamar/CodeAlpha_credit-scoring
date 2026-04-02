# CreditIQ — Credit Scoring ML System

**CodeAlpha Internship Project**

A production-ready ML application that predicts creditworthiness (300-850 credit score) using a Random Forest classifier trained on Kaggle credit data.

[Live Demo](http://localhost:5000)
[GitHub](https://github.com/omamaqamar/CodeAlpha_credit-scoring)

## Overview

Features:
- Random Forest with 32 engineered financial features
- Interactive web dashboard (dark theme)
- Sub-100ms API responses
- REST API with CORS support
- Real-time model metrics

## Project Structure

credit-scoring/
├── backend/app.py                 # Flask API
├── frontend/index.html            # Web dashboard
├── model/                         # ML artifacts
│   ├── best_model.pkl             # Trained model
│   ├── scaler.pkl                 # Feature scaler
│   ├── feature_names.json         # Feature order
│   └── model_metrics.json         # Metrics
├── data/credit_data.csv           # Dataset
└── requirements.txt

## Quick Start

### 1. Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
source venv/bin/activate        # macOS/Linux

pip install -r requirements.txt


### 2. Run
```bash
python backend/app.py
```

### 3. Open
```
http://localhost:5000

## Model Performance

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|                     |          |           |        |       |         |
| Random Forest       | 84.1%    | 72.2%     | 65.0%  | 68.4% | 75.8%   |
| Logistic Regression | 82.5%    | 65.9%     | 66.1%  | 65.9% | 74.6%   |
| Gradient Boosting   | 83.2%    | 69.3%     | 63.8%  | 66.5% | 73.3%   |
| Decision Tree       | 79.1%    | 58.9%     | 61.4%  | 60.1% | 69.5%   |


## API Endpoints

POST /api/predict — Predict credit score & risk

Request (example):
```json
{
  "age": 35,
  "income": 75000,
  "debt_to_income": 0.3,
  "credit_utilization": 0.25,
  "missed_payments": 0,
  "loan_amount": 20000,
  "loan_grade": "C"
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "credit_score": 728,
    "grade": "AA",
    "tier": "Good",
    "default_risk": 8.5,
    "approved": true
  },
  "recommendations": [...]
}
```

**GET `/api/metrics`** — Model performance metrics

**GET `/api/health`** — API status

---
## Credit Score Scale

| Score   | Tier      | Grade | Risk    |
| 750-850 | Excellent | AAA   | Low     |
| 700-749 | Good      | AA    | Low-Med |
| 650-699 | Fair      | BB    | Medium  |
| 600-649 | Poor      | CC    | Med-High|
| 300-599 | Very Poor | D     | High    |

## Tech Stack

Backend: Flask, Scikit-learn, Pandas, NumPy, Joblib
Frontend:HTML5, CSS3, JavaScript
ML:Random Forest, StandardScaler, StratifiedKFold

## Engineered Features

Base (16): age, income, employment_years, debt_to_income, credit_utilization, missed_payments, num_open_accounts, num_credit_inquiries, total_debt, credit_history_years, savings_balance, monthly_expenses, loan_amount, num_dependents, has_mortgage, has_auto_loan

Engineered (16):
- payment_stress = missed_payments × debt_to_income
- income_per_dependent = income / (dependents + 1)
- savings_ratio = savings_balance / income
- monthly_debt_burden = total_debt / income
- credit_experience_score = credit_history × (1 - utilization)
- loan_to_income = loan_amount / income
- Plus 10 more loan-specific features


## Training Data

Kaggle Credit Risk Dataset — 32K+ records, 16 features, binary classification

Alternatives: German Credit (1K), Home Credit (350K), Credit Card Default (30K)

## Use Cases

| User             | Benefit                                   |
| Banks            | Automate loan approval, reduce risk       |
| Credit Companies | Determine limits, risk pricing            |
| Fintech          | Alternative lending, portfolio management |
| Consumers        | Understand factors, improvement tips      |


## Learning Outcomes

- End-to-end ML pipeline
- Feature engineering & selection
- Model evaluation & comparison
- Handling imbalanced data
- Flask API development
- Frontend-backend integration
- REST API design

## Future Enhancements

- [ ] Model explainability (SHAP)
- [ ] Database integration
- [ ] User authentication
- [ ] Real-time monitoring
- [ ] Auto-retraining pipeline
- [ ] Mobile app


## License

MIT License — Free for educational and commercial use

## Author

Omama qamar — CodeAlpha Intern (April 2026)

GitHub: [omamaqamar](https://github.com/omamaqamar)
LinkedIn:(https://www.linkedin.com/in/omama-qamar/)