# 🏦 CreditIQ — Credit Scoring ML System

A production-ready credit risk prediction application using Random Forest ML to automatically score and assess creditworthiness (300-850 credit score). Built with Flask backend and interactive dark-themed web dashboard.

**Project Type:** CodeAlpha Internship | **Status:** ✅ Complete & Tested  
**Repository:** [GitHub](https://github.com/omamaqamar/CodeAlpha_credit-scoring)

---

## ⚡ Quick Start (3 Steps)

### Prerequisites
- **Python 3.8+** installed ([Download here](https://www.python.org/downloads/) if needed)
- **Windows/Mac/Linux** supported
- ~200MB disk space for dependencies

### Step 1: Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Start Backend
```bash
# Open a terminal/PowerShell and run:
python backend/app.py

# You should see:
# * Running on http://127.0.0.1:5000
# ✓ Model loaded successfully
```

### Step 3: Open Dashboard
Visit **`http://localhost:5000`** in your browser. You'll see the interactive credit scoring dashboard.

---

## 📁 Project Structure

```
credit-scoring/
├── backend/
│   └── app.py                    # Flask REST API (main server)
├── frontend/
│   └── index.html                # Web dashboard (loads in browser)
├── model/
│   ├── best_model.pkl            # Trained Random Forest (32 features)
│   ├── scaler.pkl                # Feature normalization
│   ├── feature_names.json        # Feature order (required)
│   └── model_metrics.json        # Performance metrics
├── data/
│   ├── credit_data_cleaned.csv   # Training dataset
│   └── credit_risk_dataset.csv   # Original Kaggle data
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

---

## 🎯 How It Works

### Architecture
1. **Frontend (`index.html`)** → User enters credit data in browser form
2. **REST API (`backend/app.py`)** → Receives data, runs prediction, returns results
3. **ML Model** → Random Forest processes 32 financial features
4. **Response** → Credit score (300-850), risk tier, loan approval decision

### Key Features
- ✅ **Fast predictions** — Sub-100ms response times
- ✅ **32 engineered features** — Income, debt, payment history, etc.
- ✅ **Real-time metrics** — Accuracy: 84.1%, ROC-AUC: 75.8%
- ✅ **Dark theme UI** — Professional, modern dashboard
- ✅ **REST API** — Integrate with your systems
- ✅ **CORS enabled** — Works with external front-ends

---

## 🚀 API Endpoints

All endpoints are available at `http://localhost:5000`

### **POST /api/predict** — Get Credit Score & Risk
**Request:**
```json
{
  "age": 35,
  "income": 75000,
  "debt_to_income": 0.3,
  "credit_utilization": 0.25,
  "missed_payments": 0,
  "loan_amount": 20000,
  "loan_grade": "C",
  "employment_years": 5,
  "credit_history_years": 12
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "credit_score": 728,
    "grade": "AA",
    "tier": "Good",
    "default_risk": 8.5,
    "approved": true,
    "recommendations": [
      "Reduce credit card utilization below 30%",
      "Maintain perfect payment history"
    ]
  }
}
```

### **GET /api/metrics** — Model Performance
Returns accuracy, precision, recall, F1-score, ROC-AUC for the Random Forest model.

### **GET /api/health** — API Status
Returns `{"status": "healthy"}` if backend is running.

---

## 🔧 Troubleshooting

### ❌ "Address already in use" error
**Problem:** Port 5000 is occupied by another app.

**Solution:**
```bash
# Option 1: Kill the process using port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Option 2: Use different port
# Edit backend/app.py, change:
# app.run(debug=True, port=5000)
# to:
# app.run(debug=True, port=5001)
```

### ❌ "ModuleNotFoundError" or "No module named 'flask'"
**Problem:** Virtual environment not activated.

**Solution:**
```bash
# Reactivate and reinstall:
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

### ❌ "Model not loaded" error
**Problem:** Missing trained model files.

**Solution:** Ensure these files exist in `model/` folder:
- ✓ `best_model.pkl`
- ✓ `scaler.pkl`
- ✓ `feature_names.json`
- ✓ `model_metrics.json`

If missing, contact the developer or retrain the model using `model/train_kaggle.py`.

### ❌ Browser shows "Connection refused"
**Problem:** Backend is not running.

**Solution:** Open a terminal and run:
```bash
python backend/app.py
```
Then visit `http://localhost:5000` again.

---

## 📊 Model Performance

Tested on 32,000+ records from Kaggle Credit Risk Dataset

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:-------------------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Random Forest**   | **84.1%**| **72.2%** | **65.0%** | **68.4%**| **75.8%** |
| Logistic Regression | 82.5%   | 65.9%    | 66.1% | 65.9%   | 74.6%   |
| Gradient Boosting   | 83.2%   | 69.3%    | 63.8% | 66.5%   | 73.3%   |
| Decision Tree       | 79.1%   | 58.9%    | 61.4% | 60.1%   | 69.5%   |

---

## 📈 Credit Score Scale

| Score Range | Tier | Grade | Risk Level | Notes |
|:-----------|:-----|:-----:|:----------:|:------|
| 750–850 | Excellent | AAA | Very Low | Lowest rates, immediate approval |
| 700–749 | Good | AA | Low | Approved, standard rates |
| 650–699 | Fair | BB | Medium | May require additional checks |
| 600–649 | Poor | CC | Med-High | Limited options, higher rates |
| 300–599 | Very Poor | D | High | Likely declined or subprime rates |

---

## 🔌 Tech Stack

| Component | Technology |
|:----------|:-----------|
| **Backend** | Flask 3.0, Python 3.8+ |
| **ML** | Scikit-learn, Random Forest, StandardScaler |
| **Data** | Pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Gunicorn (production) |

---

## 📝 Features Used

**Base Features (16):**
- Demographics: age, employment_years, num_dependents
- Income & Expenses: income, monthly_expenses, savings_balance
- Debt Metrics: debt_to_income, total_debt, credit_utilization
- Credit History: credit_history_years, num_credit_inquiries, missed_payments
- Accounts: num_open_accounts, has_mortgage, has_auto_loan

**Engineered Features (16):** 
Derived from base features (payment_stress, income_per_dependent, savings_ratio, etc.) for better predictive power.

**Total: 32 features → High-accuracy predictions**

---

## 🚢 Deployment Options

### Option 1: Local/Internal Server (Recommended for Testing)
```bash
python backend/app.py
# Access at http://localhost:5000
```

### Option 2: Production Server (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
# Access at http://your-server-ip:5000
```

### Option 3: Docker Container
```bash
# Build image
docker build -t creditiq .

# Run container
docker run -p 5000:5000 creditiq

# Access at http://localhost:5000
```
*(Include Dockerfile in repository)*

### Option 4: Cloud Platforms (AWS, Heroku, Azure, Railway)
Auto-deploys with CI/CD pipelines. Contact developer for setup instructions.

---

## ❓ FAQ

**Q: Do I need Python installed?**  
A: Yes. Download from [python.org](https://www.python.org/downloads/). Verify with `python --version`.

**Q: How do I know the backend is running?**  
A: Terminal shows `* Running on http://127.0.0.1:5000` and model loads successfully.

**Q: Can I use this without the UI?**  
A: Yes! Use the REST API directly:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 75000, ...}'
```

**Q: How accurate are the predictions?**  
A: 84.1% accuracy on test data. See [Model Performance](#-model-performance) section.

**Q: Can I train a new model?**  
A: Yes. Run `python model/train_kaggle.py` after updating data files.

**Q: Is this production-ready?**  
A: Yes, but add authentication, logging, and monitoring for enterprise use.

---

## 👨‍💻 Developer Info

- **Author:** Omama Qamar — CodeAlpha Intern
- **Created:** April 2026
- **License:** MIT (Free for educational & commercial use)
- **Status:** ✅ Tested & ready to deploy

---

## 📞 Support

For issues, questions, or deployment help:
- Check [Troubleshooting](#-troubleshooting) section above
- Review code comments in `backend/app.py`
- Contact: omamaqamar [GitHub](https://github.com/omamaqamar)

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
