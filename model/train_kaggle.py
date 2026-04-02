"""Train credit scoring model on Kaggle dataset."""

import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED       = 42
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'model')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
DATASET_ID = 'laotse/credit-risk-dataset'
CSV_NAME   = 'credit_risk_dataset.csv'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
np.random.seed(SEED)


def download_dataset():
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    if os.path.exists(csv_path):
        print(f" Dataset already present at {csv_path}")
        return csv_path

    print(" Downloading dataset from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            DATASET_ID, path=DATA_DIR, unzip=True, quiet=False
        )
        print(" Download complete.")
    except Exception as e:
        print(f"\n Kaggle download failed: {e}")
        print("""
To fix this:
  1. pip install kaggle
  2. Go to https://www.kaggle.com/settings → API → Create New Token
  3. Place kaggle.json at ~/.kaggle/kaggle.json  (Linux/Mac)
                        or C:\\Users\\<You>\\.kaggle\\kaggle.json (Windows)
  4. Run again.

Alternatively, download manually from:
  https://www.kaggle.com/datasets/laotse/credit-risk-dataset
and place 'credit_risk_dataset.csv' in the data/ folder.
        """)
        sys.exit(1)

    if not os.path.exists(csv_path):
        candidates = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if candidates:
            os.rename(os.path.join(DATA_DIR, candidates[0]), csv_path)
        else:
            print(" CSV not found after download. Check the data/ folder.")
            sys.exit(1)

    return csv_path


def load_and_clean(csv_path: str) -> pd.DataFrame:
    print(" Loading and cleaning data...")
    df = pd.read_csv(csv_path)

    print(f"   Raw shape      : {df.shape}")
    print(f"   Columns        : {list(df.columns)}")
    print(f"   Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    rename_map = {
        'person_age'              : 'age',
        'person_income'           : 'income',
        'person_emp_length'       : 'employment_years',
        'loan_amnt'               : 'loan_amount',
        'loan_int_rate'           : 'loan_int_rate',
        'loan_status'             : 'default',
        'loan_percent_income'     : 'debt_to_income',
        'cb_person_cred_hist_length': 'credit_history_years',
        'cb_person_default_on_file' : 'prev_default',
        'loan_intent'             : 'loan_intent',
        'loan_grade'              : 'loan_grade',
        'person_home_ownership'   : 'home_ownership',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = df.drop_duplicates()
    df = df[df['age'].between(18, 100)]
    df = df[df['income'] < 6_000_000]
    df = df[df['employment_years'].notna()]

    # Fill remaining NaNs
    df['employment_years'] = df['employment_years'].fillna(0)
    df['loan_int_rate']    = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df = df.dropna(subset=['default'])

    if 'prev_default' in df.columns:
        df['prev_default'] = (df['prev_default'] == 'Y').astype(int)

    if 'home_ownership' in df.columns:
        df['has_mortgage'] = (df['home_ownership'].str.upper() == 'MORTGAGE').astype(int)
        df['has_rent']     = (df['home_ownership'].str.upper() == 'RENT').astype(int)
    else:
        df['has_mortgage'] = 0
        df['has_rent']     = 0

    # Loan intent one-hot (keep top 5 most frequent)
    if 'loan_intent' in df.columns:
        top_intents = df['loan_intent'].value_counts().nlargest(5).index
        for intent in top_intents:
            col = 'intent_' + intent.lower()
            df[col] = (df['loan_intent'] == intent).astype(int)

    # Loan grade ordinal encoding  A=6 … G=0
    if 'loan_grade' in df.columns:
        grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
        df['loan_grade_num'] = df['loan_grade'].map(grade_map).fillna(3)

    # Derived columns that map to what app.py expects
    df['credit_utilization'] = (df['debt_to_income'] * 0.6).clip(0, 1)
    df['total_debt']         = df['income'] * df['debt_to_income']
    df['missed_payments']    = df.get('prev_default', pd.Series(0, index=df.index)) * 3
    df['num_open_accounts']  = np.random.poisson(4, len(df))   # not in dataset, approximate
    df['num_credit_inquiries'] = np.random.poisson(1.5, len(df))
    df['savings_balance']    = (df['income'] * np.random.beta(1.5, 5, len(df)) * 0.5).round(2)
    df['monthly_expenses']   = (df['income'] / 12 * np.random.uniform(0.35, 0.85, len(df))).round(2)
    df['num_dependents']     = np.random.choice([0,1,2,3,4], len(df), p=[0.35,0.30,0.20,0.10,0.05])
    df['has_auto_loan']      = np.random.binomial(1, 0.35, len(df))

    df['default'] = df['default'].astype(int)

    print(f"   Clean shape    : {df.shape}")
    print(f"   Default rate   : {df['default'].mean():.1%}  ({df['default'].sum()} defaults)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Engineering features...")
    df = df.copy()

    df['payment_stress']          = df['missed_payments'] * df['debt_to_income']
    df['income_per_dependent']    = df['income'] / (df['num_dependents'] + 1)
    df['savings_ratio']           = df['savings_balance'] / (df['income'] + 1)
    df['monthly_debt_burden']     = (df['total_debt'] / 12) / (df['income'] / 12 + 1)
    df['credit_experience_score'] = df['credit_history_years'] * (1 - df['credit_utilization'])
    df['loan_to_income']          = df['loan_amount'] / (df['income'] + 1)
    df['int_rate_norm']           = df.get('loan_int_rate', pd.Series(10, index=df.index)) / 30
    df['grade_util_interaction']  = df.get('loan_grade_num', pd.Series(3, index=df.index)) * df['credit_utilization']

    print(f"   Total features : {df.shape[1]}")
    return df


CORE_FEATURES = [
    'age', 'income', 'employment_years', 'debt_to_income', 'credit_utilization',
    'missed_payments', 'num_open_accounts', 'num_credit_inquiries', 'total_debt',
    'credit_history_years', 'savings_balance', 'monthly_expenses', 'loan_amount',
    'num_dependents', 'has_mortgage', 'has_auto_loan',
    # Engineered
    'payment_stress', 'income_per_dependent', 'savings_ratio',
    'monthly_debt_burden', 'credit_experience_score', 'loan_to_income',
    'int_rate_norm', 'grade_util_interaction', 'prev_default', 'has_rent', 'loan_grade_num',
]

def select_features(df: pd.DataFrame) -> list[str]:
    available = [c for c in CORE_FEATURES if c in df.columns]
    intent_cols = [c for c in df.columns if c.startswith('intent_')]
    return available + intent_cols


def balance_data(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=SEED, k_neighbors=5)
        X_bal, y_bal = sm.fit_resample(X_train, y_train)
        counts = np.bincount(y_bal)
        print(f"   SMOTE → No-default: {counts[0]:,}  Default: {counts[1]:,}")
        return X_bal, y_bal, False          # False = don't use class_weight
    except ImportError:
        print("    imbalanced-learn not installed. Using class_weight='balanced'.")
        return X_train, y_train, True       # True = use class_weight


def train_models(X_train, X_test, y_train, y_test, use_class_weight: bool) -> tuple[dict, str]:
    print(" Training models...")

    cw = 'balanced' if use_class_weight else None

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED, C=0.5, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=20, random_state=SEED, class_weight=cw),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            random_state=SEED, class_weight=cw, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for name, clf in models.items():
        print(f"   Training {name}…", end='', flush=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        cv_auc = cross_val_score(clf, X_train, y_train,
                                  cv=cv, scoring='roc_auc', n_jobs=-1).mean()

        results[name] = {
            "model"           : clf,
            "accuracy"        : accuracy_score(y_test, y_pred),
            "precision"       : precision_score(y_test, y_pred, zero_division=0),
            "recall"          : recall_score(y_test, y_pred, zero_division=0),
            "f1"              : f1_score(y_test, y_pred, zero_division=0),
            "roc_auc"         : roc_auc_score(y_test, y_prob),
            "cv_roc_auc"      : cv_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        r = results[name]
        print(f"  ROC-AUC {r['roc_auc']:.4f}  |  F1 {r['f1']:.4f}  |  Recall {r['recall']:.4f}")

    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n Best: {best_name}  (ROC-AUC: {results[best_name]['roc_auc']:.4f})")
    return results, best_name


def save_artifacts(results, best_name, scaler, feature_names):
    print(" Saving artifacts…")
    joblib.dump(results[best_name]['model'], os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(scaler,                      os.path.join(MODEL_DIR, 'scaler.pkl'))

    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    all_metrics = {}
    for name, r in results.items():
        all_metrics[name] = {
            "accuracy"        : round(r['accuracy'],   4),
            "precision"       : round(r['precision'],  4),
            "recall"          : round(r['recall'],     4),
            "f1"              : round(r['f1'],         4),
            "roc_auc"         : round(r['roc_auc'],    4),
            "cv_roc_auc"      : round(r['cv_roc_auc'], 4),
            "confusion_matrix": r['confusion_matrix'],
            "is_best"         : name == best_name,
        }

    with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
        json.dump({
            "best_model"   : best_name,
            "models"       : all_metrics,
            "trained_at"   : pd.Timestamp.now().isoformat(),
            "n_features"   : len(feature_names),
            "feature_names": feature_names,
            "dataset"      : "Kaggle Credit Risk Dataset (laotse/credit-risk-dataset)",
        }, f, indent=2)
    print(f"   Saved to {MODEL_DIR}/")


# ── 8. Evaluation Plots ───────────────────────────────────────────────────────
def generate_plots(results, best_name, X_test, y_test, feature_names):
    print(" Generating plots…")
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CreditIQ — Model Evaluation (Kaggle Dataset)', fontsize=16, fontweight='bold')

    # ROC curves
    ax = axes[0, 0]
    for name, r in results.items():
        y_prob = r['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2.5 if name == best_name else 1,
                label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0,1],[0,1], 'k--', alpha=0.4)
    ax.set_title('ROC Curves'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Metrics bar chart
    ax = axes[0, 1]
    metric_names = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    x, w = np.arange(len(metric_names)), 0.2
    for i, (name, r) in enumerate(results.items()):
        vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
        ax.bar(x + i*w, vals, w, label=name, alpha=0.85)
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(metric_names, rotation=15)
    ax.set_ylim(0, 1.15); ax.set_title('Metrics Comparison')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Confusion matrix
    ax = axes[1, 0]
    cm = np.array(results[best_name]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Default','Default'],
                yticklabels=['No Default','Default'])
    ax.set_title(f'Confusion Matrix ({best_name})')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

    # Feature importance
    ax = axes[1, 1]
    bm = results[best_name]['model']
    if hasattr(bm, 'feature_importances_'):
        fi = pd.DataFrame({'feature': feature_names, 'importance': bm.feature_importances_})
        fi = fi.sort_values('importance').tail(12)
        fi['feature'] = fi['feature'].str.replace('_', ' ').str.title()
        ax.barh(fi['feature'], fi['importance'], color='steelblue', alpha=0.85)
        ax.set_title('Feature Importances'); ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_title('Feature Importances')

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, 'evaluation_plots.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Plot saved → {out}")


# Main 
if __name__ == '__main__':
    print("=" * 60)
    print("  CREDITIQ — KAGGLE DATASET TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Download
    csv_path = download_dataset()

    # Step 2: Load & clean
    df = load_and_clean(csv_path)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Select features
    FEATURE_COLS = select_features(df)
    print(f"\n Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # Save cleaned dataset
    df.to_csv(os.path.join(DATA_DIR, 'credit_data_cleaned.csv'), index=False)

    # Step 5: Split
    X = df[FEATURE_COLS].values
    y = df['default'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"   Class distribution — Train: {np.bincount(y_train)}   Test: {np.bincount(y_test)}")

    # Step 6: Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Step 7: Balance
    X_train_bal, y_train_bal, use_cw = balance_data(X_train_sc, y_train)

    # Step 8: Train
    results, best_name = train_models(X_train_bal, X_test_sc, y_train_bal, y_test, use_cw)

    # Step 9: Save
    save_artifacts(results, best_name, scaler, FEATURE_COLS)

    # Step 10: Plots
    generate_plots(results, best_name, X_test_sc, y_test, FEATURE_COLS)

    # Step 11: Print classification report
    print("\n Classification Report (Best Model):")
    bm = results[best_name]['model']
    y_pred_final = bm.predict(X_test_sc)
    print(classification_report(y_test, y_pred_final, target_names=['No Default','Default']))

    print("\n Training complete!")
    print(f"   Dataset        : Kaggle Credit Risk Dataset ({len(df):,} records)")
    print(f"   Best Model     : {best_name}")
    print(f"   ROC-AUC        : {results[best_name]['roc_auc']:.4f}")
    print(f"   F1-Score       : {results[best_name]['f1']:.4f}")
    print(f"   Recall         : {results[best_name]['recall']:.4f}")
    print(f"   Artifacts at   : {MODEL_DIR}/")
    print("\n Now start the API: python backend/app.py")