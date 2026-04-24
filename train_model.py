# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv("ar_large_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# 3. ENCODE CATEGORICAL FEATURES
# =========================

label_encoders = {}

for col in df.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =========================
# 4. FEATURE SELECTION (NO LEAKAGE)
# =========================
# We intentionally remove leakage columns:
# - delay_days (directly derived from target)
# - payment_date, due_date (future info leak)
# - IDs (not useful for prediction)

X = df[[
    "invoice_amount",
    "industry",
    "region",
    "past_avg_delay",
    "num_invoices"
]]

y = df["late_flag"]

print("\nFeatures used:", X.columns.tolist())

# =========================
# 5. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. HANDLE CLASS IMBALANCE (SMOTE)
# =========================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE distribution:", np.bincount(y_train))

# =========================
# 7. COMPUTE CLASS WEIGHT (IMPORTANT)
# =========================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# =========================
# 8. XGBOOST MODEL
# =========================

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

xgb_model.fit(X_train, y_train)

# =========================
# 9. PREDICTIONS
# =========================

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# =========================
# 10. EVALUATION (IMPORTANT METRICS)
# =========================

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# 11. BUSINESS RISK LOGIC
# =========================

def risk_predict(prob):
    if prob >= 0.5:
        return "High Risk (Late Payment)", "Send reminder immediately"
    elif prob >= 0:
        return "Medium Risk", "Follow up soon"
    else:
        return "Low Risk", "No action needed"

# =========================
# 12. SAMPLE OUTPUT
# =========================

print("\n==============================")
print("📌 SAMPLE CUSTOMER PREDICTIONS")
print("==============================\n")

for i in range(10):
    risk, action = risk_predict(y_prob[i])
    
    print(f"Customer {i+1}")
    print("Probability:", round(y_prob[i], 3))
    print("Risk Level:", risk)
    print("Action:", action)
    print("----------------------")

# =========================
# 13. FEATURE IMPORTANCE (VERY IMPORTANT FOR INTERVIEWS)
# =========================

print("\n📊 Feature Importance Plot:")
plot_importance(xgb_model)
plt.show()