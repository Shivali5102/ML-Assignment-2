# ===============================
# Breast Cancer Classification
# Machine Learning Assignment 2
# ===============================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Load dataset
data = pd.read_csv("Breast_Cancer.csv")

# Remove extra spaces in column names
data.columns = data.columns.str.strip()

# Check columns
print("Columns:", data.columns)

# Drop non-useful text columns if needed
# (Optional but safe: remove 'Race', 'Marital Status', etc. if causing issues)

# Convert categorical columns using one-hot encoding

data = pd.get_dummies(data, drop_first=True)

# Save feature column order
feature_columns = data.columns
joblib.dump(feature_columns, "model/feature_columns.pkl")

# Encode target column
# Assuming Status_Dead exists after get_dummies
if "Status_Dead" in data.columns:
    y = data["Status_Dead"]
    X = data.drop("Status_Dead", axis=1)
else:
    # If Status column still exists
    data["Status"] = data["Status"].map({"Alive":0, "Dead":1})
    y = data["Status"]
    X = data.drop("Status", axis=1)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#joblib.dump(scaler, "model/scaler.pkl")
#joblib.dump(feature_columns, "model/feature_columns.pkl")

# ==========================
# Define Models
# ==========================
from sklearn.pipeline import Pipeline

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}


results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Save full pipeline
    joblib.dump(pipeline, f"model/{name}.pkl")

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, acc, auc, prec, rec, f1, mcc])


results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "AUC", "Precision",
    "Recall", "F1 Score", "MCC"
])

print("\nModel Comparison:\n")
print(results_df)
