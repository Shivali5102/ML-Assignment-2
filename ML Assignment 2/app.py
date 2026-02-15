# ==========================================
# Streamlit App - ML Assignment 2
# Breast Cancer Survival Prediction
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Breast Cancer Survival Prediction")

st.title("Breast Cancer Survival Prediction App")
st.write("Upload a CSV file and select a model to predict patient survival status.")

# ===============================
# Upload Dataset
# ===============================

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

# ===============================
# Model Selection
# ===============================

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

# ===============================
# Prediction Block
# ===============================

if uploaded_file is not None:

    # Load dataset
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    # Save original copy for evaluation
    original_data = data.copy()

    # Remove target column if present
    if "Status" in data.columns:
        data = data.drop(columns=["Status"])

    # One-hot encode features
    data = pd.get_dummies(data, drop_first=True)

    # Load trained pipeline model
    model = joblib.load(f"model/{model_choice}.pkl")

    # Make predictions
    predictions = model.predict(data)

    # Convert predictions to readable labels
    prediction_labels = ["Dead" if p == 1 else "Alive" for p in predictions]

    st.subheader("Predictions")
    st.write(pd.DataFrame({"Prediction": prediction_labels}))

    # ===============================
    # Evaluation Metrics (If Status Exists)
    # ===============================

    if "Status" in original_data.columns:

        st.subheader("Model Evaluation")

        # Encode true labels
        true_labels = original_data["Status"].map({"Alive": 0, "Dead": 1})

        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(true_labels, predictions))

        # Confusion Matrix
        st.text("Confusion Matrix:")
        cm = confusion_matrix(true_labels, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

st.write("--------------------------------------------------")
st.write("Developed for ML Assignment 2 - BITS WILP")
