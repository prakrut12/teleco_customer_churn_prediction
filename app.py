import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

# -----------------------------
# Load Model Artifacts
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Telco Customer Churn", layout="centered")

st.title("📊 Telco Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn.")

# -----------------------------
# User Inputs
# -----------------------------
tenure = st.number_input("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method,
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)

# Ensure all training features exist
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn Risk"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("⚠ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.info(f"Confidence: {round(probability * 100, 2)}%")

# =========================================================
# MODEL EVALUATION SECTION (Recreated from Dataset)
# =========================================================

st.markdown("---")
st.subheader("📈 Model Performance Analysis")

if st.button("Show Model Evaluation"):

    # Load dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Ensure same feature order
    X = X[feature_names]

    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {round(auc_score, 2)})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )

    st.plotly_chart(fig_roc)

    # ---------------- Precision Recall ----------------
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap_score = average_precision_score(y, y_prob)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
    fig_pr.update_layout(
        title=f"Precision-Recall Curve (AP = {round(ap_score, 2)})",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )

    st.plotly_chart(fig_pr)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y, y_pred)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Stay", "Predicted Churn"],
            y=["Actual Stay", "Actual Churn"],
        )
    )

    fig_cm.update_layout(title="Confusion Matrix")

    st.plotly_chart(fig_cm)

    # ---------------- Business Explanation ----------------
    st.markdown("### 📌 Business Interpretation")

    st.write(f"""
    - **AUC Score:** {round(auc_score,2)} → Model separates churners vs non-churners well.
    - **Precision:** Measures how many predicted churners were correct.
    - **Recall:** Measures how many actual churners were detected.
    - **Confusion Matrix:** Shows classification distribution.
    """)

