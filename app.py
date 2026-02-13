import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ----------------------------
# Load model artifacts
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Telco Customer Churn", layout="centered")

st.title("📊 Telco Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn.")

# ----------------------------
# INPUTS
# ----------------------------
tenure = st.number_input("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Predict Churn Risk"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("⚠ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.info(f"Confidence: {probability*100:.2f}%")

# ----------------------------
# MODEL EVALUATION SECTION
# ----------------------------
st.markdown("---")
st.subheader("📈 Model Performance Analysis")

if st.button("Show ROC Curve & Confusion Matrix"):

    # Load dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    selected_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod"
    ]

    X = df[selected_features]
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    for col in feature_names:
        if col not in X:
            X[col] = 0

    X = X[feature_names]
    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # ---------------- ROC CURVE ----------------
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                 mode='lines',
                                 name='ROC Curve'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                 mode='lines',
                                 name='Random',
                                 line=dict(dash='dash')))

    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {auc_score:.2f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig_roc)

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y, y_pred)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No", "Predicted Yes"],
        y=["Actual No", "Actual Yes"],
        colorscale="Blues"
    ))

    fig_cm.update_layout(title="Confusion Matrix")

    st.plotly_chart(fig_cm)

