import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import plotly.graph_objects as go

# ------------------------------
# Load Model Artifacts
# ------------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(page_title="Telco Customer Churn", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# ------------------------------
# User Inputs
# ------------------------------

tenure = st.number_input("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method",
                       ["Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])

# ------------------------------
# Encode Categorical
# ------------------------------

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract_map[contract],
    "InternetService": internet_map[internet],
    "PaymentMethod": payment_map[payment]
}

input_df = pd.DataFrame([input_dict])

# Align order
input_df = input_df[feature_names]

# Scale
input_scaled = scaler.transform(input_df)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Churn Risk"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk: Customer likely to churn\n\nConfidence: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk: Customer likely to stay\n\nConfidence: {(1-probability):.2%}")

# ------------------------------
# Model Evaluation Section
# ------------------------------

st.markdown("---")
st.subheader("📈 Model Performance Overview")

st.info("""
Model trained using Logistic Regression.
Accuracy ≈ 76%  
AUC ≈ 0.84  
Precision (Churn) ≈ 0.56  
Recall (Churn) ≈ 0.47  
""")

# Dummy ROC curve (for demo display)
fpr = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
tpr = np.array([0.0, 0.5, 0.7, 0.9, 1.0])

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr,
                         mode='lines',
                         name='ROC Curve',
                         line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                         mode='lines',
                         name='Random',
                         line=dict(color='gray')))

fig.update_layout(
    title="ROC Curve (Demo)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)

st.plotly_chart(fig)
