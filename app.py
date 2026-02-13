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
    average_precision_score
)

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

contract = st.selectbox("Contract Type",
                        ["Month-to-month", "One year", "Two year"])

internet_service = st.selectbox("Internet Service",
                                ["DSL", "Fiber optic", "No"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Decision Threshold
threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)

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

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if probability >= threshold else 0

    if prediction == 1:
        st.error("⚠ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.info(f"Churn Probability: {probability*100:.2f}%")

# ----------------------------
# MODEL PERFORMANCE SECTION
# ----------------------------
st.markdown("---")
st.subheader("📈 Model Performance Analysis")

if st.button("Show Model Evaluation"):

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
    y_pred = (y_prob >= threshold).astype(int)

    # -------- ROC Curve --------
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                 mode='lines',
                                 name='Random',
                                 line=dict(dash='dash')))

    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {auc_score:.2f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig_roc)

    # -------- Precision-Recall Curve --------
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap_score = average_precision_score(y, y_prob)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision,
                                mode='lines',
                                name='Precision-Recall'))

    fig_pr.update_layout(
        title=f"Precision-Recall Curve (AP = {ap_score:.2f})",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    st.plotly_chart(fig_pr)

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y, y_pred)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted No", "Predicted Yes"],
        y=["Actual No", "Actual Yes"],
        colorscale="Blues"
    ))

    fig_cm.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig_cm)

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------
st.markdown("---")
st.subheader("📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(10))
else:
    st.write("Feature importance not available for this model.")

# ----------------------------
# BUSINESS INSIGHTS
# ----------------------------
st.markdown("---")
st.subheader("💼 Business Insights")

st.write("""
- Customers with month-to-month contracts show higher churn risk.
- Higher monthly charges increase churn probability.
- Low tenure customers are more likely to churn.
- Fiber optic customers tend to churn more than DSL users.
- Adjusting decision threshold allows business to trade-off between recall and precision.
""")
