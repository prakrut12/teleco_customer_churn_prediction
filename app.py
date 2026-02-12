import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load model artifacts
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
feature_names = pickle.load(open("models/features.pkl", "rb"))

st.set_page_config(page_title="Telco Customer Churn Prediction")

st.title("📊 Telco Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn.")

# ---------------- INPUTS ----------------

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ---------------- PREDICTION ----------------

if st.button("Predict Churn Risk"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    # Match training features
    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.info(f"Confidence: {np.max(probability) * 100:.2f}%")

# ---------------- MODEL INSIGHTS ----------------

st.markdown("---")
st.subheader("📈 Model Insights")

# Feature Importance
if st.button("Show Feature Importance"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

# Confusion Matrix
if st.button("Show Confusion Matrix"):

    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
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

    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
