import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# -------------------------------
# Load Model Files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# -------------------------------
# User Inputs
# -------------------------------
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Churn Risk"):

    # Create input dictionary
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # -------------------------------
    # Show Results
    # -------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")

    st.write(f"Churn Probability: **{probability:.2%}**")
