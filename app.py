import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# ---------------------------------
# Load Files
# ---------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# ---------------------------------
# User Inputs
# ---------------------------------
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

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Churn Risk"):

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
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------------------------
    # Result
    # ---------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is not likely to churn.")

    st.write(f"### Churn Probability: {probability:.2%}")

    # ---------------------------------
    # Risk Meter
    # ---------------------------------
    st.progress(float(probability))

    # ---------------------------------
    # Feature Importance Graph
    # ---------------------------------
    st.subheader("📈 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")

        st.pyplot(fig)

    # ---------------------------------
    # Business Insights Section
    # ---------------------------------
    st.subheader("💡 Business Insights")

    if probability > 0.7:
        st.warning("""
        🔴 High Risk Customer  
        Recommended Actions:
        - Offer retention discount
        - Upgrade to long-term contract
        - Personalized engagement call
        """)
    elif probability > 0.4:
        st.info("""
        🟡 Medium Risk Customer  
        Recommended Actions:
        - Offer bundled services
        - Loyalty rewards
        - Monitor usage pattern
        """)
    else:
        st.success("""
        🟢 Low Risk Customer  
        Recommended Actions:
        - Upsell premium services
        - Referral programs
        """)
