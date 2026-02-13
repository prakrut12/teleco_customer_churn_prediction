import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load Saved Model Files
# ---------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Telco Customer Churn", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# ---------------------------
# USER INPUTS
# ---------------------------
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", 
                         ["Month-to-month", "One year", "Two year"])

internet_service = st.selectbox("Internet Service",
                                 ["DSL", "Fiber optic", "No"])

payment_method = st.selectbox("Payment Method",
                              ["Electronic check", 
                               "Mailed check",
                               "Bank transfer (automatic)",
                               "Credit card (automatic)"])

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("Predict Churn Risk"):

    # Create input dictionary
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Align with training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # ---------------------------
    # RESULTS SECTION
    # ---------------------------
    st.subheader("Prediction Result")

    # Dynamic Risk Category
    if probability > 0.6:
        st.error("🔴 High Risk Customer (Churn Probability > 60%)")
    elif probability > 0.4:
        st.warning("🟡 Medium Risk Customer (Churn Probability 40% – 60%)")
    else:
        st.success("🟢 Low Risk Customer (Churn Probability < 40%)")

    st.write(f"Churn Probability: {probability*100:.2f}%")

    # Progress bar
    st.progress(int(probability * 100))

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("📈 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance", ascending=False
        ).head(10)

        fig, ax = plt.subplots()
        ax.barh(importance_df["Feature"],
                importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Important Features")

        st.pyplot(fig)

    # ---------------------------
    # BUSINESS INSIGHTS
    # ---------------------------
    st.subheader("💡 Business Insights")

    if probability > 0.6:
        st.markdown("""
        **Recommended Actions:**
        - Offer retention discount
        - Provide long-term contract option
        - Assign dedicated customer support
        """)
    elif probability > 0.4:
        st.markdown("""
        **Recommended Actions:**
        - Send loyalty email
        - Offer minor upgrade incentives
        - Monitor engagement
        """)
    else:
        st.markdown("""
        **Recommended Actions:**
        - Upsell premium services
        - Promote referral programs
        - Maintain engagement campaigns
        """)
