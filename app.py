import streamlit as st
import joblib
import numpy as np

# 1. Load your saved model
model = joblib.load("xgboost_churn_model.joblib")

st.title("📱 Telecom Customer Churn Predictor")
st.write("Input the top customer account metrics below to evaluate churn risk.")

st.divider()

# 2. Build input fields for your top actual features
st.subheader("Key Predictive Features")

# Feature 1: Contract (Highest importance: 0.369)
# Format options: 0 = Month-to-month, 1 = One year, 2 = Two year (adjust if your encoding differs!)
contract = st.selectbox("Contract Type", options=[0, 1, 2], 
                        format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x])

# Feature 2: Internet Service
internet_service = st.selectbox("Internet Service Type", options=[0, 1, 2], 
                                format_func=lambda x: ["DSL", "Fiber Optic", "No Internet"][x])

# Feature 3: Online Security
online_security = st.selectbox("Online Security Enabled", options=[0, 1, 2], 
                               format_func=lambda x: ["No", "Yes", "No Internet Service"][x])

# Feature 4: Tenure
tenure = st.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=12)

# Feature 5: Monthly Charges
monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=65.0)

st.divider()

if st.button("Run Churn Analysis", type="primary"):
    # CRITICAL: We need to match the 20-feature input shape your model expects.
    # We will pass your top inputs and pad the remaining 15 features with 0 for placeholder values.
    base_features = [contract, internet_service, online_security, tenure, monthly_charges]
    padding = [0] * 15 
    full_input = np.array([base_features + padding])
    
    # Run prediction
    prediction = model.predict(full_input)[0]
    prediction_proba = model.predict_proba(full_input)[0][1]
    
    # Display results
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!** Probability: {prediction_proba:.1%}")
        st.write("Action Required: Prioritize for retention offers.")
    else:
        st.success(f"✅ **Low Churn Risk.** Probability: {prediction_proba:.1%}")
        st.write("Customer account is highly stable.")