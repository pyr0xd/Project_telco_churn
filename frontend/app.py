# frontend/app.py
import streamlit as st
import requests
import sys
import os

# Add parent directory to path to import root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config
from db import save_prediction_log, fetch_recent_logs

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🔮",
    layout="wide"
)

st.title("📊 Telco Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of churn and log the result to Supabase.")

# Create Form Layout with 3 Columns
st.subheader("Customer Demographics & Account Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0, step=1.0)

# Build feature dictionary
input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "TechSupport": tech_support,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

st.markdown("---")

if st.button("🔮 Predict Churn Risk", type="primary"):
    with st.spinner("Processing prediction..."):
        try:
            # 1. Call the FastAPI endpoint
            response = requests.post(config.API_URL, json=input_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "Unknown")
                probability = result.get("probability", 0.0)

                # Display Results
                st.success("Prediction completed successfully!")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if prediction in ["Churn", 1, "1"]:
                        st.error(f"### Result: High Risk of Churn")
                    else:
                        st.success(f"### Result: Low Risk of Churn")
                
                with res_col2:
                    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")

                # 2. Log result to Supabase
                db_success = save_prediction_log(
                    features=input_data,
                    prediction=str(prediction),
                    probability=probability
                )

                if db_success:
                    st.toast("Result logged to Supabase successfully!", icon="✅")
                else:
                    st.warning("Failed to save prediction log to Supabase.")

            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except Exception as e:
            st.error(f"Could not connect to prediction service: {e}")

# Sidebar: View Recent Prediction Logs
st.sidebar.title("📜 Prediction History")
if st.sidebar.button("Refresh Logs"):
    logs = fetch_recent_logs(limit=5)
    if logs:
        for log in logs:
            st.sidebar.markdown(f"**ID {log['id']}** | *{log['created_at'].strftime('%Y-%m-%d %H:%M')}*")
            st.sidebar.write(f"**Prediction:** {log['prediction']} ({log['probability']*100:.1f}%)")
            st.sidebar.json(log['features'], expanded=False)
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("No logs found.")