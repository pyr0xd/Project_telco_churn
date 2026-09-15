import pandas as pd
import joblib

full_pipeline = joblib.load('artifacts/model.joblib')

def predict_churn(customer: dict) -> dict:
    input_df = pd.DataFrame([customer])
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

    prediction = full_pipeline.predict(input_df)[0]
    probability = full_pipeline.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability)
    }