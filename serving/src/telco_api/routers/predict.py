from fastapi import APIRouter
import pandas as pd
from serving.src.telco_api.registry import registry

router = APIRouter(prefix="/predict", tags=["Inference"])

@router.post("/")
def predict_churn(customer: dict):
    # Check if the model loaded properly in the registry
    if registry.model is None:
        return {"error": "Model not loaded on server."}

    # Your friend's core ML logic:
    input_df = pd.DataFrame([customer])
    
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

    # Using the model from the registry instead of a local variable
    prediction = registry.model.predict(input_df)[0]
    probability = registry.model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability)
    }