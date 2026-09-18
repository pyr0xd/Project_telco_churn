# config.py
import os
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

class Config:
    # Supabase / PostgreSQL Connection Details
    # Example format: postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:your_password@localhost:5432/postgres")
    
    # Alternatively, direct Supabase REST API credentials if using supabase-py
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # FastAPI Model Serving Endpoint
    API_URL: str = os.getenv("API_URL", "http://localhost:8000/predict")
    
    # App Settings
    APP_NAME: str = "Telco Customer Churn Predictor"

config = Config()