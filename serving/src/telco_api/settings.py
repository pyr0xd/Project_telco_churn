from pydantic_settings import BaseSettings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Settings(BaseSettings):
    app_name: str = "Telco Churn API"
    model_path: str = os.path.join(BASE_DIR, "model_store", "model.joblib")

settings = Settings()