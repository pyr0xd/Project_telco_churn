from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Telco Churn API"
    model_path: str = "artifacts/churn_model.pkl"

settings = Settings()