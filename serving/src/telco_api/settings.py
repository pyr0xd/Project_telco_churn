from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Telco Churn API"
    model_path: str = "artifacts/baseline/randomforest_best.joblib"
    # Model Changing path example down below
    # model_path: str = "artifacts/model.joblib"

settings = Settings()