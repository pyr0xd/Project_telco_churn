from pathlib import Path
from fastapi import FastAPI
from serving.src.telco_api.settings import settings
from serving.src.telco_api.registry import registry
from serving.src.telco_api.routers import predict

app = FastAPI(title=settings.app_name)

@app.on_event("startup")
def startup_event():
    # Resolve the root project directory (4 levels up from this file)
    project_root = Path(__file__).resolve().parents[3]
    full_model_path = project_root / settings.model_path
    
    print(f"Loading model from: {full_model_path}")
    registry.load_model(full_model_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(predict.router)