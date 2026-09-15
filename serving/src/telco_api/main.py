from fastapi import FastAPI
from telco_api.settings import settings
from telco_api.registry import registry
from telco_api.routers import predict

app = FastAPI(title=settings.app_name)

@app.on_event("startup")
def startup_event():
    registry.load_model(settings.model_path)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(predict.router)