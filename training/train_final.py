import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .dataset_load import fetch_and_clean_data, get_preprocessor
from .export import export_artifacts
from .manifest import model_key

def run_final_training(model_name: str, custom_params: dict = None):
    print(f"Preparing final 100% full-data training for {model_name}...")

    df = fetch_and_clean_data()
    y = df['Churn']
    X = df.drop(columns=['Churn'], errors='ignore')

    preprocessor = get_preprocessor(X.columns)
    clean_params = {k.replace("model__", ""): v for k, v in (custom_params or {}).items()}
    
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42, **clean_params)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42, **clean_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    print(f"Training on 100% full dataset ({len(X)} rows)...")
    pipeline.fit(X, y)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = f"artifacts/final_{timestamp}"

    mkey = model_key(model_name)
    models_fit = {model_name: pipeline}
    metrics_map = {
        mkey: {
            "training_set": "100%_full_data",
            "rows_trained": len(X),
            "best_params": clean_params
        }
    }

    saved_path = export_artifacts(
        models_fit=models_fit,
        metrics_map=metrics_map,
        out_dir=version_dir,
        artifact_version=f"final_{timestamp}",
        notes=f"Final full-data training for {model_name}"
    )
    print(f"Immutable final artifacts and manifest exported to: {saved_path}")

    api_model_dir = Path("serving/src/telco_api/model_store")
    api_model_dir.mkdir(parents=True, exist_ok=True)
    model_source = saved_path / f"{mkey}_best.joblib"
    api_dest = api_model_dir / "model.joblib"
    
    if model_source.exists():
        shutil.copy(model_source, api_dest)
        print(f"Synced final production model to serving store: {api_dest}\n")

if __name__ == "__main__":
    run_final_training("LogisticRegression", custom_params={"C": 10.0})