from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .dataset_load import get_data_splits
from .export import export_artifacts
from .manifest import model_key

def run_production_training(model_name: str, custom_params: dict = None):
    print(f"Preparing production training for {model_name}...")

    # Hämta split och preprocessor via funktionsmönstret
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = get_data_splits()
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

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

    print(f"Training on combined train+val data ({len(X_full)} rows)...")
    pipeline.fit(X_full, y_full)

    print("Evaluating model on the unseen test set...")
    preds = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\n" + "="*40)
    print(f" FINAL TEST RESULTS FOR: {model_name}")
    print("="*40)
    print(f" Test Accuracy : {acc:.4f}")
    print(f" Test F1-Score : {f1:.4f}")
    print("="*40 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = f"artifacts/production_{timestamp}"

    mkey = model_key(model_name)
    models_fit = {model_name: pipeline}
    metrics_map = {
        mkey: {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "best_params": clean_params,
            "evaluation_set": "test"
        }
    }

    saved_path = export_artifacts(
        models_fit=models_fit,
        metrics_map=metrics_map,
        out_dir=version_dir,
        artifact_version=f"production_{timestamp}",
        notes=f"Production training and test evaluation for {model_name}"
    )
    print(f"Immutable production artifacts and manifest exported to: {saved_path}\n")

if __name__ == "__main__":
    run_production_training("LogisticRegression", custom_params={"C": 10.0})