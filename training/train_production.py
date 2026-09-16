import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .dataset_load import X_train, y_train, X_val, y_val, X_test, y_test, preprocessor

def run_production_training(model_name: str, custom_params: dict = None):
    
    print(f"Preparing production training for {model_name}...")

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

    os.makedirs("artifacts/production", exist_ok=True)
    model_path = f"artifacts/production/{model_name.lower()}_production.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Production model successfully saved to {model_path}\n")

if __name__ == "__main__":
    run_production_training("LogisticRegression", custom_params={"C": 10.0})