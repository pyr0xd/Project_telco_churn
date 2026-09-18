import os
import glob
from datetime import datetime
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .dataset_load import get_data_splits
from .export import export_artifacts
from .manifest import model_key


def get_cluster_distances(X_raw):
    """Hämtar klusteravstånd om klustring finns, annars returnerar None."""
    cluster_dirs = sorted(glob.glob("artifacts/cluster_*"), reverse=True)
    if not cluster_dirs:
        return None
    cluster_files = glob.glob(os.path.join(cluster_dirs[0], "*.joblib"))
    if not cluster_files:
        return None
    try:
        cluster_pipe = joblib.load(cluster_files[0])
        if 'preprocessing' in cluster_pipe.named_steps and 'model' in cluster_pipe.named_steps:
            X_prep_c = cluster_pipe.named_steps['preprocessing'].transform(X_raw)
            return cluster_pipe.named_steps['model'].transform(X_prep_c)
    except Exception:
        return None
    return None


def run_training(selected_models=None):
    mlflow.set_experiment("basemodel_test")

    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = get_data_splits()

    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)

    train_cluster_dist = get_cluster_distances(X_train)
    val_cluster_dist = get_cluster_distances(X_val)

    if train_cluster_dist is not None and val_cluster_dist is not None:
        print("Kluster-features hittade och integreras i träningsdatan.")
        X_train_final = np.hstack([X_train_prep, train_cluster_dist])
        X_val_final = np.hstack([X_val_prep, val_cluster_dist])
    else:
        print("Inga kluster-features hittade, kör enbart på standardpreprocessor.")
        X_train_final = X_train_prep
        X_val_final = X_val_prep

    models_to_tune = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                'model__C': [0.01, 0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 5, 10],
                'model__min_samples_split': [2, 5]
            }
        }
    }
    
    if selected_models:
        models_to_tune = {name: config for name, config in models_to_tune.items() if name in selected_models}
        
    models_fit = {}
    metrics_map = {}

    for name, config in models_to_tune.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            
            pipeline = Pipeline([
                ('model', config['model'])
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["params"],
                cv=5,
                scoring='f1',
                n_jobs=-1 
            )
            
            print(f"Training and tuning {name}...")
            grid_search.fit(X_train_final, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"[{name}] Best params: {best_params}")
            
            clean_params = {}
            for param_name, param_value in best_params.items():
                clean_name = param_name.replace("model__", "")
                mlflow.log_param(clean_name, param_value)
                clean_params[clean_name] = param_value

            pred = best_model.predict(X_val_final)
            acc = accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred)
            
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1_score", f1)
            
            os.makedirs("artifacts/_temp_mlflow", exist_ok=True)
            temp_path = f"artifacts/_temp_mlflow/{name.lower()}_best.joblib"
            joblib.dump(best_model, temp_path)
            mlflow.log_artifact(temp_path)
            
            models_fit[name] = best_model
            metrics_map[model_key(name)] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "best_params": clean_params
            }
            
            print(f"Model: [{name}] | Val F1: {f1:.4f} | Val Accuracy: {acc:.4f}\n")
        
    if models_fit:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = f"artifacts/baseline_{timestamp}"
        
        saved_path = export_artifacts(
            models_fit=models_fit,
            metrics_map=metrics_map,
            out_dir=version_dir,
            artifact_version=f"baseline_{timestamp}",
            notes="GridSearch baseline training with cluster feature enrichment",
        )
        print(f"models and manifest exported to: {saved_path}")

if __name__ == "__main__":
    run_training()
    


