import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import os

from .dataset_load import X_train, y_train, X_val, y_val, preprocessor

def run_training(selected_models=None):
    mlflow.set_experiment("basemodel_test")

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
        
    os.makedirs("artifacts/baseline", exist_ok=True)

    for name, config in models_to_tune.items():
        with mlflow.start_run(run_name=name):
            
            mlflow.log_param("model_type", name)
            
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
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
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"[{name}] Best params: {best_params}")
            

            for param_name, param_value in best_params.items():
                clean_name = param_name.replace("model__", "")
                mlflow.log_param(clean_name, param_value)

            pred = best_model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            f1 = f1_score(y_val, pred)
            
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1_score", f1)
            

            model_path = f"artifacts/baseline/{name.lower()}_best.joblib"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)
            
            print(f"Model: [{name}] | Val F1: {f1:.4f} | Val Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    run_training()