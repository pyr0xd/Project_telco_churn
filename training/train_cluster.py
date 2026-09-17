import os
from datetime import datetime
import joblib
import mlflow

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline

from .dataset_load import X_train, preprocessor
from .export import export_artifacts
from .manifest import model_key

def run_clustering(selected_models=None):
    mlflow.end_run()
    mlflow.set_experiment("payment_clustering_k4")

    # Models constrained to k=4 clusters
    models_to_tune = {
        "KMeans": {
            "class": KMeans,
            "params": [{"n_clusters": 4, "random_state": 42, "n_init": 10}]
        },
        "Agglomerative": {
            "class": AgglomerativeClustering,
            "params": [{"n_clusters": 4, "linkage": link} for link in ["ward", "complete", "average"]]
        }
    }

    if selected_models:
        models_to_tune = {name: config for name, config in models_to_tune.items() if name in selected_models}

    models_fit = {}
    metrics_map = {}

    X_preprocessed = preprocessor.fit_transform(X_train)

    for name, config in models_to_tune.items():
        with mlflow.start_run(run_name=f"{name}_Parent"):
            mlflow.log_param("model_type", name)
            
            best_score = -1.0
            best_model = None
            best_params = {}
            best_calinski = 0.0

            for param_dict in config["params"]:
                run_name = f"{name}_k4_{param_dict.get('linkage', 'default')}"

                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_param("model_type", name)
                    for p_k, p_v in param_dict.items():
                        mlflow.log_param(p_k, p_v)

                    model = config["class"](**param_dict)
                    labels = model.fit_predict(X_preprocessed)

                    sil = silhouette_score(X_preprocessed, labels)
                    cal = calinski_harabasz_score(X_preprocessed, labels)

                    mlflow.log_metric("silhouette_score", sil)
                    mlflow.log_metric("calinski_harabasz_score", cal)

                    print(f"[{run_name}] Silhouette: {sil:.4f} | Calinski-Harabasz: {cal:.2f}")

                    if sil > best_score:
                        best_score = sil
                        best_model = model
                        best_params = param_dict
                        best_calinski = cal

            # Assemble pipeline with fitted preprocessor and best cluster model
            best_pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', best_model)
            ])

            # Save temporary artifact for MLflow run tracking
            os.makedirs("artifacts/_temp_mlflow", exist_ok=True)
            temp_path = f"artifacts/_temp_mlflow/{name.lower()}_best.joblib"
            joblib.dump(best_pipeline, temp_path)
            mlflow.log_artifact(temp_path)

            # Store pipeline and metrics using project manifest keys
            models_fit[name] = best_pipeline
            metrics_map[model_key(name)] = {
                "silhouette_score": float(best_score),
                "calinski_harabasz_score": float(best_calinski),
                "best_params": best_params
            }

            mlflow.log_metric("best_silhouette_score", best_score)
            print(f"Model: [{name}] | Best Silhouette: {best_score:.4f} | Calinski-Harabasz: {best_calinski:.2f}\n")

    # Export fitted models, evaluation payload, and manifest.json
    if models_fit:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = f"artifacts/cluster_{timestamp}"

        saved_path = export_artifacts(
            models_fit=models_fit,
            metrics_map=metrics_map,
            out_dir=version_dir,
            artifact_version=f"cluster_{timestamp}",
            notes="K-Means and Agglomerative k=4 payment method clustering models",
        )
        print(f"Models and manifest exported to: {saved_path}")

if __name__ == "__main__":
    run_clustering()