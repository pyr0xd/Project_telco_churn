import os
import joblib
import mlflow
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline

from dataset_load import X_train, preprocessor

def run_clustering(selected_models=None):
    mlflow.end_run()
    mlflow.set_experiment("payment_clustering_k4")

    # Fixed strictly to k=4 clusters
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

    os.makedirs("artifacts/clustering", exist_ok=True)
    X_preprocessed = preprocessor.fit_transform(X_train)

    for name, config in models_to_tune.items():
        with mlflow.start_run(run_name=f"{name}_Parent"):
            best_score = -1.0
            best_model = None
            best_params = {}

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

            best_pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', best_model)
            ])

            model_path = f"artifacts/clustering/{name.lower()}_k4_best.joblib"
            joblib.dump(best_pipeline, model_path)
            mlflow.log_artifact(model_path)
            mlflow.log_metric("best_silhouette_score", best_score)

            print(f"--> Saved Best {name} Model (k=4) | Params: {best_params} | Silhouette: {best_score:.4f}\n")

if __name__ == "__main__":
    run_clustering()