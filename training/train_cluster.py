import os
from datetime import datetime
import joblib
import mlflow
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline

from .dataset_load import fetch_and_clean_data, get_preprocessor
from .export import export_artifacts
from .manifest import model_key

def run_clustering():
    mlflow.end_run()
    mlflow.set_experiment("payment_clustering_k4")

    df = fetch_and_clean_data()
    y = df['Churn'] if 'Churn' in df.columns else None
    X = df.drop(columns=['Churn'], errors='ignore')

    preprocessor = get_preprocessor(X.columns)
    clustering_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])

    with mlflow.start_run(run_name="KMeans_k4_Pipeline"):
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("n_clusters", 4)

        clustering_pipeline.fit(X)
        
        X_prep = clustering_pipeline.named_steps['preprocessing'].transform(X)
        labels = clustering_pipeline.named_steps['model'].labels_

        sil = silhouette_score(X_prep, labels)
        cal = calinski_harabasz_score(X_prep, labels)

        cluster_churn = {}
        if y is not None:
            cluster_churn = pd.Series(y.values).groupby(labels).mean().to_dict()
            print(f"Churn rate per cluster: {cluster_churn}")

        mlflow.log_metric("silhouette_score", sil)
        mlflow.log_metric("calinski_harabasz_score", cal)

        os.makedirs("artifacts/_temp_mlflow", exist_ok=True)
        temp_path = "artifacts/_temp_mlflow/kmeans_best.joblib"
        joblib.dump(clustering_pipeline, temp_path)
        mlflow.log_artifact(temp_path)

        models_fit = {"KMeans": clustering_pipeline}
        metrics_map = {
            model_key("KMeans"): {
                "silhouette_score": float(sil),
                "calinski_harabasz_score": float(cal),
                "cluster_churn_rates": {str(k): float(v) for k, v in cluster_churn.items()},
                "best_params": {"n_clusters": 4}
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        version_dir = f"artifacts/cluster_{timestamp}"

        saved_path = export_artifacts(
            models_fit=models_fit,
            metrics_map=metrics_map,
            out_dir=version_dir,
            artifact_version=f"cluster_{timestamp}",
            notes="K-Means k=4 clustering pipeline",
        )
        print(f"Cluster model and manifest exported to: {saved_path}")

if __name__ == "__main__":
    run_clustering()