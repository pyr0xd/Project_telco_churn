import mlflow
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_data")
DB_PATH = os.path.join(MLFLOW_DIR, "mlflow.db")
ARTIFACT_PATH = os.path.join(MLFLOW_DIR, "mlruns")


def init_mlflow(experiment_name="basemodel_test"):
    mlflow.set_tracking_uri("sqlite:///training/mlflow_data/mlflow.db")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    experiment = mlflow.get_experiment_by_name(experiment_name)
    print("Found experiment:", experiment)

    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location="./mlruns")
        print("Created new experiment:", experiment_name)

    mlflow.set_experiment(experiment_name)