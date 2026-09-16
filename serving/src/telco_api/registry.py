from pathlib import Path
import joblib

class ModelRegistry:
    def __init__(self):
        self.model = None

    def load_model(self, path: str):
        file_path = Path(path)
        if file_path.exists():
            self.model = joblib.load(file_path)
            print(f"Loaded model from {path}")
        else:
            print(f"Model file not found at {path}. Running in mock mode.")

registry = ModelRegistry()