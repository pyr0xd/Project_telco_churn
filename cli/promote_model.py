import glob
import os
import shutil
import sys

def promote_latest_model():
    production_dirs = sorted(glob.glob("artifacts/production_*"), reverse=True)
    if not production_dirs:
        print("Inga production-mappar hittades (artifacts/production_*). Kör 'train-prod' först.")
        return

    latest = production_dirs[0]
    model_files = glob.glob(os.path.join(latest, "*_best.joblib"))
    if not model_files:
        model_files = glob.glob(os.path.join(latest, "*.joblib"))

    if not model_files:
        print(f"Ingen .joblib-fil hittades i {latest}.")
        return

    source_path = model_files[0]

    dest_dir = os.path.join("serving", "src", "telco_api", "model_store")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "model.joblib")

    shutil.copy(source_path, dest_path)
    print(f"Modell befordrad:\n  Från: {source_path}\n  Till: {dest_path}")

if __name__ == "__main__":
    promote_latest_model()