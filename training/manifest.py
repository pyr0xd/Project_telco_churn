import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANIFEST_FILE = "manifest.json"
PAYLOAD_FILE = "telco_metrics_payload.json"
SCHEMA_VERSION = 1
TRACKED_LIBRARIES = ("scikit-learn", "pandas", "numpy", "joblib", "mlflow")


def model_key(model_name: str) -> str:
    """Registry key for a model name: 'LogisticRegression' -> 'logisticregression'."""
    return model_name.strip().lower()

def model_filename(name: str) -> str:
    return f"{model_key(name)}_best.joblib"


class ModelEvaluationPayload(BaseModel):
    accuracy: float
    f1_score: float
    best_params: dict[str, Any] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    artifact_version: str
    created_at: datetime
    library_versions: dict[str, str] = Field(default_factory=dict)
    payload_file: str = PAYLOAD_FILE
    models: dict[str, str]
    diagnostics: dict[str, dict[str, float | int | bool]] = Field(default_factory=dict)
    notes: str | None = None


def load_manifest(artifact_dir: Path | str) -> ArtifactManifest:
    return ArtifactManifest.model_validate_json(
        (Path(artifact_dir) / MANIFEST_FILE).read_text(encoding="utf-8")
    )


def load_payload(artifact_dir: Path | str, manifest: ArtifactManifest) -> dict[str, Any]:
    return json.loads((Path(artifact_dir) / manifest.payload_file).read_text(encoding="utf-8"))