import json
from datetime import UTC, datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any
import joblib

from .manifest import (
    MANIFEST_FILE,
    PAYLOAD_FILE,
    ArtifactManifest,
    TRACKED_LIBRARIES,
    model_key,
)

def _get_lib_version(lib: str) -> str:
    try:
        return version(lib)
    except PackageNotFoundError:
        return "unknown"


def export_telco_artifacts(
    models_fit: dict[str, Any],
    metrics_map: dict[str, dict[str, Any]],
    out_dir: Path | str,
    artifact_version: str,
    notes: str | None = None,
    diagnostics: dict[str, dict[str, float | int | bool]] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    models_map = {}
    for name, model_obj in models_fit.items():
        mkey = model_key(name)
        fname = f"{mkey}_best.joblib"
        with (out_dir / fname).open("wb") as f:
            joblib.dump(model_obj, f)
        models_map[mkey] = fname

    (out_dir / PAYLOAD_FILE).write_text(json.dumps(metrics_map, indent=4), encoding="utf-8")

    lib_versions = {lib: _get_lib_version(lib) for lib in TRACKED_LIBRARIES}

    manifest = ArtifactManifest(
        artifact_version=artifact_version,
        created_at=datetime.now(UTC),
        library_versions=lib_versions,
        models=models_map,
        diagnostics=diagnostics or {},
        notes=notes,
    )
    (out_dir / MANIFEST_FILE).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out_dir