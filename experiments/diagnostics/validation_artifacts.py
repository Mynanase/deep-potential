"""Readers for optional, separately generated truth-validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.diagnostics.artifact_paths import artifact_candidates


def _artifact_path(
    validation_dir: str | Path,
    filename: str,
    *,
    kind: str | None = None,
) -> Path:
    names = [f"validation_{kind}_{filename}" if kind else filename, filename]
    for name in names:
        candidates = list(artifact_candidates(validation_dir, name))
        if kind:
            candidates.append(Path(validation_dir) / "validation" / kind / filename)
        for path in candidates:
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"Missing truth-validation artifact {filename!r} under {validation_dir}."
    )


def load_validation_metrics(
    validation_dir: str | Path,
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Load the optional ``metrics.json`` artifact."""
    path = _artifact_path(validation_dir, "metrics.json", kind=kind)
    return json.loads(path.read_text(encoding="utf-8"))


def load_validation_diagnostics(
    validation_dir: str | Path,
    *,
    kind: str | None = None,
) -> dict[str, np.ndarray]:
    """Load the optional ``diagnostics.npz`` artifact."""
    path = _artifact_path(validation_dir, "diagnostics.npz", kind=kind)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}
