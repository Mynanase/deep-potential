"""Readers for optional, separately generated truth-validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _artifact_path(validation_dir: str | Path, filename: str) -> Path:
    path = Path(validation_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing truth-validation artifact: {path}")
    return path


def load_validation_metrics(validation_dir: str | Path) -> dict[str, Any]:
    """Load the optional ``metrics.json`` artifact."""
    path = _artifact_path(validation_dir, "metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_validation_diagnostics(
    validation_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Load the optional ``diagnostics.npz`` artifact."""
    path = _artifact_path(validation_dir, "diagnostics.npz")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}
