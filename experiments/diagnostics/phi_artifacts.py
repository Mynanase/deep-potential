"""Readers for persisted Phi evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _artifact_path(eval_dir: str | Path, filename: str) -> Path:
    path = Path(eval_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phi evaluation artifact: {path}")
    return path


def load_phi_metrics(eval_dir: str | Path) -> dict[str, Any]:
    """Load ``phi_metrics.json`` from a flat evaluation directory."""
    path = _artifact_path(eval_dir, "phi_metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_phi_diagnostics(eval_dir: str | Path) -> dict[str, np.ndarray]:
    """Load the consolidated ``phi_diagnostics.npz`` artifact."""
    path = _artifact_path(eval_dir, "phi_diagnostics.npz")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}
