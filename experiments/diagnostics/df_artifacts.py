"""Readers for persisted DF evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.diagnostics.artifact_paths import resolve_artifact


def _artifact_path(eval_dir: str | Path, filename: str) -> Path:
    return resolve_artifact(eval_dir, filename)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_df_metrics(eval_dir: str | Path) -> dict[str, Any]:
    """Load ``df_metrics.json`` from a flat evaluation directory."""
    path = _artifact_path(eval_dir, "df_metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_df_diagnostics(eval_dir: str | Path) -> dict[str, np.ndarray]:
    """Load ``df_diagnostics.npz`` without loading generated samples."""
    return _load_npz(_artifact_path(eval_dir, "df_diagnostics.npz"))


def load_df_samples(eval_dir: str | Path) -> dict[str, np.ndarray]:
    """Load the potentially large ``df_samples.npz`` artifact explicitly."""
    return _load_npz(_artifact_path(eval_dir, "df_samples.npz"))
