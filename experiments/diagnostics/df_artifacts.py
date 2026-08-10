"""Readers and plots for persisted DF diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_df_evaluation(path: str | Path) -> dict[str, Any]:
    """Load available DF metrics and arrays without recomputing the model."""
    path = Path(path)
    result: dict[str, Any] = {}
    metrics_path = path / "auriga_df_metrics.json"
    diagnostics_path = path / "auriga_df_diagnostics.npz"
    samples_path = path / "df_samples.npz"
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    if diagnostics_path.exists():
        with np.load(diagnostics_path) as data:
            result["diagnostics"] = {key: data[key] for key in data.files}
    if samples_path.exists():
        with np.load(samples_path) as data:
            result["samples"] = {key: data[key] for key in data.files}
    return result


def plot_density_profile(
    diagnostics: dict[str, np.ndarray],
    *,
    dpi: int = 150,
) -> Any:
    """Create a radial data/model density figure from saved DF arrays."""
    import matplotlib.pyplot as plt

    required = {"radial_edges", "reference_density", "model_density"}
    missing = required.difference(diagnostics)
    if missing:
        raise ValueError("DF diagnostics are missing: " + ", ".join(sorted(missing)))
    edges = np.asarray(diagnostics["radial_edges"])
    radius = 0.5 * (edges[:-1] + edges[1:])
    reference = np.asarray(diagnostics["reference_density"])
    model = np.asarray(diagnostics["model_density"])

    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(radius, reference, marker="o", label="data")
    ax.plot(radius, model, marker="o", label="DF")
    if "model_density_by_model" in diagnostics:
        for values in np.asarray(diagnostics["model_density_by_model"]):
            ax.plot(radius, values, color="C1", alpha=0.2, lw=0.8)
    if np.all(radius > 0):
        ax.set_xscale("log")
    if np.all(reference > 0) and np.all(model > 0):
        ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("tracer density")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig
