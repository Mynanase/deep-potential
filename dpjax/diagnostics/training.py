"""Training-curve diagnostics from persisted ``metrics.csv`` files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def load_metrics(path: str | Path) -> dict[str, np.ndarray]:
    """Read numeric metric columns while ignoring malformed/repeated headers."""
    path = Path(path)
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return {}
        fieldnames = list(reader.fieldnames)
        columns: dict[str, list[float]] = {key: [] for key in fieldnames}

        def maybe_float(value: str | None) -> float:
            if value is None or not value.strip():
                return float("nan")
            try:
                return float(value)
            except ValueError:
                return float("nan")

        for row in reader:
            if not np.isfinite(maybe_float(row.get("step"))):
                continue
            for key in fieldnames:
                columns[key].append(maybe_float(row.get(key)))

    return {
        key: np.asarray(values, dtype=np.float32)
        for key, values in columns.items()
        if values
    }


def plot_training_metrics(
    metrics: dict[str, np.ndarray],
    *,
    title: str | None = None,
    dpi: int = 150,
) -> dict[str, Any]:
    """Create matplotlib figures without writing them to disk."""
    import matplotlib.pyplot as plt

    step = metrics.get("step")
    if step is None:
        raise ValueError("metrics are missing the 'step' column.")
    figures: dict[str, Any] = {}

    if "loss" in metrics:
        fig, ax = plt.subplots(dpi=dpi)
        ax.plot(step, metrics["loss"], lw=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.2)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        figures["loss"] = fig

    score_keys = ("score_p50", "score_p99", "score_max_abs")
    if any(key in metrics for key in score_keys):
        fig, ax = plt.subplots(dpi=dpi)
        labels = {
            "score_p50": "score | p50",
            "score_p99": "score | p99",
            "score_max_abs": "score | max",
        }
        for key in score_keys:
            if key in metrics:
                ax.plot(step, metrics[key], label=labels[key], lw=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("|score| stats")
        ax.grid(True, alpha=0.2)
        ax.legend()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        figures["df_score_stats"] = fig

    residual_keys = ("residual_mean", "residual_std", "residual_p99_abs")
    if any(key in metrics for key in residual_keys):
        fig, ax = plt.subplots(dpi=dpi)
        labels = {
            "residual_mean": "mean",
            "residual_std": "std",
            "residual_p99_abs": "p99(|r|)",
        }
        for key in residual_keys:
            if key in metrics:
                ax.plot(step, metrics[key], label=labels[key], lw=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("residual stats")
        ax.grid(True, alpha=0.2)
        ax.legend()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        figures["phi_residual_stats"] = fig

    return figures
