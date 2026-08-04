"""Readers and lightweight plots for persisted Phi diagnostic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_phi_evaluation(path: str | Path) -> dict[str, Any]:
    """Load Phi metrics and radial curves from an evaluation directory."""
    path = Path(path)
    result: dict[str, Any] = {}
    stats_path = path / "eval_stats.json"
    if stats_path.exists():
        result["stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
    radial_path = path / "radial_curves_plummer.npz"
    if not radial_path.exists():
        radial_path = path / "radial_curves.npz"
    if radial_path.exists():
        with np.load(radial_path) as data:
            result["radial"] = {key: data[key] for key in data.files}
    return result


def plot_radial_curves(
    radial: dict[str, np.ndarray],
    *,
    dpi: int = 150,
) -> dict[str, Any]:
    """Create potential, acceleration, and density figures from saved arrays."""
    import matplotlib.pyplot as plt

    if "r" not in radial:
        raise ValueError("radial curves are missing 'r'.")
    radius = radial["r"]
    figures: dict[str, Any] = {}

    potential_key = (
        "phi_learned_shift" if "phi_learned_shift" in radial else "phi_learned"
    )
    if potential_key in radial:
        fig, ax = plt.subplots(dpi=dpi)
        if "phi_true" in radial:
            ax.plot(radius, radial["phi_true"], label="truth")
        ax.plot(radius, radial[potential_key], label="learned")
        ax.set_xscale("log")
        ax.set_xlabel("r")
        ax.set_ylabel("Phi")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        figures["potential"] = fig

    if "ar_learned" in radial:
        fig, ax = plt.subplots(dpi=dpi)
        if "ar_true" in radial:
            ax.plot(radius, radial["ar_true"], label="truth")
        ax.plot(radius, radial["ar_learned"], label="learned")
        ax.set_xscale("log")
        ax.set_xlabel("r")
        ax.set_ylabel("radial acceleration")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        figures["acceleration"] = fig

    if "rho_learned" in radial:
        fig, ax = plt.subplots(dpi=dpi)
        truth_key = "rho_analytic" if "rho_analytic" in radial else "rho_true"
        if truth_key in radial:
            ax.plot(radius, radial[truth_key], label="truth")
        ax.plot(radius, radial["rho_learned"], label="learned")
        ax.set_xscale("log")
        if np.all(np.asarray(radial["rho_learned"]) > 0):
            ax.set_yscale("log")
        ax.set_xlabel("r")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        figures["density"] = fig

    return figures
