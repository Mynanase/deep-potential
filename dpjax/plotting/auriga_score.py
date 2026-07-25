"""Plots for validating a Halo DF score against simulator acceleration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


COMPONENT_LABELS = {
    0: "cold",
    1: "warm",
    2: "hot",
    3: "counter",
}


def _scatter_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(x), np.asarray(y)])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    bound = float(np.percentile(np.abs(finite), 99.0))
    if not bound > 0:
        bound = 1.0
    return -bound, bound


def _draw_term_scatter(
    ax,
    transport_term: np.ndarray,
    acceleration_term: np.ndarray,
    *,
    title: str,
) -> None:
    x = np.asarray(transport_term)
    y = -np.asarray(acceleration_term)
    lo, hi = _scatter_limits(x, y)
    ax.hexbin(
        x,
        y,
        gridsize=80,
        bins="log",
        mincnt=1,
        extent=(lo, hi, lo, hi),
        cmap="viridis",
    )
    ax.plot([lo, hi], [lo, hi], color="tab:red", lw=1.2, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\mathbf{v}\cdot s_x$")
    ax.set_ylabel(r"$-\mathbf{a}_{\rm true}\cdot s_v$")
    ax.set_title(title)


def plot_truth_cbe_diagnostics(
    diagnostics: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    radial_profile: Mapping[str, Any],
    *,
    component: np.ndarray | None = None,
    fig_dir: str | Path,
    dpi: int = 180,
) -> list[Path]:
    """Create truth-acceleration CBE scatter, histogram, and profile plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    transport = np.asarray(diagnostics["transport_term"])
    acceleration = np.asarray(diagnostics["acceleration_term"])
    residual = np.asarray(diagnostics["residual"])
    normalized = np.asarray(diagnostics["normalized_residual"])

    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=dpi)
    _draw_term_scatter(
        ax,
        transport,
        acceleration,
        title="Truth-acceleration CBE terms",
    )
    comparison_metrics = metrics["transport_vs_negative_acceleration"]
    normalized_metrics = metrics["normalized_residual"]
    ax.text(
        0.04,
        0.96,
        (
            f"slope={comparison_metrics['slope_through_origin']:.3f}\n"
            f"Pearson r={comparison_metrics['pearson_r']:.3f}\n"
            f"median normalized={normalized_metrics['median']:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    path = fig_dir / "cbe_truth_terms_scatter.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    if component is not None:
        component = np.asarray(component)
        if component.shape != residual.shape:
            raise ValueError(f"Expected component shape {residual.shape}.")
        present = [
            value for value in COMPONENT_LABELS if np.any(component == value)
        ]
        if present:
            fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=dpi)
            for ax, value in zip(axes.flat, present):
                mask = component == value
                _draw_term_scatter(
                    ax,
                    transport[mask],
                    acceleration[mask],
                    title=f"{COMPONENT_LABELS[value]} (n={np.count_nonzero(mask)})",
                )
            for ax in axes.flat[len(present) :]:
                ax.set_visible(False)
            fig.suptitle("Truth-acceleration CBE terms by component")
            fig.tight_layout()
            path = fig_dir / "cbe_truth_terms_by_component.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=dpi)
    finite_residual = residual[np.isfinite(residual)]
    signed_bound = (
        float(np.percentile(np.abs(finite_residual), 99.5))
        if finite_residual.size
        else 1.0
    )
    if not signed_bound > 0:
        signed_bound = 1.0
    axes[0].hist(
        finite_residual,
        bins=101,
        range=(-signed_bound, signed_bound),
        log=True,
        color="tab:blue",
        alpha=0.8,
    )
    axes[0].axvline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel(r"$r_{\rm true}$")
    axes[0].set_ylabel("count")
    axes[0].set_title("Signed truth-field CBE residual")

    positive_abs = np.abs(finite_residual)
    positive_abs = positive_abs[positive_abs > 0]
    if positive_abs.size:
        lower = max(float(np.percentile(positive_abs, 0.5)), np.finfo(float).tiny)
        upper = max(float(np.percentile(positive_abs, 99.5)), lower * 10.0)
        axes[1].hist(
            positive_abs,
            bins=np.geomspace(lower, upper, 80),
            log=True,
            color="tab:orange",
            alpha=0.8,
        )
        axes[1].set_xscale("log")
    residual_metrics = metrics["residual"]
    axes[1].set_xlabel(r"$|r_{\rm true}|$")
    axes[1].set_ylabel("count")
    axes[1].set_title(
        "Absolute residual\n"
        f"median={residual_metrics['median_abs']:.3g}, "
        f"p90={residual_metrics['p90_abs']:.3g}, "
        f"p99={residual_metrics['p99_abs']:.3g}"
    )
    fig.tight_layout()
    path = fig_dir / "cbe_truth_residual_hist.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=dpi)
    ax.hist(
        normalized[np.isfinite(normalized)],
        bins=np.linspace(0.0, 1.0, 81),
        log=True,
        color="tab:green",
        alpha=0.85,
    )
    ax.axvline(0.1, color="tab:orange", ls="--", label="0.1")
    ax.axvline(0.2, color="tab:red", ls="--", label="0.2")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        r"$|r_{\rm true}|/(|T_x|+|T_a|+\epsilon)$"
    )
    ax.set_ylabel("count")
    ax.set_title(
        "Normalized truth-field CBE residual\n"
        f"median={normalized_metrics['median']:.3f}, "
        f"p90={normalized_metrics['p90']:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "cbe_truth_normalized_residual_hist.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    rows = radial_profile["bins"]
    radius = np.array(
        [np.sqrt(row["r_left"] * row["r_right"]) for row in rows],
        dtype=np.float64,
    )
    residual_median = np.array(
        [row["residual_median_abs"] for row in rows],
        dtype=np.float64,
    )
    residual_p90 = np.array(
        [row["residual_p90_abs"] for row in rows],
        dtype=np.float64,
    )
    normalized_median = np.array(
        [row["normalized_median"] for row in rows],
        dtype=np.float64,
    )
    normalized_p90 = np.array(
        [row["normalized_p90"] for row in rows],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=dpi)
    axes[0].plot(radius, residual_median, marker="o", label="median")
    axes[0].plot(radius, residual_p90, marker="o", label="p90")
    axes[0].set_xscale("log")
    if np.any(residual_p90 > 0):
        axes[0].set_yscale("log")
    axes[0].set_xlabel("radius")
    axes[0].set_ylabel(r"$|r_{\rm true}|$")
    axes[0].set_title("Absolute residual radial profile")
    axes[0].legend()

    axes[1].plot(radius, normalized_median, marker="o", label="median")
    axes[1].plot(radius, normalized_p90, marker="o", label="p90")
    axes[1].axhline(0.1, color="tab:orange", ls="--", lw=0.8)
    axes[1].axhline(0.2, color="tab:red", ls="--", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("radius")
    axes[1].set_ylabel("normalized residual")
    axes[1].set_title("Normalized residual radial profile")
    axes[1].legend()
    fig.tight_layout()
    path = fig_dir / "cbe_truth_residual_profiles.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)
    return written
