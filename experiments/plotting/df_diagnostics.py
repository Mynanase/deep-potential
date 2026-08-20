"""Composable DF diagnostic plots built from in-memory artifact arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.plotting.style import format_display_unit, label_with_unit


def _require(
    diagnostics: Mapping[str, np.ndarray],
    names: set[str],
) -> None:
    missing = names.difference(diagnostics)
    if missing:
        raise KeyError(
            "DF diagnostics are missing: " + ", ".join(sorted(missing))
        )


def _model_labels(
    diagnostics: Mapping[str, np.ndarray],
    n_models: int,
) -> list[str]:
    if "model_labels" in diagnostics:
        labels = [str(value) for value in diagnostics["model_labels"]]
        if len(labels) >= n_models:
            return labels[:n_models]
    return [f"model_{index}" for index in range(n_models)]


def _nanmedian(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Compute a NaN-aware median without warnings for unsupported cells."""
    result = np.ma.median(np.ma.masked_invalid(values), axis=axis)
    return np.asarray(result.filled(np.nan))


def plot_density_profile(
    diagnostics: Mapping[str, np.ndarray],
    *,
    reference_label: str = "data",
    model_label: str = "DF model",
    radius_label: str = "r",
    density_label: str = "tracer density",
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot radial density and fractional error from one DF evaluation."""
    import matplotlib.pyplot as plt

    _require(
        diagnostics,
        {"radial_edges", "reference_density", "model_density"},
    )
    edges = np.asarray(diagnostics["radial_edges"])
    radius = 0.5 * (edges[:-1] + edges[1:])
    reference = np.asarray(diagnostics["reference_density"])
    model = np.asarray(diagnostics["model_density"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=dpi)
    axes[0].plot(radius, reference, "k-o", lw=1.8, label=reference_label)
    if "model_density_by_model" in diagnostics:
        values = np.asarray(diagnostics["model_density_by_model"])
        if values.shape[0] > 1:
            for density in values:
                axes[0].plot(
                    radius,
                    density,
                    color="#3070b3",
                    alpha=0.25,
                    lw=0.9,
                )
    axes[0].plot(
        radius,
        model,
        "s--",
        color="#3070b3",
        lw=1.8,
        label=model_label,
    )
    if np.all(radius > 0):
        axes[0].set_xscale("log")
    if np.all(reference > 0) and np.all(model > 0):
        axes[0].set_yscale("log")
    axes[0].set_xlabel(radius_label)
    axes[0].set_ylabel(density_label)
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend()

    fractional_error = np.abs(model - reference) / np.maximum(reference, 1e-12)
    axes[1].plot(radius, fractional_error, "s-", color="C3", lw=1.5)
    if np.all(radius > 0):
        axes[1].set_xscale("log")
    axes[1].set_xlabel(radius_label)
    axes[1].set_ylabel("absolute fractional error")
    axes[1].grid(True, alpha=0.25)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cylindrical_rz_density(
    diagnostics: Mapping[str, np.ndarray],
    *,
    reference_label: str = "data",
    model_label: str = "DF model",
    radius_label: str = "R",
    height_label: str = "z",
    density_label: str = "normalized tracer density",
    title: str | None = None,
    density_percentiles: tuple[float, float] = (5.0, 99.5),
    ratio_percentile: float = 98.0,
    dpi: int = 150,
) -> Any:
    """Plot reference/model cylindrical density and their log ratio."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    _require(
        diagnostics,
        {
            "spatial_phi_edges",
            "spatial_r_edges",
            "spatial_z_edges",
            "spatial_reference_density",
            "spatial_reference_count",
            "spatial_model_count",
            "spatial_min_cell_count",
        },
    )
    phi_edges = np.asarray(diagnostics["spatial_phi_edges"])
    radius_edges = np.asarray(diagnostics["spatial_r_edges"])
    z_edges = np.asarray(diagnostics["spatial_z_edges"])
    reference_density = np.asarray(diagnostics["spatial_reference_density"])
    if "spatial_model_median_density" in diagnostics:
        model_density = np.asarray(diagnostics["spatial_model_median_density"])
    else:
        model_density = np.median(
            np.asarray(diagnostics["spatial_model_density"]), axis=0
        )
    reference_count = np.asarray(diagnostics["spatial_reference_count"])
    model_count_raw = np.asarray(diagnostics["spatial_model_count"])
    model_count = (
        np.median(model_count_raw, axis=0)
        if model_count_raw.ndim == reference_count.ndim + 1
        else model_count_raw
    )
    min_cell_count = int(np.asarray(diagnostics["spatial_min_cell_count"]).item())
    n_phi = phi_edges.size - 1

    positive = np.concatenate(
        [
            reference_density[reference_density > 0],
            model_density[model_density > 0],
        ]
    )
    if positive.size == 0:
        raise ValueError("DF spatial densities contain no positive cells.")
    density_vmin, density_vmax = np.percentile(positive, density_percentiles)
    if density_vmax <= density_vmin:
        density_vmax = density_vmin * 10.0
    density_norm = colors.LogNorm(
        vmin=max(float(density_vmin), np.finfo(float).tiny),
        vmax=float(density_vmax),
    )

    valid = (
        (reference_count >= min_cell_count)
        & (model_count >= min_cell_count)
        & (reference_density > 0)
        & (model_density > 0)
    )
    log_ratio = np.full_like(reference_density, np.nan, dtype=np.float64)
    log_ratio[valid] = np.log10(model_density[valid] / reference_density[valid])
    finite_ratio = np.abs(log_ratio[np.isfinite(log_ratio)])
    ratio_limit = (
        max(float(np.percentile(finite_ratio, ratio_percentile)), 0.1)
        if finite_ratio.size
        else 1.0
    )
    ratio_norm = colors.TwoSlopeNorm(
        vmin=-ratio_limit,
        vcenter=0.0,
        vmax=ratio_limit,
    )

    fig, axes = plt.subplots(
        3,
        n_phi,
        figsize=(max(14.0, 3.0 * n_phi), 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        dpi=dpi,
    )
    axes = np.asarray(axes).reshape(3, n_phi)
    density_mappable = None
    ratio_mappable = None
    for phi_index in range(n_phi):
        phi_left = np.degrees(phi_edges[phi_index])
        phi_right = np.degrees(phi_edges[phi_index + 1])
        density_mappable = axes[0, phi_index].pcolormesh(
            radius_edges,
            z_edges,
            np.ma.masked_less_equal(reference_density[phi_index].T, 0.0),
            cmap="magma",
            norm=density_norm,
            shading="auto",
            rasterized=True,
        )
        axes[1, phi_index].pcolormesh(
            radius_edges,
            z_edges,
            np.ma.masked_less_equal(model_density[phi_index].T, 0.0),
            cmap="magma",
            norm=density_norm,
            shading="auto",
            rasterized=True,
        )
        ratio_mappable = axes[2, phi_index].pcolormesh(
            radius_edges,
            z_edges,
            np.ma.masked_invalid(log_ratio[phi_index].T),
            cmap="coolwarm",
            norm=ratio_norm,
            shading="auto",
            rasterized=True,
        )
        axes[0, phi_index].set_title(
            rf"${phi_left:.0f}^\circ\leq\phi<{phi_right:.0f}^\circ$"
        )
        axes[2, phi_index].set_xlabel(radius_label)
    axes[0, 0].set_ylabel(f"{reference_label}\n{height_label}")
    axes[1, 0].set_ylabel(f"{model_label}\n{height_label}")
    axes[2, 0].set_ylabel(f"log10 model/reference\n{height_label}")
    fig.colorbar(
        density_mappable,
        ax=axes[:2, :].ravel().tolist(),
        label=density_label,
        shrink=0.85,
    )
    fig.colorbar(
        ratio_mappable,
        ax=axes[2, :].ravel().tolist(),
        label="log10 density ratio",
        shrink=0.85,
    )
    if title:
        fig.suptitle(title)
    return fig


def plot_velocity_marginals(
    diagnostics: Mapping[str, np.ndarray],
    coordinate: str,
    *,
    reference_label: str = "data",
    model_labels: Sequence[str] | None = None,
    velocity_labels: Sequence[str] = (r"$v_r$", r"$v_\theta$", r"$v_\phi$"),
    velocity_unit: str = "",
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot velocity marginals conditioned on ``r``, ``theta``, or ``phi``."""
    import matplotlib.pyplot as plt

    if coordinate not in {"r", "theta", "phi"}:
        raise ValueError("coordinate must be 'r', 'theta', or 'phi'.")
    prefix = f"conditional_{coordinate}_"
    _require(
        diagnostics,
        {
            "conditional_velocity_edges",
            f"{prefix}edges",
            f"{prefix}reference_hist",
            f"{prefix}model_hist",
            f"{prefix}reference_effective_count",
            f"{prefix}model_count",
            f"{prefix}wasserstein",
        },
    )
    velocity_edges = np.asarray(diagnostics["conditional_velocity_edges"])
    coordinate_edges = np.asarray(diagnostics[f"{prefix}edges"])
    reference_hist = np.asarray(diagnostics[f"{prefix}reference_hist"])
    model_hist = np.asarray(diagnostics[f"{prefix}model_hist"])
    if model_hist.ndim == reference_hist.ndim:
        model_hist = model_hist[None, ...]
    reference_count = np.asarray(
        diagnostics[f"{prefix}reference_effective_count"]
    )
    model_count = np.asarray(diagnostics[f"{prefix}model_count"])
    if model_count.ndim == 1:
        model_count = model_count[None, ...]
    wasserstein = np.asarray(diagnostics[f"{prefix}wasserstein"])
    if wasserstein.ndim == 2:
        wasserstein = wasserstein[None, ...]
    labels = (
        list(model_labels)
        if model_labels is not None
        else _model_labels(diagnostics, model_hist.shape[0])
    )

    n_rows = coordinate_edges.size - 1
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(13.5, max(3.5, 1.9 * n_rows)),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
        dpi=dpi,
    )
    colors = plt.get_cmap("tab10")
    for row in range(n_rows):
        for velocity_index in range(3):
            ax = axes[row, velocity_index]
            ax.stairs(
                reference_hist[row, velocity_index],
                velocity_edges[velocity_index],
                color="black",
                lw=1.8,
                label=reference_label,
            )
            for model_index in range(model_hist.shape[0]):
                ax.stairs(
                    model_hist[model_index, row, velocity_index],
                    velocity_edges[velocity_index],
                    color=colors(model_index % 10),
                    lw=1.1,
                    alpha=0.85,
                    label=labels[model_index],
                )
            finite_w1 = wasserstein[:, row, velocity_index]
            finite_w1 = finite_w1[np.isfinite(finite_w1)]
            if finite_w1.size:
                ax.text(
                    0.98,
                    0.93,
                    f"W1={np.median(finite_w1):.2g}",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=7,
                )
            ax.grid(True, alpha=0.2)
            if row == 0:
                ax.set_title(velocity_labels[velocity_index])
            if row == n_rows - 1:
                formatted_unit = format_display_unit(velocity_unit)
                unit = f" [{formatted_unit}]" if formatted_unit else ""
                ax.set_xlabel(f"velocity{unit}")

        left = coordinate_edges[row]
        right = coordinate_edges[row + 1]
        interval = (
            f"{left:.2g} <= r < {right:.2g}"
            if coordinate == "r"
            else f"{np.degrees(left):.0f} deg <= {coordinate} < "
            f"{np.degrees(right):.0f} deg"
        )
        median_model_count = int(np.median(model_count[:, row]))
        axes[row, 0].set_ylabel(
            f"{interval}\nPDF\n"
            f"N_eff={reference_count[row]:.0f}, N_model~{median_model_count}"
        )
    axes[0, -1].legend(loc="upper left", fontsize=7)
    fig.suptitle(title or f"Velocity marginals conditioned on {coordinate}")
    return fig


def plot_score_distribution(
    diagnostics: Mapping[str, np.ndarray],
    *,
    coordinate_labels: Sequence[str] = ("x", "y", "z", "vx", "vy", "vz"),
    model_labels: Sequence[str] | None = None,
    max_points: int = 2_000,
    seed: int = 0,
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot per-dimension physical-score distributions."""
    import matplotlib.pyplot as plt

    _require(diagnostics, {"scores"})
    scores = np.asarray(diagnostics["scores"])
    if scores.ndim == 2:
        scores = scores[None, ...]
    if scores.ndim != 3 or scores.shape[-1] != len(coordinate_labels):
        raise ValueError(
            "scores must have shape (N, D) or (n_models, N, D) matching "
            "coordinate_labels."
        )
    labels = (
        list(model_labels)
        if model_labels is not None
        else _model_labels(diagnostics, scores.shape[0])
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        scores.shape[1],
        size=min(int(max_points), scores.shape[1]),
        replace=False,
    )
    n_columns = 3
    n_rows = int(np.ceil(scores.shape[-1] / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(13, 3.6 * n_rows),
        squeeze=False,
        dpi=dpi,
    )
    for dim, ax in enumerate(axes.ravel()):
        if dim >= scores.shape[-1]:
            ax.set_visible(False)
            continue
        for model_index in range(scores.shape[0]):
            ax.hist(
                scores[model_index, indices, dim],
                bins=40,
                histtype="step",
                lw=1.2,
                label=labels[model_index],
            )
        ax.set_xlabel(f"score[{coordinate_labels[dim]}]")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(title or "Per-dimension physical-score distribution")
    fig.tight_layout()
    return fig


def _display_label(symbol: str, unit: str | None) -> str:
    return label_with_unit(symbol, unit)


def plot_cylindrical_marginals_by_radius(
    diagnostics: Mapping[str, np.ndarray],
    *,
    length_unit: str = "",
    velocity_unit: str = "",
    reference_label: str = "data",
    model_label: str = "model",
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot five cylindrical marginal PDFs in reference-defined R quantiles."""
    import matplotlib.pyplot as plt

    _require(
        diagnostics,
        {
            "cylindrical_R_edges",
            "cylindrical_component_names",
            "cylindrical_component_edges",
            "cylindrical_reference_pdf",
            "cylindrical_model_pdf",
            "cylindrical_reference_effective_count",
            "cylindrical_model_count",
        },
    )
    radius_edges = np.asarray(diagnostics["cylindrical_R_edges"])
    names = [str(value) for value in diagnostics["cylindrical_component_names"]]
    edges = np.asarray(diagnostics["cylindrical_component_edges"])
    reference_pdf = np.asarray(diagnostics["cylindrical_reference_pdf"])
    model_pdf = np.asarray(diagnostics["cylindrical_model_pdf"])
    if model_pdf.ndim == reference_pdf.ndim:
        model_pdf = model_pdf[None, ...]
    effective_count = np.asarray(diagnostics["cylindrical_reference_effective_count"])
    model_count = np.asarray(diagnostics["cylindrical_model_count"])
    if model_count.ndim == 1:
        model_count = model_count[None, ...]

    n_radius = radius_edges.size - 1
    fig, axes = plt.subplots(
        n_radius,
        5,
        figsize=(12.5, max(5.0, 1.55 * n_radius)),
        squeeze=False,
        constrained_layout=True,
        dpi=dpi,
    )
    labels = {
        "phi": r"$\phi$",
        "z": _display_label(r"$z$", length_unit),
        "v_R": _display_label(r"$v_R$", velocity_unit),
        "v_phi": _display_label(r"$v_\phi$", velocity_unit),
        "v_z": _display_label(r"$v_z$", velocity_unit),
    }
    for radius_index in range(n_radius):
        for component_index, name in enumerate(names):
            ax = axes[radius_index, component_index]
            ax.stairs(
                reference_pdf[radius_index, component_index],
                edges[component_index],
                color="black",
                linewidth=1.35,
                label=reference_label,
            )
            median_model = _nanmedian(
                model_pdf[:, radius_index, component_index], axis=0
            )
            ax.stairs(
                median_model,
                edges[component_index],
                color="#3070b3",
                linewidth=1.25,
                label=model_label,
            )
            ax.grid(True, alpha=0.18)
            if radius_index == 0:
                ax.set_title(labels.get(name, name))
            if radius_index == n_radius - 1:
                ax.set_xlabel(labels.get(name, name))
        unit = (
            f" {format_display_unit(length_unit)}" if length_unit else ""
        )
        axes[radius_index, 0].set_ylabel(
            rf"${radius_edges[radius_index]:.2g}\leq R<"
            rf"{radius_edges[radius_index + 1]:.2g}$" + unit + "\nPDF\n"
            + rf"$N_{{eff}}={effective_count[radius_index]:.0f}$, "
            + rf"$N_m\sim{np.median(model_count[:, radius_index]):.0f}$"
        )
    axes[0, -1].legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def plot_score_field_rv(
    diagnostics: Mapping[str, np.ndarray],
    *,
    length_unit: str = "",
    velocity_unit: str = "",
    score_r_label: str = r"$s_r$",
    score_v_label: str = r"$s_v$",
    min_effective_count: float | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot signed weighted-median radial score components on ``(r, |v|)``."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    _require(
        diagnostics,
        {
            "score_field_r_edges",
            "score_field_v_edges",
            "score_field_effective_count",
            "score_field_r_median",
            "score_field_v_median",
        },
    )
    r_edges = np.asarray(diagnostics["score_field_r_edges"])
    v_edges = np.asarray(diagnostics["score_field_v_edges"])
    effective_count = np.asarray(diagnostics["score_field_effective_count"])
    threshold = (
        float(min_effective_count)
        if min_effective_count is not None
        else float(np.asarray(diagnostics.get("score_field_min_effective_count", 20.0)))
    )
    fields = []
    for key in ("score_field_r_median", "score_field_v_median"):
        values = np.asarray(diagnostics[key])
        if values.ndim == 3:
            values = _nanmedian(values, axis=0)
        fields.append(np.where(effective_count >= threshold, values, np.nan))
    finite = np.concatenate([np.abs(value[np.isfinite(value)]) for value in fields])
    if finite.size == 0:
        raise ValueError("No score-field cells satisfy the support threshold.")
    limit = max(float(np.percentile(finite, 98.0)), np.finfo(float).eps)
    positive = finite[finite > 0]
    linthresh = max(
        min(float(np.percentile(positive, 10.0)) if positive.size else limit, limit * 0.1),
        limit * 1.0e-6,
    )
    norm = colors.SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        dpi=dpi,
    )
    mappable = None
    for ax, values, label in zip(axes, fields, (score_r_label, score_v_label)):
        mappable = ax.pcolormesh(
            r_edges,
            v_edges,
            np.ma.masked_invalid(values.T),
            shading="auto",
            cmap="coolwarm",
            norm=norm,
            rasterized=True,
        )
        ax.set_title(label)
        ax.set_xlabel(_display_label(r"$r$", length_unit))
        ax.set_ylabel(_display_label(r"$|v|$", velocity_unit))
    fig.colorbar(mappable, ax=axes.ravel().tolist(), label="weighted median score")
    if title:
        fig.suptitle(title)
    return fig


def plot_score_slices_by_radius(
    diagnostics: Mapping[str, np.ndarray],
    *,
    length_unit: str = "",
    velocity_unit: str = "",
    min_effective_count: float | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot score medians and weighted 16--84% bands in three R slices."""
    import matplotlib.pyplot as plt

    _require(
        diagnostics,
        {
            "score_field_v_edges",
            "score_field_slice_quantiles",
            "score_field_slice_radii",
            "score_field_slice_effective_count",
            "score_slice_r_q16",
            "score_slice_r_median",
            "score_slice_r_q84",
            "score_slice_v_q16",
            "score_slice_v_median",
            "score_slice_v_q84",
        },
    )
    v_edges = np.asarray(diagnostics["score_field_v_edges"])
    speed = 0.5 * (v_edges[:-1] + v_edges[1:])
    slice_quantiles = np.asarray(diagnostics["score_field_slice_quantiles"])
    slice_radii = np.asarray(diagnostics["score_field_slice_radii"])
    effective_count = np.asarray(diagnostics["score_field_slice_effective_count"])
    threshold = (
        float(min_effective_count)
        if min_effective_count is not None
        else float(np.asarray(diagnostics.get("score_field_min_effective_count", 20.0)))
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.0),
        sharex=True,
        constrained_layout=True,
        dpi=dpi,
    )
    for component_index, component in enumerate(("r", "v")):
        low = np.asarray(diagnostics[f"score_slice_{component}_q16"])
        median = np.asarray(diagnostics[f"score_slice_{component}_median"])
        high = np.asarray(diagnostics[f"score_slice_{component}_q84"])
        if median.ndim == 3:
            low, median, high = (
                _nanmedian(values, axis=0) for values in (low, median, high)
            )
        for slice_index in range(3):
            ax = axes[component_index, slice_index]
            supported = effective_count[slice_index] >= threshold
            ax.fill_between(
                speed,
                np.where(supported, low[slice_index], np.nan),
                np.where(supported, high[slice_index], np.nan),
                color="#3070b3",
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                speed,
                np.where(supported, median[slice_index], np.nan),
                color="#3070b3",
                linewidth=1.3,
            )
            ax.axhline(0.0, color="0.3", linewidth=0.7)
            ax.grid(True, alpha=0.18)
            if component_index == 0:
                unit = (
                    f" {format_display_unit(length_unit)}"
                    if length_unit
                    else ""
                )
                ax.set_title(
                    f"R q={slice_quantiles[slice_index]:.2g} "
                    f"({slice_radii[slice_index]:.3g}{unit})"
                )
            if component_index == 1:
                ax.set_xlabel(_display_label(r"$|v|$", velocity_unit))
        axes[component_index, 0].set_ylabel(r"$s_r$" if component == "r" else r"$s_v$")
    if title:
        fig.suptitle(title)
    return fig


def plot_radial_speed_density(
    diagnostics: Mapping[str, np.ndarray],
    *,
    length_unit: str = "",
    velocity_unit: str = "",
    title: str | None = None,
    dpi: int = 150,
) -> Any:
    """Plot data/model ``(r, |v|)`` densities and ``log10(model/data)``."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    _require(
        diagnostics,
        {
            "radial_speed_r_edges",
            "radial_speed_v_edges",
            "radial_speed_reference_density",
            "radial_speed_model_density",
            "radial_speed_log10_ratio",
        },
    )
    r_edges = np.asarray(diagnostics["radial_speed_r_edges"])
    v_edges = np.asarray(diagnostics["radial_speed_v_edges"])
    reference = np.asarray(diagnostics["radial_speed_reference_density"])
    model = _nanmedian(
        np.asarray(diagnostics["radial_speed_model_density"]), axis=0
    )
    ratio = _nanmedian(
        np.asarray(diagnostics["radial_speed_log10_ratio"]), axis=0
    )
    positive = np.concatenate([reference[reference > 0], model[model > 0]])
    density_norm = colors.LogNorm(vmin=np.percentile(positive, 2), vmax=np.percentile(positive, 99))
    finite_ratio = np.abs(ratio[np.isfinite(ratio)])
    ratio_limit = max(float(np.percentile(finite_ratio, 98)) if finite_ratio.size else 1.0, 0.1)
    ratio_norm = colors.TwoSlopeNorm(vmin=-ratio_limit, vcenter=0.0, vmax=ratio_limit)
    fig = plt.figure(figsize=(11.5, 3.5), layout="constrained", dpi=dpi)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=(1.0, 1.0, 0.055, 1.0, 0.055),
    )
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(grid[0, 3], sharex=axes[0], sharey=axes[0]))
    density_colorbar_axis = fig.add_subplot(grid[0, 2])
    ratio_colorbar_axis = fig.add_subplot(grid[0, 4])
    for ax, values, panel_title in zip(axes[:2], (reference, model), ("Data", "Model")):
        mesh = ax.pcolormesh(
            r_edges,
            v_edges,
            np.ma.masked_less_equal(values.T, 0),
            cmap="viridis",
            norm=density_norm,
            shading="auto",
            rasterized=True,
        )
        ax.set_title(panel_title)
    ratio_mesh = axes[2].pcolormesh(
        r_edges,
        v_edges,
        np.ma.masked_invalid(ratio.T),
        cmap="coolwarm",
        norm=ratio_norm,
        shading="auto",
        rasterized=True,
    )
    axes[2].set_title(r"$\log_{10}(model/data)$")
    for ax in axes:
        ax.set_xlabel(_display_label(r"$r$", length_unit))
    axes[0].set_ylabel(_display_label(r"$|v|$", velocity_unit))
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.colorbar(
        mesh,
        cax=density_colorbar_axis,
        label="probability density",
    )
    fig.colorbar(
        ratio_mesh,
        cax=ratio_colorbar_axis,
        label="log10 density ratio",
    )
    if title:
        fig.suptitle(title)
    return fig


def _finite_phase_space(
    eta: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"{name} must have shape (N, 6), got {eta.shape}.")
    finite = np.all(np.isfinite(eta), axis=1)
    if not np.any(finite):
        raise ValueError(f"{name} contains no finite phase-space rows.")
    eta = eta[finite]
    radius = np.linalg.norm(eta[:, :3], axis=1)
    speed = np.linalg.norm(eta[:, 3:], axis=1)
    return radius, speed


def plot_radial_speed_comparison(
    reference_eta: np.ndarray | None,
    model_eta: np.ndarray,
    *,
    reference_weights: np.ndarray | None = None,
    reference_probability_mass: np.ndarray | None = None,
    radius_range: tuple[float, float] = (0.0, 5.0),
    speed_range: tuple[float, float] = (0.0, 1.5),
    bins: int | tuple[int, int] = 48,
    escape_curve: tuple[np.ndarray, np.ndarray] | None = None,
    reference_label: str = "Reference DF",
    model_label: str = "Normalizing flow",
    radius_label: str = r"$r$",
    speed_label: str = r"$v$",
    unbound_label: str = r"$E > 0$",
    significance_limit: float = 5.0,
    count_gamma: float = 0.5,
    count_cmap: str = "viridis",
    significance_cmap: str = "coolwarm_r",
    title: str | None = None,
    figsize: tuple[float, float] = (11.5, 3.6),
    dpi: int = 150,
) -> Any:
    """Compare reference and flow samples after integrating over all angles.

    Phase-space inputs use the physical order ``(x, y, z, vx, vy, vz)``. By
    default, the reference histogram is rescaled to the model sample count. An
    exact reference probability mass per bin may instead be supplied for an
    analytic validation case. The third panel shows
    ``(n_model - n_reference) / sqrt(n_reference)`` in each ``(r, v)`` bin.
    An optional externally computed escape curve can mark the unbound region
    without coupling this generic plot to a particular potential.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    model_radius, model_speed = _finite_phase_space(
        model_eta,
        name="model_eta",
    )

    radius_min, radius_max = map(float, radius_range)
    speed_min, speed_max = map(float, speed_range)
    if not radius_min < radius_max:
        raise ValueError("radius_range must be strictly increasing.")
    if not speed_min < speed_max:
        raise ValueError("speed_range must be strictly increasing.")
    if isinstance(bins, int):
        radius_bins = speed_bins = int(bins)
    else:
        if len(bins) != 2:
            raise ValueError("bins must be an integer or a pair of integers.")
        radius_bins, speed_bins = (int(value) for value in bins)
    if radius_bins <= 0 or speed_bins <= 0:
        raise ValueError("bins must be positive.")
    if significance_limit <= 0.0:
        raise ValueError("significance_limit must be positive.")
    if count_gamma <= 0.0:
        raise ValueError("count_gamma must be positive.")

    radius_edges = np.linspace(radius_min, radius_max, radius_bins + 1)
    speed_edges = np.linspace(speed_min, speed_max, speed_bins + 1)
    model_hist, _, _ = np.histogram2d(
        model_radius,
        model_speed,
        bins=(radius_edges, speed_edges),
    )
    if reference_probability_mass is not None:
        if reference_weights is not None:
            raise ValueError(
                "reference_weights cannot be used with "
                "reference_probability_mass."
            )
        probability_mass = np.asarray(
            reference_probability_mass,
            dtype=np.float64,
        )
        expected_shape = (radius_bins, speed_bins)
        if probability_mass.shape != expected_shape:
            raise ValueError(
                "reference_probability_mass must have shape "
                f"{expected_shape}, got {probability_mass.shape}."
            )
        if np.any(~np.isfinite(probability_mass)) or np.any(
            probability_mass < 0.0
        ):
            raise ValueError(
                "reference_probability_mass must be finite and non-negative."
            )
        expected_reference = probability_mass * model_radius.size
    else:
        if reference_eta is None:
            raise ValueError(
                "reference_eta is required when no exact reference probability "
                "mass is supplied."
            )
        reference_radius, reference_speed = _finite_phase_space(
            reference_eta,
            name="reference_eta",
        )
        reference_eta_array = np.asarray(reference_eta)
        reference_finite = np.all(np.isfinite(reference_eta_array), axis=1)
        if reference_weights is None:
            weights = np.ones(reference_radius.size, dtype=np.float64)
        else:
            raw_weights = np.asarray(reference_weights, dtype=np.float64)
            if raw_weights.shape != (reference_eta_array.shape[0],):
                raise ValueError(
                    "reference_weights must have shape "
                    f"({reference_eta_array.shape[0]},), got "
                    f"{raw_weights.shape}."
                )
            weights = raw_weights[reference_finite]
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError(
                    "reference_weights must be finite and non-negative."
                )
        reference_total = float(np.sum(weights))
        if reference_total <= 0.0:
            raise ValueError("reference_weights must have a positive total.")
        reference_hist, _, _ = np.histogram2d(
            reference_radius,
            reference_speed,
            bins=(radius_edges, speed_edges),
            weights=weights,
        )
        expected_reference = reference_hist * (
            model_radius.size / reference_total
        )
    significance = np.full_like(expected_reference, np.nan, dtype=np.float64)
    populated = expected_reference > 0.0
    significance[populated] = (
        model_hist[populated] - expected_reference[populated]
    ) / np.sqrt(expected_reference[populated])

    count_vmax = float(max(np.max(expected_reference), np.max(model_hist)))
    if count_vmax <= 0.0:
        raise ValueError("No samples fall inside the requested plotting ranges.")
    count_norm = colors.PowerNorm(
        gamma=float(count_gamma),
        vmin=0.0,
        vmax=count_vmax,
    )
    significance_norm = colors.Normalize(
        vmin=-float(significance_limit),
        vmax=float(significance_limit),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        dpi=dpi,
    )
    axes[0].pcolormesh(
        radius_edges,
        speed_edges,
        expected_reference.T,
        cmap=count_cmap,
        norm=count_norm,
        shading="auto",
        rasterized=True,
    )
    axes[1].pcolormesh(
        radius_edges,
        speed_edges,
        model_hist.T,
        cmap=count_cmap,
        norm=count_norm,
        shading="auto",
        rasterized=True,
    )
    significance_mappable = axes[2].pcolormesh(
        radius_edges,
        speed_edges,
        np.ma.masked_invalid(significance.T),
        cmap=significance_cmap,
        norm=significance_norm,
        shading="auto",
        rasterized=True,
    )

    for index, (axis, panel_title) in enumerate(
        zip(
            axes,
            (
                reference_label,
                model_label,
                f"{model_label} - {reference_label}",
            ),
        )
    ):
        axis.set_title(panel_title)
        axis.set_xlabel(radius_label)
        axis.set_box_aspect(1)
        if index > 0:
            axis.tick_params(labelleft=False)
        if escape_curve is not None:
            escape_radius = np.asarray(escape_curve[0], dtype=np.float64)
            escape_speed = np.asarray(escape_curve[1], dtype=np.float64)
            if escape_radius.shape != escape_speed.shape:
                raise ValueError(
                    "escape_curve radius and speed arrays must have matching "
                    "shapes."
                )
            curve_finite = np.isfinite(escape_radius) & np.isfinite(escape_speed)
            curve_color = "white" if index < 2 else "black"
            axis.plot(
                escape_radius[curve_finite],
                escape_speed[curve_finite],
                color=curve_color,
                linewidth=1.8,
            )
            axis.text(
                0.96,
                0.94,
                unbound_label,
                color=curve_color,
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=13,
            )
    axes[0].set_ylabel(speed_label)
    colorbar = fig.colorbar(
        significance_mappable,
        ax=axes,
        extend="both",
        fraction=0.035,
        pad=0.025,
    )
    colorbar.set_label(r"Poisson significance ($\sigma$)")
    if title:
        fig.suptitle(title)
    return fig
