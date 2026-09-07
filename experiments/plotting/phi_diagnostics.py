"""Composable potential diagnostics built only from in-memory arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.plotting.style import format_display_unit


def _profile_arrays(
    radius: np.ndarray,
    model: np.ndarray,
    truth: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    radius = np.asarray(radius, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    truth = None if truth is None else np.asarray(truth, dtype=np.float64)
    if radius.ndim != 1 or model.shape != radius.shape:
        raise ValueError("radius and model values must be one-dimensional and aligned.")
    if truth is not None and truth.shape != radius.shape:
        raise ValueError("truth values must have the same shape as radius.")
    return radius, model, truth


def _line_kwargs(
    defaults: dict[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(defaults)
    if overrides is not None:
        result.update(overrides)
    return result


def plot_potential_profile(
    radius: np.ndarray,
    model_potential: np.ndarray,
    *,
    truth_potential: np.ndarray | None = None,
    model_label: str = "model",
    truth_label: str = "truth",
    radius_label: str = "r",
    potential_label: str = r"$\Phi$",
    title: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    truth_kwargs: Mapping[str, Any] | None = None,
    dpi: int = 150,
) -> Any:
    """Plot an offset-aligned potential profile."""
    import matplotlib.pyplot as plt

    radius, model, truth = _profile_arrays(
        radius,
        model_potential,
        truth_potential,
    )
    fig, ax = plt.subplots(constrained_layout=True, dpi=dpi)
    if truth is not None:
        ax.plot(
            radius,
            truth,
            label=truth_label,
            **_line_kwargs({"color": "#e1812c", "linewidth": 1.8}, truth_kwargs),
        )
    ax.plot(
        radius,
        model,
        label=model_label,
        **_line_kwargs(
            {"color": "#3070b3", "linewidth": 1.5, "linestyle": "--"},
            model_kwargs,
        ),
    )
    if np.all(radius > 0):
        ax.set_xscale("log")
    ax.set_xlabel(radius_label)
    ax.set_ylabel(potential_label)
    ax.grid(True, alpha=0.2)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_mass_density_profile(
    radius: np.ndarray,
    model_density: np.ndarray,
    *,
    truth_density: np.ndarray | None = None,
    model_label: str = "model",
    truth_label: str = "truth",
    radius_label: str = "r",
    density_label: str = r"$\rho$",
    title: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    truth_kwargs: Mapping[str, Any] | None = None,
    dpi: int = 150,
) -> Any:
    """Plot total mass density without discarding negative predictions."""
    import matplotlib.pyplot as plt

    radius, model, truth = _profile_arrays(
        radius,
        model_density,
        truth_density,
    )
    fig, ax = plt.subplots(constrained_layout=True, dpi=dpi)
    if truth is not None:
        ax.plot(
            radius,
            truth,
            label=truth_label,
            **_line_kwargs({"color": "#e1812c", "linewidth": 1.8}, truth_kwargs),
        )
    ax.plot(
        radius,
        model,
        label=model_label,
        **_line_kwargs(
            {"color": "#3070b3", "linewidth": 1.5, "linestyle": "--"},
            model_kwargs,
        ),
    )
    if np.all(radius > 0):
        ax.set_xscale("log")
    plotted = model if truth is None else np.concatenate([model, truth])
    finite = plotted[np.isfinite(plotted)]
    if finite.size and np.all(finite > 0):
        ax.set_yscale("log")
    elif finite.size:
        nonzero = np.abs(finite[finite != 0])
        linthresh = max(
            float(np.percentile(nonzero, 10.0)) if nonzero.size else 1.0e-12,
            1.0e-12,
        )
        ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel(radius_label)
    ax.set_ylabel(density_label)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_mass_density_residual(
    radius: np.ndarray,
    model_density: np.ndarray,
    truth_density: np.ndarray,
    *,
    radius_label: str = "r",
    residual_label: str = r"$\rho_{model}-\rho_{truth}$",
    title: str | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    dpi: int = 150,
) -> Any:
    """Plot signed total-density residuals on a separate Figure."""
    import matplotlib.pyplot as plt

    radius, model, truth = _profile_arrays(
        radius,
        model_density,
        truth_density,
    )
    assert truth is not None
    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(
        radius,
        model - truth,
        **_line_kwargs({"color": "C2", "linewidth": 1.2}, plot_kwargs),
    )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    if np.all(radius > 0):
        ax.set_xscale("log")
    ax.set_xlabel(radius_label)
    ax.set_ylabel(residual_label)
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_radial_acceleration_profile(
    radius: np.ndarray,
    model_acceleration: np.ndarray,
    *,
    truth_acceleration: np.ndarray | None = None,
    model_label: str = "model",
    truth_label: str = "truth",
    radius_label: str = "r",
    acceleration_label: str = r"$a_r$",
    title: str | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    truth_kwargs: Mapping[str, Any] | None = None,
    dpi: int = 150,
) -> Any:
    """Plot the signed radial acceleration profile."""
    import matplotlib.pyplot as plt

    radius, model, truth = _profile_arrays(
        radius,
        model_acceleration,
        truth_acceleration,
    )
    fig, ax = plt.subplots(dpi=dpi)
    if truth is not None:
        ax.plot(
            radius,
            truth,
            label=truth_label,
            **_line_kwargs({"color": "#e1812c", "linewidth": 1.8}, truth_kwargs),
        )
    ax.plot(
        radius,
        model,
        label=model_label,
        **_line_kwargs(
            {"color": "#3070b3", "linewidth": 1.5, "linestyle": "--"},
            model_kwargs,
        ),
    )
    if np.all(radius > 0):
        ax.set_xscale("log")
    ax.set_xlabel(radius_label)
    ax.set_ylabel(acceleration_label)
    ax.grid(True, alpha=0.2)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _slice_arrays(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional coordinate arrays.")
    if values.shape != (y.size, x.size):
        raise ValueError(
            f"Expected slice shape {(y.size, x.size)}, got {values.shape}."
        )
    invalid = ~np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as the slice.")
        invalid |= ~mask
    return x, y, np.ma.array(values, mask=invalid)


def _finite_slice_values(values: np.ma.MaskedArray) -> np.ndarray:
    finite = np.asarray(values.compressed(), dtype=np.float64)
    if finite.size == 0:
        raise ValueError("The requested slice contains no finite supported values.")
    return finite


def plot_potential_slice(
    x: np.ndarray,
    y: np.ndarray,
    model_potential: np.ndarray,
    *,
    truth_potential: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    truth_contour_levels: Sequence[float] | None = None,
    x_label: str = "x",
    y_label: str = "y",
    potential_label: str = r"$\Phi$",
    title: str | None = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
    cmap: str = "magma",
    dpi: int = 150,
) -> Any:
    """Plot one model potential slice with optional truth contours."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    x, y, model = _slice_arrays(x, y, model_potential, mask)
    finite = _finite_slice_values(model)
    vmin, vmax = np.percentile(finite, percentiles)
    if vmax <= vmin:
        vmax = vmin + max(abs(float(vmin)) * 1.0e-6, 1.0e-12)
    fig, ax = plt.subplots(constrained_layout=True, dpi=dpi)
    mappable = ax.pcolormesh(
        x,
        y,
        model,
        shading="auto",
        cmap=cmap,
        norm=colors.Normalize(vmin=float(vmin), vmax=float(vmax)),
        rasterized=True,
    )
    if truth_potential is not None:
        _, _, truth = _slice_arrays(x, y, truth_potential, mask)
        truth_values = _finite_slice_values(truth)
        levels = (
            np.asarray(truth_contour_levels, dtype=np.float64)
            if truth_contour_levels is not None
            else np.unique(np.percentile(truth_values, [20.0, 50.0, 80.0]))
        )
        if levels.size:
            ax.contour(x, y, truth, levels=levels, colors="#e1812c", linewidths=0.7)
    fig.colorbar(mappable, ax=ax, label=potential_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return fig


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges from pcolormesh-style cell-center coordinates."""
    centers = np.asarray(centers, dtype=np.float64)
    inner = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[first], inner, [last]])


def plot_mass_density_slice(
    x: np.ndarray,
    y: np.ndarray,
    model_density: np.ndarray,
    *,
    truth_density: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    truth_contour_levels: Sequence[float] | None = None,
    data_positions: np.ndarray | None = None,
    min_data_count: int = 1,
    smooth_cells: float | None = None,
    length_unit: str = "",
    x_label: str = "x",
    y_label: str = "y",
    density_label: str = r"$\rho$",
    title: str | None = None,
    percentile: float = 99.0,
    dpi: int = 150,
) -> Any:
    """Plot total mass density with a signed scale when negatives occur.

    When ``data_positions`` (the evaluation particles behind this slice) is
    given, the color range is derived only from cells supported by at least
    ``min_data_count`` particles, and unsupported cells are grayed out:
    outside the data support the learned Laplacian is unconstrained
    extrapolation, and its speckle otherwise dominates the color scale and
    hides the physical inner structure.

    ``smooth_cells`` applies a NaN-aware Gaussian smoothing to the slice
    before rendering. Point constraints (probe spacing of a few kpc) cannot
    pin the Laplacian between samples, so the raw field carries
    constraint-free curvature ripples below the probe spacing; smoothing to
    roughly the constraint resolution renders the constrained density
    instead. Set to 0 or None to show the raw field.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    from scipy.ndimage import gaussian_filter

    x, y, model = _slice_arrays(x, y, model_density, mask)
    if smooth_cells is not None and smooth_cells > 0:
        valid = ~np.ma.getmaskarray(model) & np.isfinite(
            np.ma.filled(model, np.nan)
        )
        filled = np.ma.filled(model, 0.0) * valid
        denom = gaussian_filter(valid.astype(float), float(smooth_cells))
        safe = denom > 1.0e-12
        smoothed = np.where(
            safe,
            gaussian_filter(filled, float(smooth_cells))
            / np.where(safe, denom, 1.0),
            np.nan,
        )
        model = np.ma.masked_invalid(smoothed)
    finite = _finite_slice_values(model)

    data_count: np.ndarray | None = None
    trusted: np.ndarray | None = None
    if data_positions is not None:
        positions = np.asarray(data_positions, dtype=np.float64)
        if positions.ndim == 2 and positions.shape == (positions.shape[0], 2):
            data_count, _, _ = np.histogram2d(
                positions[:, 0],
                positions[:, 1],
                bins=[_cell_edges(x), _cell_edges(y)],
            )
            data_count = data_count.T  # (n_y, n_x) to match the slice shape
            if data_count.any():
                trusted = data_count >= float(min_data_count)
            else:
                data_count = None

    def _range_values() -> np.ndarray:
        if trusted is not None and bool(trusted.any()):
            values = np.ma.filled(model, np.nan)[trusted]
            values = values[np.isfinite(values)]
            if values.size:
                return values
        return finite

    range_values = _range_values()
    if np.all(range_values > 0) and np.all(finite > 0):
        vmin, vmax = np.percentile(range_values, [100.0 - percentile, percentile])
        norm = colors.LogNorm(
            vmin=max(float(vmin), np.finfo(float).tiny),
            vmax=max(float(vmax), float(vmin) * (1.0 + 1.0e-6)),
        )
        cmap = "viridis"
    else:
        absolute = np.abs(range_values)
        limit = max(float(np.percentile(absolute, percentile)), 1.0e-12)
        nonzero = absolute[absolute > 0]
        linthresh = max(
            min(
                float(np.percentile(nonzero, 10.0)) if nonzero.size else limit,
                limit * 0.1,
            ),
            limit * 1.0e-6,
        )
        norm = colors.SymLogNorm(
            linthresh=linthresh,
            vmin=-limit,
            vmax=limit,
        )
        cmap = "coolwarm"

    fig, ax = plt.subplots(constrained_layout=True, dpi=dpi)
    mappable = ax.pcolormesh(
        x,
        y,
        model,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    if trusted is not None and not bool(trusted.all()):
        untrusted = ~trusted
        ax.contourf(
            x,
            y,
            np.where(untrusted, 1.0, 0.0),
            levels=[0.5, 1.5],
            colors=["0.82"],
            alpha=0.65,
        )
    if data_positions is not None:
        from matplotlib.patches import Circle

        flat = np.asarray(data_positions, dtype=np.float64)
        radii = np.sqrt(flat[:, 0] ** 2 + flat[:, 1] ** 2)
        for fraction in (68.0, 95.0, 99.5):
            support_radius = float(np.percentile(radii, fraction))
            ax.add_patch(
                Circle(
                    (0.0, 0.0),
                    support_radius,
                    fill=False,
                    linestyle="--",
                    linewidth=0.8,
                    edgecolor="0.30",
                    alpha=0.85,
                )
            )
            ax.text(
                0.01 * (x[-1] - x[0]),
                support_radius,
                f"{fraction:g}%",
                fontsize=6,
                color="0.30",
                va="bottom",
            )
    if trusted is not None and not bool(trusted.all()):
        ax.text(
            0.01,
            0.01,
            f"gray: < {min_data_count} data particles (extrapolated);"
            " dashed: data radii",
            transform=ax.transAxes,
            fontsize=7,
            color="0.30",
            va="bottom",
        )
    if smooth_cells is not None and smooth_cells > 0:
        cell_size = float(x[1] - x[0])
        sigma = float(smooth_cells) * cell_size
        formatted = format_display_unit(length_unit)
        unit_note = f" {formatted}" if formatted else ""
        ax.text(
            0.01,
            0.045 if trusted is not None and not bool(trusted.all()) else 0.01,
            f"smoothed: Gaussian sigma = {sigma:.2g}{unit_note}"
            f" ({smooth_cells:g} cells, ~constraint resolution)",
            transform=ax.transAxes,
            fontsize=7,
            color="0.30",
            va="bottom",
        )
    if truth_density is not None:
        _, _, truth = _slice_arrays(x, y, truth_density, mask)
        positive_truth = _finite_slice_values(truth)
        positive_truth = positive_truth[positive_truth > 0]
        if truth_contour_levels is None:
            levels = (
                np.unique(np.percentile(positive_truth, [20.0, 50.0, 80.0]))
                if positive_truth.size
                else np.asarray([])
            )
        else:
            levels = np.asarray(truth_contour_levels, dtype=np.float64)
        if levels.size:
            ax.contour(x, y, truth, levels=levels, colors="#e1812c", linewidths=0.7)
    fig.colorbar(mappable, ax=ax, label=density_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return fig
