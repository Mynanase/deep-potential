"""Reusable scientific metrics for simulator-truth field validation."""

from __future__ import annotations

from typing import Any

import numpy as np


SPHERICAL_PHASE_SPACE_COLUMNS = (
    "r",
    "theta",
    "phi",
    "v_r",
    "v_theta",
    "v_phi",
)


def cartesian_to_spherical_phase_space(eta: np.ndarray) -> np.ndarray:
    """Convert ``[x,y,z,vx,vy,vz]`` rows to spherical phase-space rows.

    ``theta`` is the colatitude in ``[0, pi]`` and ``phi`` is the azimuth in
    ``[-pi, pi]``.  The velocity basis is orthonormal, so transforming samples
    and then histogramming them performs the requested velocity marginalization
    without an additional Jacobian.
    """
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta contains NaN or Inf values.")

    x, y, z, vx, vy, vz = eta.T
    cylindrical_radius = np.hypot(x, y)
    radius = np.sqrt(cylindrical_radius**2 + z**2)
    theta = np.arctan2(cylindrical_radius, z)
    phi = np.arctan2(y, x)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v_r = (
        vx * sin_theta * cos_phi
        + vy * sin_theta * sin_phi
        + vz * cos_theta
    )
    v_theta = (
        vx * cos_theta * cos_phi
        + vy * cos_theta * sin_phi
        - vz * sin_theta
    )
    v_phi = -vx * sin_phi + vy * cos_phi
    return np.column_stack(
        [radius, theta, phi, v_r, v_theta, v_phi]
    )


def _validate_edges(edges: np.ndarray, *, name: str) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return edges


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return float(
        np.interp(
            float(np.clip(quantile, 0.0, 1.0)),
            cumulative,
            values,
            left=values[0],
            right=values[-1],
        )
    )


def _bin_mask(
    values: np.ndarray,
    edges: np.ndarray,
    index: int,
) -> np.ndarray:
    if index == edges.size - 2:
        return (values >= edges[index]) & (values <= edges[index + 1])
    return (values >= edges[index]) & (values < edges[index + 1])


def _normalized_histogram(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        finite &= np.isfinite(weights) & (weights > 0)
        weights = weights[finite]
    values = values[finite]
    if values.size:
        # The robust display range is finite.  Accumulate more extreme values
        # in the edge bins so each conditional histogram still integrates to 1.
        values = np.clip(
            values,
            edges[0],
            np.nextafter(edges[-1], edges[0]),
        )
    count, _ = np.histogram(values, bins=edges, weights=weights)
    total = float(np.sum(count))
    probability = (
        count.astype(np.float64) / total
        if total > 0
        else np.full(edges.size - 1, np.nan, dtype=np.float64)
    )
    density = probability / np.diff(edges)
    return density, probability


def _weighted_mean_std(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    if weights is None:
        return float(np.mean(values)), float(np.std(values))
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan"), float("nan")
    mean = float(np.sum(values * weights) / total)
    variance = float(np.sum(weights * (values - mean) ** 2) / total)
    return mean, float(np.sqrt(max(variance, 0.0)))


def conditional_velocity_diagnostics(
    reference_eta: np.ndarray,
    model_eta: np.ndarray,
    *,
    reference_weights: np.ndarray | None = None,
    conditioning_edges: dict[str, np.ndarray],
    n_velocity_bins: int = 64,
    velocity_percentile_range: tuple[float, float] = (0.5, 99.5),
) -> dict[str, Any]:
    """Compare data/model spherical-velocity marginals in spatial bins.

    For a given conditioning coordinate (r, theta, or phi), the other two
    spatial coordinates and the other two velocity coordinates are marginalized
    by selecting Monte Carlo samples and drawing a one-dimensional histogram.
    """
    reference_eta = np.asarray(reference_eta, dtype=np.float64)
    model_eta = np.asarray(model_eta, dtype=np.float64)
    if model_eta.ndim == 2:
        model_eta = model_eta[None, ...]
    if model_eta.ndim != 3 or model_eta.shape[2] != 6:
        raise ValueError("model_eta must have shape (n_models, n_samples, 6).")
    if n_velocity_bins < 4:
        raise ValueError("n_velocity_bins must be at least 4.")
    if reference_weights is None:
        reference_weights = np.ones(reference_eta.shape[0], dtype=np.float64)
    else:
        reference_weights = np.asarray(reference_weights, dtype=np.float64)
    if reference_weights.shape != (reference_eta.shape[0],):
        raise ValueError(
            f"Expected reference_weights shape ({reference_eta.shape[0]},)."
        )
    if (
        not np.all(np.isfinite(reference_weights))
        or np.any(reference_weights <= 0)
    ):
        raise ValueError("reference_weights must be finite and positive.")

    requested = ("r", "theta", "phi")
    missing = [name for name in requested if name not in conditioning_edges]
    if missing:
        raise ValueError(f"Missing conditioning edges for {missing}.")
    resolved_edges = {
        name: _validate_edges(conditioning_edges[name], name=f"{name}_edges")
        for name in requested
    }

    reference_spherical = cartesian_to_spherical_phase_space(reference_eta)
    model_spherical = np.stack(
        [cartesian_to_spherical_phase_space(rows) for rows in model_eta],
        axis=0,
    )
    n_models = model_spherical.shape[0]
    lower_pct, upper_pct = velocity_percentile_range
    if not 0 <= lower_pct < upper_pct <= 100:
        raise ValueError("velocity_percentile_range must lie within [0, 100].")

    velocity_edges = np.empty((3, int(n_velocity_bins) + 1), dtype=np.float64)
    for velocity_index in range(3):
        ref_values = reference_spherical[:, 3 + velocity_index]
        ref_lower = _weighted_quantile(
            ref_values, reference_weights, lower_pct / 100.0
        )
        ref_upper = _weighted_quantile(
            ref_values, reference_weights, upper_pct / 100.0
        )
        model_values = model_spherical[:, :, 3 + velocity_index]
        model_lower, model_upper = np.percentile(
            model_values[np.isfinite(model_values)],
            [lower_pct, upper_pct],
        )
        lower = float(min(ref_lower, model_lower))
        upper = float(max(ref_upper, model_upper))
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("No finite spherical velocities are available.")
        if upper <= lower:
            padding = max(abs(lower) * 1.0e-3, 1.0)
            lower -= padding
            upper += padding
        velocity_edges[velocity_index] = np.linspace(
            lower,
            upper,
            int(n_velocity_bins) + 1,
        )

    result: dict[str, Any] = {"velocity_edges": velocity_edges}
    total_reference_weight = float(np.sum(reference_weights))
    for coordinate_index, name in enumerate(requested):
        edges = resolved_edges[name]
        n_condition_bins = edges.size - 1
        reference_hist = np.full(
            (n_condition_bins, 3, int(n_velocity_bins)),
            np.nan,
            dtype=np.float64,
        )
        model_hist = np.full(
            (n_models, n_condition_bins, 3, int(n_velocity_bins)),
            np.nan,
            dtype=np.float64,
        )
        reference_count = np.zeros(n_condition_bins, dtype=np.int64)
        reference_effective_count = np.zeros(
            n_condition_bins,
            dtype=np.float64,
        )
        model_count = np.zeros((n_models, n_condition_bins), dtype=np.int64)
        reference_probability = np.zeros(n_condition_bins, dtype=np.float64)
        model_probability = np.zeros(
            (n_models, n_condition_bins),
            dtype=np.float64,
        )
        wasserstein = np.full(
            (n_models, n_condition_bins, 3),
            np.nan,
            dtype=np.float64,
        )
        ks_histogram = np.full_like(wasserstein, np.nan)
        js_divergence = np.full_like(wasserstein, np.nan)
        mean_bias = np.full_like(wasserstein, np.nan)
        dispersion_bias = np.full_like(wasserstein, np.nan)

        reference_coordinate = reference_spherical[:, coordinate_index]
        for bin_index in range(n_condition_bins):
            ref_mask = _bin_mask(reference_coordinate, edges, bin_index)
            ref_weights = reference_weights[ref_mask]
            reference_count[bin_index] = int(np.count_nonzero(ref_mask))
            weight_sum = float(np.sum(ref_weights))
            reference_probability[bin_index] = (
                weight_sum / total_reference_weight
                if total_reference_weight > 0
                else np.nan
            )
            reference_effective_count[bin_index] = (
                weight_sum**2 / float(np.sum(ref_weights**2))
                if ref_weights.size and np.sum(ref_weights**2) > 0
                else 0.0
            )

            for velocity_index in range(3):
                velocity_bin_edges = velocity_edges[velocity_index]
                ref_values = reference_spherical[
                    ref_mask,
                    3 + velocity_index,
                ]
                ref_density, ref_probability = _normalized_histogram(
                    ref_values,
                    velocity_bin_edges,
                    weights=ref_weights,
                )
                reference_hist[bin_index, velocity_index] = ref_density
                ref_mean, ref_std = _weighted_mean_std(
                    ref_values,
                    ref_weights,
                )

                for model_index in range(n_models):
                    model_coordinate = model_spherical[
                        model_index,
                        :,
                        coordinate_index,
                    ]
                    model_mask = _bin_mask(
                        model_coordinate,
                        edges,
                        bin_index,
                    )
                    if velocity_index == 0:
                        model_count[model_index, bin_index] = int(
                            np.count_nonzero(model_mask)
                        )
                        model_probability[model_index, bin_index] = float(
                            np.mean(model_mask)
                        )
                    model_values = model_spherical[
                        model_index,
                        model_mask,
                        3 + velocity_index,
                    ]
                    model_density, model_probability_bins = (
                        _normalized_histogram(
                            model_values,
                            velocity_bin_edges,
                        )
                    )
                    model_hist[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = model_density
                    if (
                        ref_values.size == 0
                        or model_values.size == 0
                        or not np.all(np.isfinite(ref_probability))
                        or not np.all(np.isfinite(model_probability_bins))
                    ):
                        continue

                    from scipy.stats import wasserstein_distance

                    wasserstein[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = wasserstein_distance(
                        ref_values,
                        model_values,
                        u_weights=ref_weights,
                    )
                    ks_histogram[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = float(
                        np.max(
                            np.abs(
                                np.cumsum(ref_probability)
                                - np.cumsum(model_probability_bins)
                            )
                        )
                    )
                    midpoint = 0.5 * (
                        ref_probability + model_probability_bins
                    )
                    ref_positive = ref_probability > 0
                    model_positive = model_probability_bins > 0
                    js_divergence[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = 0.5 * float(
                        np.sum(
                            ref_probability[ref_positive]
                            * np.log(
                                ref_probability[ref_positive]
                                / midpoint[ref_positive]
                            )
                        )
                        + np.sum(
                            model_probability_bins[model_positive]
                            * np.log(
                                model_probability_bins[model_positive]
                                / midpoint[model_positive]
                            )
                        )
                    )
                    model_mean, model_std = _weighted_mean_std(model_values)
                    mean_bias[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = model_mean - ref_mean
                    dispersion_bias[
                        model_index,
                        bin_index,
                        velocity_index,
                    ] = model_std - ref_std

        result[name] = {
            "edges": edges,
            "reference_hist": reference_hist,
            "model_hist": model_hist,
            "reference_count": reference_count,
            "reference_effective_count": reference_effective_count,
            "model_count": model_count,
            "reference_probability": reference_probability,
            "model_probability": model_probability,
            "wasserstein": wasserstein,
            "ks_histogram": ks_histogram,
            "js_divergence": js_divergence,
            "mean_bias": mean_bias,
            "dispersion_bias": dispersion_bias,
        }
    return result


def cylindrical_rz_density_by_phi(
    reference_positions: np.ndarray,
    model_positions: np.ndarray,
    *,
    reference_weights: np.ndarray | None,
    phi_edges: np.ndarray,
    cylindrical_radius_edges: np.ndarray,
    z_edges: np.ndarray,
    min_cell_count: int = 5,
) -> dict[str, np.ndarray]:
    """Estimate normalized 3D density in cylindrical ``(phi, R, z)`` cells."""
    reference_positions = np.asarray(reference_positions, dtype=np.float64)
    model_positions = np.asarray(model_positions, dtype=np.float64)
    if reference_positions.ndim != 2 or reference_positions.shape[1] != 3:
        raise ValueError("reference_positions must have shape (N, 3).")
    if model_positions.ndim == 2:
        model_positions = model_positions[None, ...]
    if model_positions.ndim != 3 or model_positions.shape[2] != 3:
        raise ValueError(
            "model_positions must have shape (n_models, n_samples, 3)."
        )
    phi_edges = _validate_edges(phi_edges, name="phi_edges")
    cylindrical_radius_edges = _validate_edges(
        cylindrical_radius_edges,
        name="cylindrical_radius_edges",
    )
    z_edges = _validate_edges(z_edges, name="z_edges")
    if cylindrical_radius_edges[0] < 0:
        raise ValueError("cylindrical_radius_edges must be non-negative.")
    if min_cell_count < 1:
        raise ValueError("min_cell_count must be positive.")
    if reference_weights is None:
        reference_weights = np.ones(
            reference_positions.shape[0],
            dtype=np.float64,
        )
    else:
        reference_weights = np.asarray(reference_weights, dtype=np.float64)
    if reference_weights.shape != (reference_positions.shape[0],):
        raise ValueError(
            f"Expected reference_weights shape ({reference_positions.shape[0]},)."
        )
    if (
        not np.all(np.isfinite(reference_weights))
        or np.any(reference_weights <= 0)
    ):
        raise ValueError("reference_weights must be finite and positive.")

    def coordinates(positions: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.arctan2(positions[:, 1], positions[:, 0]),
                np.hypot(positions[:, 0], positions[:, 1]),
                positions[:, 2],
            ]
        )

    bins = [phi_edges, cylindrical_radius_edges, z_edges]
    reference_coordinates = coordinates(reference_positions)
    reference_weight_grid, _ = np.histogramdd(
        reference_coordinates,
        bins=bins,
        weights=reference_weights,
    )
    reference_count, _ = np.histogramdd(reference_coordinates, bins=bins)

    delta_phi = np.diff(phi_edges)[:, None, None]
    annular_area = (
        0.5
        * (
            cylindrical_radius_edges[1:] ** 2
            - cylindrical_radius_edges[:-1] ** 2
        )[None, :, None]
    )
    delta_z = np.diff(z_edges)[None, None, :]
    cell_volume = delta_phi * annular_area * delta_z
    total_reference_weight = float(np.sum(reference_weights))
    reference_density = (
        reference_weight_grid / total_reference_weight / cell_volume
    )

    model_count = np.empty(
        (model_positions.shape[0],) + reference_count.shape,
        dtype=np.float64,
    )
    model_density = np.empty_like(model_count)
    for model_index, positions in enumerate(model_positions):
        counts, _ = np.histogramdd(coordinates(positions), bins=bins)
        model_count[model_index] = counts
        model_density[model_index] = counts / positions.shape[0] / cell_volume

    model_median_density = np.median(model_density, axis=0)
    log10_rmse_by_model_phi = np.full(
        (model_positions.shape[0], phi_edges.size - 1),
        np.nan,
        dtype=np.float64,
    )
    for model_index in range(model_positions.shape[0]):
        for phi_index in range(phi_edges.size - 1):
            valid = (
                (reference_count[phi_index] >= int(min_cell_count))
                & (model_count[model_index, phi_index] >= int(min_cell_count))
                & (reference_density[phi_index] > 0)
                & (model_density[model_index, phi_index] > 0)
            )
            if np.any(valid):
                log_ratio = np.log10(
                    model_density[model_index, phi_index][valid]
                    / reference_density[phi_index][valid]
                )
                log10_rmse_by_model_phi[model_index, phi_index] = float(
                    np.sqrt(np.mean(log_ratio**2))
                )

    return {
        "phi_edges": phi_edges,
        "cylindrical_radius_edges": cylindrical_radius_edges,
        "z_edges": z_edges,
        "cell_volume": cell_volume,
        "reference_density": reference_density,
        "model_density": model_density,
        "model_median_density": model_median_density,
        "reference_count": reference_count.astype(np.int64),
        "model_count": model_count.astype(np.int64),
        "log10_rmse_by_model_phi": log10_rmse_by_model_phi,
        "min_cell_count": np.asarray(min_cell_count, dtype=np.int64),
    }


def binned_potential_truth_by_phi(
    positions: np.ndarray,
    potential: np.ndarray,
    *,
    phi_edges: np.ndarray,
    cylindrical_radius_edges: np.ndarray,
    z_edges: np.ndarray,
    min_cell_count: int = 3,
) -> dict[str, np.ndarray]:
    """Bin simulator potential truth into cylindrical ``(phi, R, z)`` cells.

    Auriga provides potential values at particle positions rather than on a
    regular volume grid.  This helper records the per-cell median and leaves
    under-populated cells as NaN, so downstream figures do not interpolate
    unsupported simulator truth.
    """
    from scipy.stats import binned_statistic_dd

    positions = np.asarray(positions, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if potential.shape != (positions.shape[0],):
        raise ValueError(
            f"Expected potential shape ({positions.shape[0]},), "
            f"got {potential.shape}."
        )
    if min_cell_count < 1:
        raise ValueError("min_cell_count must be positive.")
    phi_edges = _validate_edges(phi_edges, name="phi_edges")
    cylindrical_radius_edges = _validate_edges(
        cylindrical_radius_edges,
        name="cylindrical_radius_edges",
    )
    z_edges = _validate_edges(z_edges, name="z_edges")
    if cylindrical_radius_edges[0] < 0:
        raise ValueError("cylindrical_radius_edges must be non-negative.")

    finite = np.all(np.isfinite(positions), axis=1) & np.isfinite(potential)
    if not np.any(finite):
        raise ValueError("No finite position/potential pairs are available.")
    positions = positions[finite]
    potential = potential[finite]
    coordinates = np.column_stack(
        [
            np.arctan2(positions[:, 1], positions[:, 0]),
            np.hypot(positions[:, 0], positions[:, 1]),
            positions[:, 2],
        ]
    )
    bins = [phi_edges, cylindrical_radius_edges, z_edges]
    median = binned_statistic_dd(
        coordinates,
        potential,
        statistic="median",
        bins=bins,
    ).statistic
    count, _ = np.histogramdd(coordinates, bins=bins)
    median[count < int(min_cell_count)] = np.nan
    return {
        "phi_edges": phi_edges,
        "cylindrical_radius_edges": cylindrical_radius_edges,
        "z_edges": z_edges,
        "truth_median": median,
        "count": count.astype(np.int64),
        "min_cell_count": np.asarray(min_cell_count, dtype=np.int64),
    }


def _finite_pair(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"Predicted/truth shapes differ: {predicted.shape} vs {truth.shape}."
        )
    finite = np.isfinite(predicted) & np.isfinite(truth)
    if predicted.ndim > 1:
        finite = np.all(finite, axis=tuple(range(1, predicted.ndim)))
    if not np.any(finite):
        raise ValueError("No finite predicted/truth pairs are available.")
    return predicted[finite], truth[finite]


def potential_error_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Compare potentials after fitting the physically irrelevant offset."""
    predicted, truth = _finite_pair(predicted, truth)
    if predicted.ndim != 1:
        raise ValueError("Potential arrays must be one-dimensional.")

    offset = float(np.median(truth - predicted))
    aligned = predicted + offset
    error = aligned - truth
    abs_error = np.abs(error)
    dynamic_range = float(np.percentile(truth, 95) - np.percentile(truth, 5))
    rmse = float(np.sqrt(np.mean(error**2)))
    correlation = (
        float(np.corrcoef(aligned, truth)[0, 1])
        if predicted.size > 1
        and np.std(aligned) > 0
        and np.std(truth) > 0
        else float("nan")
    )
    metrics: dict[str, float | int] = {
        "n": int(predicted.size),
        "fitted_additive_offset": offset,
        "mae": float(np.mean(abs_error)),
        "rmse": rmse,
        "p50_abs_error": float(np.percentile(abs_error, 50)),
        "p90_abs_error": float(np.percentile(abs_error, 90)),
        "truth_p95_minus_p05": dynamic_range,
        "normalized_rmse": (
            rmse / dynamic_range if dynamic_range > 0 else float("nan")
        ),
        "pearson_r": correlation,
    }
    return metrics, aligned


def acceleration_error_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    magnitude_floor: float = 1.0e-12,
) -> dict[str, float | int]:
    """Compare three-dimensional accelerations with vector-aware metrics."""
    predicted, truth = _finite_pair(predicted, truth)
    if predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("Acceleration arrays must have shape (N, 3).")

    error = predicted - truth
    truth_magnitude = np.linalg.norm(truth, axis=1)
    predicted_magnitude = np.linalg.norm(predicted, axis=1)
    error_magnitude = np.linalg.norm(error, axis=1)
    valid_relative = truth_magnitude > float(magnitude_floor)
    relative_error = (
        error_magnitude[valid_relative] / truth_magnitude[valid_relative]
    )
    cosine_valid = (
        valid_relative & (predicted_magnitude > float(magnitude_floor))
    )
    cosine = np.sum(
        predicted[cosine_valid] * truth[cosine_valid],
        axis=1,
    ) / (
        predicted_magnitude[cosine_valid] * truth_magnitude[cosine_valid]
    )
    denominator = float(np.sum(truth**2))
    vector_relative_l2 = (
        float(np.sqrt(np.sum(error**2) / denominator))
        if denominator > 0
        else float("nan")
    )
    return {
        "n": int(predicted.shape[0]),
        "n_relative": int(relative_error.size),
        "vector_relative_l2": vector_relative_l2,
        "mae_vector": float(np.mean(error_magnitude)),
        "rmse_vector": float(np.sqrt(np.mean(error_magnitude**2))),
        "median_relative_error": (
            float(np.median(relative_error))
            if relative_error.size
            else float("nan")
        ),
        "p90_relative_error": (
            float(np.percentile(relative_error, 90))
            if relative_error.size
            else float("nan")
        ),
        "median_cosine_similarity": (
            float(np.median(cosine)) if cosine.size else float("nan")
        ),
        "p10_cosine_similarity": (
            float(np.percentile(cosine, 10)) if cosine.size else float("nan")
        ),
    }


def radial_acceleration_profile(
    positions: np.ndarray,
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    n_bins: int = 12,
    r_min: float | None = None,
    r_max: float | None = None,
    magnitude_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Summarize vector relative error in logarithmic radial bins."""
    positions = np.asarray(positions, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if predicted.shape != positions.shape or truth.shape != positions.shape:
        raise ValueError("positions, predicted, and truth must share shape (N, 3).")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    finite = (
        np.all(np.isfinite(positions), axis=1)
        & np.all(np.isfinite(predicted), axis=1)
        & np.all(np.isfinite(truth), axis=1)
    )
    radius = np.linalg.norm(positions[finite], axis=1)
    predicted = predicted[finite]
    truth = truth[finite]
    truth_magnitude = np.linalg.norm(truth, axis=1)
    relative_error = np.linalg.norm(predicted - truth, axis=1) / np.maximum(
        truth_magnitude,
        float(magnitude_floor),
    )

    positive = radius > 0
    radius = radius[positive]
    relative_error = relative_error[positive]
    if radius.size == 0:
        raise ValueError("No positive finite radii are available.")
    lower = float(r_min) if r_min is not None else float(np.min(radius))
    upper = float(r_max) if r_max is not None else float(np.max(radius))
    if lower <= 0 or upper <= lower:
        raise ValueError(f"Invalid radial range [{lower}, {upper}].")
    edges = np.geomspace(lower, upper, int(n_bins) + 1)

    rows: list[dict[str, float | int]] = []
    for index in range(int(n_bins)):
        include_upper = index == int(n_bins) - 1
        mask = (radius >= edges[index]) & (
            radius <= edges[index + 1]
            if include_upper
            else radius < edges[index + 1]
        )
        errors = relative_error[mask]
        rows.append(
            {
                "r_left": float(edges[index]),
                "r_right": float(edges[index + 1]),
                "n": int(errors.size),
                "median_relative_error": (
                    float(np.median(errors)) if errors.size else float("nan")
                ),
                "p90_relative_error": (
                    float(np.percentile(errors, 90))
                    if errors.size
                    else float("nan")
                ),
            }
        )
    return {"n_bins": int(n_bins), "bins": rows}


def spherical_density_profile(
    positions: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Estimate a normalized spherical density profile in fixed shells."""
    positions = np.asarray(positions, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be a strictly increasing 1D array.")
    if edges[0] < 0:
        raise ValueError("Radial edges must be non-negative.")
    if weights is None:
        weights = np.ones(positions.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (positions.shape[0],):
            raise ValueError(
                f"Expected weights shape ({positions.shape[0]},), "
                f"got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative.")

    finite = np.all(np.isfinite(positions), axis=1) & np.isfinite(weights)
    radius = np.linalg.norm(positions[finite], axis=1)
    shell_weight, _ = np.histogram(radius, bins=edges, weights=weights[finite])
    shell_count, _ = np.histogram(radius, bins=edges)
    shell_volume = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    total_weight = float(np.sum(shell_weight))
    density = (
        shell_weight / total_weight / shell_volume
        if total_weight > 0
        else np.full_like(shell_weight, np.nan, dtype=np.float64)
    )
    return {
        "edges": edges,
        "radius": np.sqrt(edges[:-1] * edges[1:]),
        "density": density,
        "shell_probability": (
            shell_weight / total_weight
            if total_weight > 0
            else np.full_like(shell_weight, np.nan, dtype=np.float64)
        ),
        "shell_count": shell_count,
    }


def density_profile_metrics(
    predicted_density: np.ndarray,
    reference_density: np.ndarray,
    *,
    floor: float = 1.0e-30,
) -> dict[str, float | int]:
    """Compare positive density estimates using fractional and dex errors."""
    predicted = np.asarray(predicted_density, dtype=np.float64)
    reference = np.asarray(reference_density, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.ndim != 1:
        raise ValueError("Density profiles must be 1D arrays with equal shape.")
    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > floor)
        & (reference > floor)
    )
    if not np.any(valid):
        raise ValueError("No positive finite density bins are available.")
    ratio = predicted[valid] / reference[valid]
    fractional = np.abs(ratio - 1.0)
    log_error = np.log10(ratio)
    return {
        "n_bins": int(np.count_nonzero(valid)),
        "median_fractional_error": float(np.median(fractional)),
        "p90_fractional_error": float(np.percentile(fractional, 90)),
        "log10_rmse_dex": float(np.sqrt(np.mean(log_error**2))),
        "max_abs_log10_error_dex": float(np.max(np.abs(log_error))),
    }


def score_ensemble_metrics(scores: np.ndarray) -> dict[str, Any]:
    """Measure repeatability of 6D physical scores across independent runs."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[2] != 6:
        raise ValueError("scores must have shape (n_models, n_points, 6).")
    if scores.shape[0] < 2:
        raise ValueError("At least two score models are required.")

    finite_by_point = np.all(np.isfinite(scores), axis=(0, 2))
    valid = scores[:, finite_by_point]
    if valid.shape[1] == 0:
        raise ValueError("No points have finite scores in every model.")
    median_score = np.median(valid, axis=0)
    mad = np.median(
        np.abs(valid - median_score[None, :, :]),
        axis=0,
    )
    scale = np.median(np.abs(median_score), axis=0)
    relative_mad = np.median(mad, axis=0) / np.maximum(scale, 1.0e-12)

    pair_cosines: list[float] = []
    for left in range(valid.shape[0]):
        left_norm = np.linalg.norm(valid[left], axis=1)
        for right in range(left + 1, valid.shape[0]):
            right_norm = np.linalg.norm(valid[right], axis=1)
            denom = left_norm * right_norm
            usable = denom > 1.0e-12
            if np.any(usable):
                cosine = np.sum(
                    valid[left, usable] * valid[right, usable],
                    axis=1,
                ) / denom[usable]
                pair_cosines.extend(cosine.tolist())
    return {
        "n_models": int(scores.shape[0]),
        "n_points": int(scores.shape[1]),
        "finite_point_fraction": float(np.mean(finite_by_point)),
        "relative_mad_by_dimension": relative_mad.tolist(),
        "median_relative_mad": float(np.median(relative_mad)),
        "pairwise_cosine_median": (
            float(np.median(pair_cosines)) if pair_cosines else float("nan")
        ),
        "pairwise_cosine_p10": (
            float(np.percentile(pair_cosines, 10))
            if pair_cosines
            else float("nan")
        ),
    }


def stein_score_metrics(
    eta: np.ndarray,
    score: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate first-order empirical Stein identities on held-out tracers.

    These are consistency diagnostics, not simulator truth for the unknown
    Auriga 6D score.
    """
    eta = np.asarray(eta, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if eta.shape != score.shape or eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError("eta and score must both have shape (N, 6).")
    if weights is None:
        weights = np.ones(eta.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (eta.shape[0],):
            raise ValueError(f"Expected weights shape ({eta.shape[0]},).")
    finite = (
        np.all(np.isfinite(eta), axis=1)
        & np.all(np.isfinite(score), axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    eta = eta[finite]
    score = score[finite]
    weights = weights[finite]
    if eta.shape[0] == 0:
        raise ValueError("No finite positive-weight score rows are available.")
    weights = weights / np.sum(weights)
    mean_score = np.sum(weights[:, None] * score, axis=0)
    cross = np.einsum("n,ni,nj->ij", weights, score, eta)
    residual = cross + np.eye(6)
    return {
        "n": int(eta.shape[0]),
        "mean_score_by_dimension": mean_score.tolist(),
        "mean_score_l2": float(np.linalg.norm(mean_score)),
        "cross_identity_frobenius": float(np.linalg.norm(residual)),
        "cross_identity_max_abs": float(np.max(np.abs(residual))),
    }
