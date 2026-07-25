"""Reusable scientific metrics for simulator-truth field validation."""

from __future__ import annotations

from typing import Any

import numpy as np


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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: float | np.ndarray,
) -> np.ndarray:
    """Return weighted empirical quantiles for finite positive-weight rows."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have equal one-dimensional shapes.")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[finite]
    weights = weights[finite]
    if values.size == 0:
        return np.full(quantiles.shape, np.nan, dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(
        np.clip(quantiles, 0.0, 1.0),
        cumulative,
        values,
        left=values[0],
        right=values[-1],
    )


def truth_cbe_score_metrics(
    eta: np.ndarray,
    score: np.ndarray,
    acceleration: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    normalization_floor_fraction: float = 1.0e-6,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Test a physical 6D DF score against simulator acceleration via the CBE.

    ``acceleration`` is the physical acceleration ``a = -grad(Phi)``.  The
    stationary truth-field residual is therefore

    ``r = v dot grad_x(log f) + a dot grad_v(log f)``.

    This checks the score projected onto the Hamiltonian flow, not all six
    score components independently.
    """
    eta = np.asarray(eta, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError("eta must have shape (N, 6).")
    if score.shape != eta.shape:
        raise ValueError(f"Expected score shape {eta.shape}, got {score.shape}.")
    if acceleration.shape != (eta.shape[0], 3):
        raise ValueError(
            f"Expected acceleration shape ({eta.shape[0]}, 3), "
            f"got {acceleration.shape}."
        )
    if normalization_floor_fraction < 0:
        raise ValueError("normalization_floor_fraction must be non-negative.")
    if weights is None:
        weights = np.ones(eta.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (eta.shape[0],):
            raise ValueError(f"Expected weights shape ({eta.shape[0]},).")

    finite = (
        np.all(np.isfinite(eta), axis=1)
        & np.all(np.isfinite(score), axis=1)
        & np.all(np.isfinite(acceleration), axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not np.any(finite):
        raise ValueError("No finite positive-weight CBE score rows are available.")

    eta_valid = eta[finite]
    score_valid = score[finite]
    acceleration_valid = acceleration[finite]
    weight_valid = weights[finite]
    weight_valid = weight_valid / np.sum(weight_valid)

    transport_term = np.sum(
        eta_valid[:, 3:] * score_valid[:, :3],
        axis=1,
    )
    acceleration_term = np.sum(
        acceleration_valid * score_valid[:, 3:],
        axis=1,
    )
    residual = transport_term + acceleration_term
    term_amplitude = np.abs(transport_term) + np.abs(acceleration_term)
    typical_amplitude = float(
        _weighted_quantile(term_amplitude, weight_valid, 0.5)
    )
    normalization_epsilon = max(
        float(normalization_floor_fraction) * typical_amplitude,
        np.finfo(np.float64).tiny,
    )
    normalized_residual = np.abs(residual) / (
        term_amplitude + normalization_epsilon
    )

    comparison = -acceleration_term
    x_mean = float(np.sum(weight_valid * transport_term))
    y_mean = float(np.sum(weight_valid * comparison))
    x_centered = transport_term - x_mean
    y_centered = comparison - y_mean
    x_var = float(np.sum(weight_valid * x_centered**2))
    y_var = float(np.sum(weight_valid * y_centered**2))
    covariance = float(np.sum(weight_valid * x_centered * y_centered))
    pearson = (
        covariance / np.sqrt(x_var * y_var)
        if x_var > 0 and y_var > 0
        else float("nan")
    )
    slope_denom = float(np.sum(weight_valid * transport_term**2))
    slope = (
        float(np.sum(weight_valid * transport_term * comparison)) / slope_denom
        if slope_denom > 0
        else float("nan")
    )

    abs_residual = np.abs(residual)
    residual_quantiles = _weighted_quantile(
        abs_residual,
        weight_valid,
        np.array([0.5, 0.9, 0.99]),
    )
    normalized_quantiles = _weighted_quantile(
        normalized_residual,
        weight_valid,
        np.array([0.5, 0.9, 0.99]),
    )
    amplitude_quantiles = _weighted_quantile(
        term_amplitude,
        weight_valid,
        np.array([0.5, 0.9, 0.99]),
    )
    residual_mean = float(np.sum(weight_valid * residual))
    metrics: dict[str, Any] = {
        "n_total": int(eta.shape[0]),
        "n_finite": int(np.count_nonzero(finite)),
        "finite_point_fraction": float(np.mean(finite)),
        "sign_convention": (
            "r_true = v dot score_x + acceleration_true dot score_v"
        ),
        "normalization_floor_fraction": float(normalization_floor_fraction),
        "normalization_epsilon": normalization_epsilon,
        "transport_vs_negative_acceleration": {
            "slope_through_origin": slope,
            "pearson_r": float(pearson),
        },
        "term_amplitude": {
            "median": float(amplitude_quantiles[0]),
            "p90": float(amplitude_quantiles[1]),
            "p99": float(amplitude_quantiles[2]),
        },
        "residual": {
            "weighted_mean": residual_mean,
            "weighted_std": float(
                np.sqrt(np.sum(weight_valid * (residual - residual_mean) ** 2))
            ),
            "median_abs": float(residual_quantiles[0]),
            "p90_abs": float(residual_quantiles[1]),
            "p99_abs": float(residual_quantiles[2]),
        },
        "normalized_residual": {
            "median": float(normalized_quantiles[0]),
            "p90": float(normalized_quantiles[1]),
            "p99": float(normalized_quantiles[2]),
            "fraction_lt_0p1": float(
                np.sum(weight_valid[normalized_residual < 0.1])
            ),
            "fraction_lt_0p2": float(
                np.sum(weight_valid[normalized_residual < 0.2])
            ),
        },
    }
    diagnostics = {
        "finite_mask": finite,
        "weights": weight_valid,
        "transport_term": transport_term,
        "acceleration_term": acceleration_term,
        "residual": residual,
        "term_amplitude": term_amplitude,
        "normalized_residual": normalized_residual,
    }
    return metrics, diagnostics


def truth_cbe_radial_profile(
    positions: np.ndarray,
    residual: np.ndarray,
    normalized_residual: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    n_bins: int = 12,
) -> dict[str, Any]:
    """Summarize truth-field CBE residuals in logarithmic radial bins."""
    positions = np.asarray(positions, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    normalized_residual = np.asarray(normalized_residual, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if residual.shape != (positions.shape[0],):
        raise ValueError(f"Expected residual shape ({positions.shape[0]},).")
    if normalized_residual.shape != residual.shape:
        raise ValueError("normalized_residual must match residual shape.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if weights is None:
        weights = np.ones(positions.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != residual.shape:
            raise ValueError(f"Expected weights shape {residual.shape}.")

    finite = (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(residual)
        & np.isfinite(normalized_residual)
        & np.isfinite(weights)
        & (weights > 0)
    )
    radius = np.linalg.norm(positions[finite], axis=1)
    residual = np.abs(residual[finite])
    normalized_residual = normalized_residual[finite]
    weights = weights[finite]
    positive = radius > 0
    radius = radius[positive]
    residual = residual[positive]
    normalized_residual = normalized_residual[positive]
    weights = weights[positive]
    if radius.size == 0:
        raise ValueError("No positive finite radii are available.")
    r_min = float(np.min(radius))
    r_max = float(np.max(radius))
    if r_max == r_min:
        r_min *= 0.5
        r_max *= 1.5
    edges = np.geomspace(r_min, r_max, int(n_bins) + 1)

    rows: list[dict[str, float | int]] = []
    for index in range(int(n_bins)):
        include_upper = index == int(n_bins) - 1
        mask = (radius >= edges[index]) & (
            radius <= edges[index + 1]
            if include_upper
            else radius < edges[index + 1]
        )
        count = int(np.count_nonzero(mask))
        residual_q = _weighted_quantile(
            residual[mask],
            weights[mask],
            np.array([0.5, 0.9]),
        )
        normalized_q = _weighted_quantile(
            normalized_residual[mask],
            weights[mask],
            np.array([0.5, 0.9]),
        )
        rows.append(
            {
                "r_left": float(edges[index]),
                "r_right": float(edges[index + 1]),
                "n": count,
                "residual_median_abs": float(residual_q[0]),
                "residual_p90_abs": float(residual_q[1]),
                "normalized_median": float(normalized_q[0]),
                "normalized_p90": float(normalized_q[1]),
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
