"""Analytic Plummer helpers for validation and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.models.potential import (
    grad_phi_apply,
    laplacian_phi_apply,
    phi_apply,
)
from dpjax.physics.units import density_from_laplacian
from experiments.datasets.phase_space import (
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from experiments.diagnostics.evaluation import (
    acceleration_error_metrics,
    potential_error_metrics,
)
from experiments.paths import ensure_dir, resolve_path
from experiments.workflows.artifacts import load_phi


def plummer_phi(r: np.ndarray) -> np.ndarray:
    """Analytic Plummer potential: Φ(r) = -(1 + r^2)^(-1/2)."""
    return -(1.0 + r**2) ** (-0.5)


def plummer_ar(r: np.ndarray) -> np.ndarray:
    """Signed radial acceleration: a_r = -dΦ/dr = -r * (1 + r^2)^(-3/2)."""
    return -r * (1.0 + r**2) ** (-1.5)


def plummer_rho(r: np.ndarray) -> np.ndarray:
    """Analytic Plummer total mass density with ``G=M=a=1``."""
    r = np.asarray(r)
    return (3.0 / (4.0 * np.pi)) * (1.0 + r**2) ** (-2.5)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# JAX-based Plummer DF score functions (used for flow diagnostics)
# ---------------------------------------------------------------------------

def plummer_score_std_batch(
    eta_std_batch: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """Analytic ∇_{η_std} log f for the Plummer DF, vectorised over a batch.

    Parameters
    ----------
    eta_std_batch : (N, 6) standardised phase-space coordinates
    mean, std : (6,) normalizer statistics (physical → standardised)

    Returns
    -------
    (N, 6) score field in standardised coordinates
    """
    mean = jnp.asarray(mean, dtype=jnp.float32)
    std = jnp.asarray(std, dtype=jnp.float32)

    def _logf_single(eta_std_single: jnp.ndarray) -> jnp.ndarray:
        eta = eta_std_single * std + mean
        x = eta[:3]
        v = eta[3:]
        r2 = jnp.sum(x**2)
        v2 = jnp.sum(v**2)
        phi = -(1.0 + r2) ** (-0.5)
        E = 0.5 * v2 + phi
        f = jnp.clip(-E, 1.0e-12, jnp.inf) ** 3.5
        return jnp.log(f + 1.0e-30)

    return jax.vmap(jax.grad(_logf_single))(eta_std_batch)


def plummer_score_phys_batch(
    eta_phys_batch: jnp.ndarray,
) -> jnp.ndarray:
    """Analytic ∇_{η_phys} log f for the Plummer DF, vectorised over a batch.

    Parameters
    ----------
    eta_phys_batch : (N, 6) physical phase-space coordinates

    Returns
    -------
    (N, 6) score field in physical coordinates
    """

    def _logf_phys_single(eta_single: jnp.ndarray) -> jnp.ndarray:
        x = eta_single[:3]
        v = eta_single[3:]
        r2 = jnp.sum(x**2)
        v2 = jnp.sum(v**2)
        phi = -(1.0 + r2) ** (-0.5)
        E = 0.5 * v2 + phi
        A = 24.0 * jnp.sqrt(2.0) / (7.0 * jnp.pi**3)
        f = A * jnp.clip(-E, 1.0e-12, jnp.inf) ** 3.5
        return jnp.log(f + 1.0e-30)

    return jax.vmap(jax.grad(_logf_phys_single))(eta_phys_batch)


def plummer_rv_ideal_grid(
    r_lim: tuple[float, float] = (0.0, 5.0),
    v_lim: tuple[float, float] = (0.0, 1.5),
    bins: tuple[int, int] = (50, 50),
    *,
    radial_selection: tuple[float, float] = (0.0, np.inf),
    quadrature_order: int = 4,
) -> dict:
    """Compute the ideal Plummer ``(r, v)`` density and bin probabilities.

    The four angular dimensions are integrated analytically. Bin probability
    masses are evaluated with tensor-product Gauss-Legendre quadrature and are
    conditioned on ``radial_selection``. This supports both the full and
    radially cut Plummer experiments without putting analytic truth in the
    generic plotting package.

    Returns
    -------
    dict with keys: ``r``, ``v``, ``r_edges``, ``v_edges``, ``n_ideal`` and
    ``probability_mass``. ``n_ideal`` retains the historical ``(Nv, Nr)``
    orientation; ``probability_mass`` uses ``(Nr, Nv)``.
    """
    if bins[0] <= 0 or bins[1] <= 0:
        raise ValueError("bins must contain positive integers.")
    if not r_lim[0] < r_lim[1] or not v_lim[0] < v_lim[1]:
        raise ValueError("r_lim and v_lim must be strictly increasing.")
    selection_min, selection_max = map(float, radial_selection)
    if selection_min < 0.0 or not selection_min < selection_max:
        raise ValueError(
            "radial_selection must be increasing with a non-negative lower "
            "bound."
        )
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    r_edges = np.linspace(r_lim[0], r_lim[1], bins[0] + 1)
    v_edges = np.linspace(v_lim[0], v_lim[1], bins[1] + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    v = 0.5 * (v_edges[:-1] + v_edges[1:])
    rr, vv = np.meshgrid(r, v)

    psi = 1.0 / np.sqrt(1.0 + rr**2)
    E = psi - vv**2 / 2.0
    df = np.clip(E, 0.0, np.inf) ** (7.0 / 2.0)
    A = 24.0 * np.sqrt(2.0) / (7.0 * np.pi**3)
    n_ideal = A * (4.0 * np.pi) ** 2 * rr**2 * vv**2 * df

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    r_integration_min = np.maximum(r_edges[:-1], selection_min)
    r_integration_max = np.minimum(r_edges[1:], selection_max)
    r_integration_width = np.maximum(
        r_integration_max - r_integration_min,
        0.0,
    )
    r_integration_center = 0.5 * (
        r_integration_min + r_integration_max
    )
    r_half_width = 0.5 * r_integration_width
    v_half_width = 0.5 * np.diff(v_edges)
    r_nodes = (
        r_integration_center[:, None]
        + r_half_width[:, None] * nodes[None, :]
    )
    v_nodes = v[:, None] + v_half_width[:, None] * nodes[None, :]
    rr_nodes = r_nodes[:, None, :, None]
    vv_nodes = v_nodes[None, :, None, :]
    relative_energy = (
        1.0 / np.sqrt(1.0 + rr_nodes**2) - 0.5 * vv_nodes**2
    )
    probability_density = (
        A
        * (4.0 * np.pi) ** 2
        * rr_nodes**2
        * vv_nodes**2
        * np.clip(relative_energy, 0.0, np.inf) ** (7.0 / 2.0)
    )
    quadrature_weights = weights[None, None, :, None] * weights[
        None, None, None, :
    ]
    probability_mass = np.sum(
        probability_density * quadrature_weights,
        axis=(2, 3),
    )
    probability_mass *= r_half_width[:, None] * v_half_width[None, :]

    def radial_cdf(radius: float) -> float:
        if np.isposinf(radius):
            return 1.0
        return float(radius**3 / (1.0 + radius**2) ** 1.5)

    selection_probability = radial_cdf(selection_max) - radial_cdf(
        selection_min
    )
    if selection_probability <= 0.0:
        raise ValueError("radial_selection contains no Plummer probability.")
    probability_mass /= selection_probability

    return {
        "r": r,
        "v": v,
        "r_edges": r_edges,
        "v_edges": v_edges,
        "n_ideal": n_ideal,
        "probability_mass": probability_mass,
        "selection_probability": selection_probability,
    }


def _sample_plummer_positions(
    n_samples: int,
    *,
    r_min: float,
    r_max: float,
    seed: int,
) -> np.ndarray:
    """Draw positions exactly from a radially truncated Plummer sphere."""
    if n_samples <= 0:
        raise ValueError("n_eval must be positive.")

    def radial_cdf(radius: float) -> float:
        return float(radius**3 / (1.0 + radius**2) ** 1.5)

    rng = np.random.default_rng(seed)
    cdf = rng.uniform(radial_cdf(r_min), radial_cdf(r_max), size=n_samples)
    cdf_power = np.power(cdf, 2.0 / 3.0)
    radius = np.sqrt(cdf_power / np.maximum(1.0 - cdf_power, 1.0e-15))
    cos_theta = rng.uniform(-1.0, 1.0, size=n_samples)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    azimuth = rng.uniform(-np.pi, np.pi, size=n_samples)
    return np.column_stack(
        [
            radius * sin_theta * np.cos(azimuth),
            radius * sin_theta * np.sin(azimuth),
            radius * cos_theta,
        ]
    ).astype(np.float32)


def _predict_plummer_fields(
    model: Any,
    params: dict,
    positions: np.ndarray,
    mean_x: np.ndarray,
    std_x: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    positions_std = (positions - mean_x[None, :]) / std_x[None, :]
    potential_parts = []
    acceleration_parts = []
    density_parts = []
    std_x_jax = jnp.asarray(std_x)
    for start in range(0, positions_std.shape[0], batch_size):
        stop = min(start + batch_size, positions_std.shape[0])
        batch = jnp.asarray(positions_std[start:stop])
        potential_parts.append(
            np.asarray(phi_apply(model, params, batch), dtype=np.float64)
        )
        gradient_std = np.asarray(
            grad_phi_apply(model, params, batch),
            dtype=np.float64,
        )
        acceleration_parts.append(-gradient_std / std_x[None, :])
        laplacian = np.asarray(
            laplacian_phi_apply(
                model,
                params,
                batch,
                std_x=std_x_jax,
            ),
            dtype=np.float64,
        )
        density_parts.append(
            density_from_laplacian(laplacian, gravitational_constant=1.0)
        )
    return {
        "potential": np.concatenate(potential_parts),
        "acceleration": np.concatenate(acceleration_parts),
        "density": np.concatenate(density_parts),
    }


def _scalar_error_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(predicted) & np.isfinite(truth)
    if not np.any(finite):
        raise ValueError("No finite scalar truth pairs are available.")
    error = predicted[finite] - truth[finite]
    truth_scale = float(np.percentile(np.abs(truth[finite]), 95.0))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "n": int(np.count_nonzero(finite)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "normalized_rmse": rmse / truth_scale if truth_scale > 0 else float("nan"),
    }


def evaluate_plummer_truth(
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    output_dir: str | Path,
    *,
    n_eval: int = 65_536,
    batch_size: int = 4_096,
    seed: int = 0,
    r_min: float = 1.0e-3,
    r_max: float = 10.0,
    n_r: int = 256,
    slice_grid: int = 128,
    slice_rmax: float | None = None,
    artifact_prefix: str | None = None,
) -> dict[str, Any]:
    """Write analytic Plummer truth metrics and arrays, but never figures.

    ``r_min`` and ``r_max`` define the validation support. A cut experiment
    should therefore pass its training cut (for example ``r_min=1``); the same
    support is saved as a mask for two-dimensional displays.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max for Plummer validation.")
    if n_r < 2:
        raise ValueError("n_r must be at least 2.")
    if slice_grid < 2:
        raise ValueError("slice_grid must be at least 2.")
    resolved_slice_rmax = r_max if slice_rmax is None else float(slice_rmax)
    if resolved_slice_rmax <= 0:
        raise ValueError("slice_rmax must be positive.")

    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)
    output_dir = ensure_dir(output_dir)
    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Plummer analytic-truth validation",
    )
    phi_model, phi_params, _ = load_phi(phi_run_dir)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)

    sample_position = _sample_plummer_positions(
        int(n_eval),
        r_min=float(r_min),
        r_max=float(r_max),
        seed=int(seed),
    )
    sample_radius = np.linalg.norm(sample_position, axis=1)
    sample_model = _predict_plummer_fields(
        phi_model,
        phi_params,
        sample_position,
        mean_x,
        std_x,
        batch_size=int(batch_size),
    )
    sample_truth_potential = plummer_phi(sample_radius)
    sample_truth_density = plummer_rho(sample_radius)
    sample_truth_acceleration = -sample_position * (
        1.0 + sample_radius[:, None] ** 2
    ) ** (-1.5)
    potential_metrics, sample_aligned_potential = potential_error_metrics(
        sample_model["potential"],
        sample_truth_potential,
    )
    fitted_offset = float(potential_metrics["fitted_additive_offset"])

    radial_r = np.geomspace(r_min, r_max, int(n_r)).astype(np.float32)
    radial_position = np.column_stack(
        [radial_r, np.zeros_like(radial_r), np.zeros_like(radial_r)]
    )
    radial_model = _predict_plummer_fields(
        phi_model,
        phi_params,
        radial_position,
        mean_x,
        std_x,
        batch_size=int(batch_size),
    )

    slice_x = np.linspace(
        -resolved_slice_rmax,
        resolved_slice_rmax,
        int(slice_grid),
        dtype=np.float32,
    )
    slice_y = slice_x.copy()
    xx, yy = np.meshgrid(slice_x, slice_y, indexing="xy")
    slice_radius = np.sqrt(xx**2 + yy**2)
    slice_position = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)]
    )
    slice_model = _predict_plummer_fields(
        phi_model,
        phi_params,
        slice_position,
        mean_x,
        std_x,
        batch_size=int(batch_size),
    )
    slice_shape = xx.shape
    slice_support_mask = (slice_radius >= r_min) & (slice_radius <= r_max)

    metrics: dict[str, Any] = {
        "schema": "dpjax.plummer.truth-validation.v1",
        "n_eval": int(n_eval),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "r_min": float(r_min),
        "r_max": float(r_max),
        "n_r": int(n_r),
        "slice_grid": int(slice_grid),
        "slice_rmax": resolved_slice_rmax,
        "fitted_additive_offset": fitted_offset,
        "potential": potential_metrics,
        "acceleration": acceleration_error_metrics(
            sample_model["acceleration"],
            sample_truth_acceleration,
        ),
        "density": _scalar_error_metrics(
            sample_model["density"],
            sample_truth_density,
        ),
        "support": {
            "radial_cut_applied": bool(r_min > 1.0e-3),
            "slice_supported_fraction": float(np.mean(slice_support_mask)),
        },
        "df_run_dir": str(df_run_dir),
        "phi_run_dir": str(phi_run_dir),
    }
    diagnostics = {
        "sample_position": sample_position,
        "sample_radius": sample_radius,
        "sample_model_potential_raw": sample_model["potential"],
        "sample_model_potential": sample_aligned_potential,
        "sample_truth_potential": sample_truth_potential,
        "sample_model_acceleration": sample_model["acceleration"],
        "sample_truth_acceleration": sample_truth_acceleration,
        "sample_model_density": sample_model["density"],
        "sample_truth_density": sample_truth_density,
        "radial_r": radial_r,
        "radial_model_potential": radial_model["potential"] + fitted_offset,
        "radial_truth_potential": plummer_phi(radial_r),
        "radial_model_acceleration": radial_model["acceleration"][:, 0],
        "radial_truth_acceleration": plummer_ar(radial_r),
        "radial_model_density": radial_model["density"],
        "radial_truth_density": plummer_rho(radial_r),
        "slice_x": slice_x,
        "slice_y": slice_y,
        "slice_model_potential": (
            slice_model["potential"].reshape(slice_shape) + fitted_offset
        ),
        "slice_truth_potential": plummer_phi(slice_radius),
        "slice_model_density": slice_model["density"].reshape(slice_shape),
        "slice_truth_density": plummer_rho(slice_radius),
        "slice_support_mask": slice_support_mask,
    }
    prefix = f"{artifact_prefix}_" if artifact_prefix else ""
    (output_dir / f"{prefix}metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / f"{prefix}diagnostics.npz", **diagnostics)
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "output_dir": output_dir,
    }
