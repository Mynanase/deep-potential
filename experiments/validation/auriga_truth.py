"""Optional Auriga simulator-truth validation outside the run workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from dpjax.models.potential import grad_phi_apply, phi_apply
from experiments.datasets.auriga import load_auriga_snapshot
from experiments.datasets.phase_space import (
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from experiments.diagnostics.evaluation import (
    acceleration_error_metrics,
    binned_potential_truth_by_phi,
    potential_error_metrics,
    radial_acceleration_profile,
)
from experiments.paths import ensure_dir, resolve_path
from experiments.workflows.artifacts import load_phi


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


def _predict_potential(
    model: Any,
    params: dict,
    positions_std: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, positions_std.shape[0], batch_size):
        stop = min(start + batch_size, positions_std.shape[0])
        parts.append(
            np.asarray(
                phi_apply(model, params, jnp.asarray(positions_std[start:stop])),
                dtype=np.float64,
            )
        )
    return np.concatenate(parts)


def _predict_acceleration(
    model: Any,
    params: dict,
    positions_std: np.ndarray,
    std_x: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, positions_std.shape[0], batch_size):
        stop = min(start + batch_size, positions_std.shape[0])
        gradient_std = np.asarray(
            grad_phi_apply(
                model,
                params,
                jnp.asarray(positions_std[start:stop]),
            ),
            dtype=np.float64,
        )
        parts.append(-gradient_std / std_x[None, :])
    return np.concatenate(parts)


def evaluate_auriga_truth(
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    output_dir: str | Path,
    *,
    n_eval: int | None = 65_536,
    batch_size: int = 4_096,
    seed: int = 0,
    truth_potential_scale: float = 1.0,
    truth_acceleration_scale: float = 1.0,
    radial_bins: int = 12,
    slice_phi_bins: int = 6,
    slice_r_bins: int = 40,
    slice_z_bins: int = 40,
    slice_min_count: int = 3,
    slice_r_max: float | None = None,
    slice_z_max: float | None = None,
) -> dict[str, Any]:
    """Write simulator-truth metrics and arrays, but never figures.

    The function accepts ordinary Python values rather than a run config.
    Potential and acceleration are validated only when the prepared snapshot
    provides the corresponding truth field. Tracer density is never treated as
    total gravitating density truth.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if radial_bins <= 0:
        raise ValueError("radial_bins must be positive.")

    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)
    output_dir = ensure_dir(output_dir)

    snapshot = load_auriga_snapshot(data_path)
    if snapshot.potential is None and snapshot.acceleration is None:
        raise ValueError(
            "Auriga truth validation requires a potential and/or acceleration "
            "dataset in the prepared mock file."
        )

    indices = np.arange(snapshot.n_particles, dtype=np.int64)
    if n_eval is not None:
        if n_eval <= 0:
            raise ValueError("n_eval must be positive or None.")
        if indices.size > int(n_eval):
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, size=int(n_eval), replace=False))
    evaluated = snapshot.subset(indices)

    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Auriga simulator-truth validation",
    )
    phi_model, phi_params, _ = load_phi(phi_run_dir)

    positions = np.asarray(evaluated.eta[:, :3], dtype=np.float32)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    positions_std = (positions - mean_x[None, :]) / std_x[None, :]

    metrics: dict[str, Any] = {
        "schema": "dpjax.auriga.truth-validation.v1",
        "n_eval": int(evaluated.n_particles),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "data_path": str(data_path),
        "df_run_dir": str(df_run_dir),
        "phi_run_dir": str(phi_run_dir),
        "truth_units": {
            key: evaluated.attrs.get(key, "unknown")
            for key in (
                "length_unit",
                "velocity_unit",
                "potential_unit",
                "acceleration_unit",
            )
        },
        "available_truth": {
            "potential": evaluated.potential is not None,
            "acceleration": evaluated.acceleration is not None,
            "total_density": False,
        },
    }
    diagnostics: dict[str, np.ndarray] = {
        "row_index": indices,
        "source_index": np.asarray(evaluated.source_index),
        "sample_position": positions,
        "sample_radius": np.linalg.norm(positions, axis=1),
    }

    if evaluated.potential is not None:
        truth_potential = (
            np.asarray(evaluated.potential, dtype=np.float64)
            * float(truth_potential_scale)
        )
        predicted_potential = _predict_potential(
            phi_model,
            phi_params,
            positions_std,
            batch_size=int(batch_size),
        )
        potential_metrics, aligned_potential = potential_error_metrics(
            predicted_potential,
            truth_potential,
        )
        metrics["potential"] = potential_metrics
        metrics["truth_potential_scale"] = float(truth_potential_scale)
        diagnostics.update(
            {
                "sample_model_potential_raw": predicted_potential,
                "sample_model_potential": aligned_potential,
                "sample_truth_potential": truth_potential,
            }
        )

        if min(slice_phi_bins, slice_r_bins, slice_z_bins, slice_min_count) <= 0:
            raise ValueError("Potential slice bin counts must be positive.")
        cylindrical_radius = np.hypot(positions[:, 0], positions[:, 1])
        r_max = (
            float(slice_r_max)
            if slice_r_max is not None
            else float(np.percentile(cylindrical_radius, 99.0))
        )
        z_max = (
            float(slice_z_max)
            if slice_z_max is not None
            else float(np.percentile(np.abs(positions[:, 2]), 99.0))
        )
        r_max = max(r_max, 1.0e-6)
        z_max = max(z_max, 1.0e-6)
        phi_edges = np.linspace(-np.pi, np.pi, int(slice_phi_bins) + 1)
        radius_edges = np.linspace(0.0, r_max, int(slice_r_bins) + 1)
        z_edges = np.linspace(-z_max, z_max, int(slice_z_bins) + 1)
        truth_slices = binned_potential_truth_by_phi(
            positions,
            truth_potential,
            phi_edges=phi_edges,
            cylindrical_radius_edges=radius_edges,
            z_edges=z_edges,
            min_cell_count=int(slice_min_count),
        )

        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        radius_grid, z_grid = np.meshgrid(
            radius_centers,
            z_centers,
            indexing="ij",
        )
        model_slices = np.empty(
            (int(slice_phi_bins), int(slice_r_bins), int(slice_z_bins)),
            dtype=np.float64,
        )
        fitted_offset = float(potential_metrics["fitted_additive_offset"])
        for phi_index, phi_center in enumerate(phi_centers):
            grid_positions = np.column_stack(
                [
                    radius_grid.ravel() * np.cos(phi_center),
                    radius_grid.ravel() * np.sin(phi_center),
                    z_grid.ravel(),
                ]
            ).astype(np.float32)
            grid_std = (grid_positions - mean_x[None, :]) / std_x[None, :]
            model_slices[phi_index] = (
                _predict_potential(
                    phi_model,
                    phi_params,
                    grid_std,
                    batch_size=int(batch_size),
                ).reshape(radius_grid.shape)
                + fitted_offset
            )

        supported = truth_slices["count"] >= int(slice_min_count)
        diagnostics.update(
            {
                "slice_phi_edges": phi_edges,
                "slice_radius_edges": radius_edges,
                "slice_z_edges": z_edges,
                "slice_model_potential": model_slices,
                "slice_truth_potential": truth_slices["truth_median"],
                "slice_truth_count": truth_slices["count"],
                "slice_truth_mask": supported,
            }
        )
        metrics["potential_slices"] = {
            "phi_bins": int(slice_phi_bins),
            "cylindrical_radius_bins": int(slice_r_bins),
            "z_bins": int(slice_z_bins),
            "min_cell_count": int(slice_min_count),
            "supported_cell_fraction": float(np.mean(supported)),
            "r_max": r_max,
            "z_max": z_max,
        }

    if evaluated.acceleration is not None:
        truth_acceleration = (
            np.asarray(evaluated.acceleration, dtype=np.float64)
            * float(truth_acceleration_scale)
        )
        predicted_acceleration = _predict_acceleration(
            phi_model,
            phi_params,
            positions_std,
            std_x,
            batch_size=int(batch_size),
        )
        metrics["acceleration"] = acceleration_error_metrics(
            predicted_acceleration,
            truth_acceleration,
        )
        metrics["radial_acceleration"] = radial_acceleration_profile(
            positions,
            predicted_acceleration,
            truth_acceleration,
            n_bins=int(radial_bins),
        )
        metrics["truth_acceleration_scale"] = float(truth_acceleration_scale)
        diagnostics.update(
            {
                "sample_model_acceleration": predicted_acceleration,
                "sample_truth_acceleration": truth_acceleration,
            }
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "diagnostics.npz", **diagnostics)
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "output_dir": output_dir,
    }
