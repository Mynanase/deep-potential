"""Evaluate persisted DF marginals and physical-score consistency."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.flows.api import sample_apply, score_apply
from experiments.datasets.auriga import load_auriga_snapshot
from experiments.datasets.phase_space import (
    preprocess_eta,
    require_physics_compatible_transform,
)
from experiments.diagnostics.evaluation import (
    conditional_velocity_diagnostics,
    cylindrical_rz_density_by_phi,
    density_profile_metrics,
    score_ensemble_metrics,
    spherical_density_profile,
    stein_score_metrics,
)
from experiments.workflows.artifacts import load_df

DEFAULT_RADIAL_EDGES = np.array(
    [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 75.0],
    dtype=np.float64,
)
DEFAULT_THETA_EDGES = np.arccos(np.linspace(1.0, -1.0, 7))
DEFAULT_PHI_EDGES = np.linspace(-np.pi, np.pi, 9)
DEFAULT_SPATIAL_R_EDGES = np.linspace(0.0, 75.0, 49)
DEFAULT_SPATIAL_Z_EDGES = np.linspace(-75.0, 75.0, 49)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _score_in_physical_coordinates(
    model,
    params,
    normalizer,
    flow_cfg: dict,
    eta: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    eta_std = preprocess_eta(eta, normalizer, None)
    for start in range(0, eta.shape[0], batch_size):
        batch = jnp.asarray(eta_std[start : start + batch_size])
        score_std = np.asarray(score_apply(model, params, batch, flow_cfg))
        rows.append(score_std / normalizer.std[None, :])
    return np.concatenate(rows, axis=0)


def evaluate_df_diagnostics(
    data_path: str | Path,
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    n_samples_per_model: int = 262_144,
    n_score_points: int = 32_768,
    score_batch_size: int = 1_024,
    seed: int = 42,
    radial_edges: np.ndarray = DEFAULT_RADIAL_EDGES,
    theta_edges: np.ndarray = DEFAULT_THETA_EDGES,
    phi_edges: np.ndarray = DEFAULT_PHI_EDGES,
    n_velocity_bins: int = 64,
    spatial_r_edges: np.ndarray = DEFAULT_SPATIAL_R_EDGES,
    spatial_z_edges: np.ndarray = DEFAULT_SPATIAL_Z_EDGES,
    spatial_min_cell_count: int = 5,
) -> dict:
    """Persist generic distribution and score diagnostics for one or more DFs."""
    if len(run_dirs) < 1:
        raise ValueError("Provide at least one DF run directory.")
    if n_samples_per_model <= 0:
        raise ValueError("n_samples_per_model must be positive.")
    if n_score_points <= 0:
        raise ValueError("n_score_points must be positive.")
    snapshot = load_auriga_snapshot(data_path)
    target_weights = (
        snapshot.tracer_weight
        if snapshot.tracer_weight is not None
        else snapshot.mass
    )
    if target_weights is None:
        target_weights = np.ones(snapshot.n_particles, dtype=np.float32)

    rng = np.random.default_rng(seed)
    n_score = min(int(n_score_points), snapshot.n_particles)
    score_indices = np.sort(
        rng.choice(snapshot.n_particles, size=n_score, replace=False)
    )
    score_eta = snapshot.eta[score_indices]
    score_weights = np.asarray(target_weights)[score_indices]

    all_samples: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for model_index, run_dir in enumerate(run_dirs):
        model, params, normalizer, cfg, transform = load_df(run_dir)
        require_physics_compatible_transform(
            transform,
            operation="physical score evaluation",
        )
        flow_cfg = cfg.get("flow", {})
        sample_key = jax.random.key(seed + model_index)
        sample_std = sample_apply(
            model,
            params,
            sample_key,
            int(n_samples_per_model),
            flow_cfg,
        )
        samples = normalizer.inverse(np.asarray(sample_std))
        all_samples.append(samples)
        all_scores.append(
            _score_in_physical_coordinates(
                model,
                params,
                normalizer,
                flow_cfg,
                score_eta,
                batch_size=int(score_batch_size),
            )
        )

    samples_by_model = np.stack(all_samples, axis=0)
    scores = np.stack(all_scores, axis=0)
    reference_profile = spherical_density_profile(
        snapshot.eta[:, :3],
        weights=np.asarray(target_weights),
        edges=radial_edges,
    )
    model_profiles = [
        spherical_density_profile(
            samples[:, :3],
            edges=radial_edges,
        )
        for samples in samples_by_model
    ]
    model_density_by_model = np.stack(
        [profile["density"] for profile in model_profiles],
        axis=0,
    )
    model_shell_probability_by_model = np.stack(
        [profile["shell_probability"] for profile in model_profiles],
        axis=0,
    )
    model_density = np.median(model_density_by_model, axis=0)
    model_shell_probability = np.median(
        model_shell_probability_by_model,
        axis=0,
    )
    density_metrics = density_profile_metrics(
        model_density,
        reference_profile["density"],
    )
    density_metrics_by_model = [
        density_profile_metrics(density, reference_profile["density"])
        for density in model_density_by_model
    ]
    ensemble_metrics = (
        score_ensemble_metrics(scores)
        if scores.shape[0] >= 2
        else None
    )
    median_score = np.median(scores, axis=0)
    stein_metrics = stein_score_metrics(
        score_eta,
        median_score,
        weights=score_weights,
    )

    conditional = conditional_velocity_diagnostics(
        snapshot.eta,
        samples_by_model,
        reference_weights=np.asarray(target_weights),
        conditioning_edges={
            "r": radial_edges,
            "theta": theta_edges,
            "phi": phi_edges,
        },
        n_velocity_bins=int(n_velocity_bins),
    )
    spatial = cylindrical_rz_density_by_phi(
        snapshot.eta[:, :3],
        samples_by_model[:, :, :3],
        reference_weights=np.asarray(target_weights),
        phi_edges=phi_edges,
        cylindrical_radius_edges=spatial_r_edges,
        z_edges=spatial_z_edges,
        min_cell_count=int(spatial_min_cell_count),
    )

    def finite_median(values: np.ndarray, axis=None):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            if axis is None:
                return None
            output_shape = np.asarray(values).shape
            if isinstance(axis, tuple):
                for index in sorted(axis, reverse=True):
                    output_shape = (
                        output_shape[:index] + output_shape[index + 1 :]
                    )
            else:
                output_shape = output_shape[:axis] + output_shape[axis + 1 :]
            return np.full(output_shape, np.nan).tolist()
        with np.errstate(invalid="ignore"):
            result = np.nanmedian(np.where(finite, values, np.nan), axis=axis)
        if np.ndim(result) == 0:
            return float(result)
        return np.asarray(result).tolist()

    velocity_labels = ("v_r", "v_theta", "v_phi")
    conditional_summary = {}
    for coordinate_name in ("r", "theta", "phi"):
        values = conditional[coordinate_name]
        conditional_summary[coordinate_name] = {
            "n_bins": int(np.asarray(values["edges"]).size - 1),
            "min_reference_effective_count": float(
                np.min(values["reference_effective_count"])
            ),
            "median_wasserstein_by_velocity": dict(
                zip(
                    velocity_labels,
                    finite_median(values["wasserstein"], axis=(0, 1)),
                )
            ),
            "median_histogram_ks_by_velocity": dict(
                zip(
                    velocity_labels,
                    finite_median(values["ks_histogram"], axis=(0, 1)),
                )
            ),
            "median_js_divergence_by_velocity": dict(
                zip(
                    velocity_labels,
                    finite_median(values["js_divergence"], axis=(0, 1)),
                )
            ),
        }

    model_labels = [Path(path).parent.name for path in run_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "data": str(Path(data_path).resolve()),
        "run_dirs": [str(Path(path).resolve()) for path in run_dirs],
        "model_labels": model_labels,
        "n_models": len(run_dirs),
        "n_data": snapshot.n_particles,
        "n_flow_samples_per_model": int(samples_by_model.shape[1]),
        "n_flow_samples": int(
            samples_by_model.shape[0] * samples_by_model.shape[1]
        ),
        "density_target": (
            "tracer_weight"
            if snapshot.tracer_weight is not None
            else "mass"
            if snapshot.mass is not None
            else "particle_count"
        ),
        "density_semantics": (
            "Normalized stellar tracer mass density learned by the DF; this is "
            "not the total gravitating density from the Poisson equation."
        ),
        "density_profile": density_metrics,
        "density_profile_by_model": density_metrics_by_model,
        "spatial_rz_by_phi": {
            "n_phi_bins": int(np.asarray(spatial["phi_edges"]).size - 1),
            "n_r_bins": int(
                np.asarray(spatial["cylindrical_radius_edges"]).size - 1
            ),
            "n_z_bins": int(np.asarray(spatial["z_edges"]).size - 1),
            "min_cell_count": int(spatial_min_cell_count),
            "median_log10_rmse_dex": finite_median(
                spatial["log10_rmse_by_model_phi"]
            ),
            "log10_rmse_by_model_phi": np.asarray(
                spatial["log10_rmse_by_model_phi"]
            ).tolist(),
        },
        "conditional_velocity": conditional_summary,
        "score_ensemble": ensemble_metrics,
        "score_stein_consistency": stein_metrics,
        "interpretation": (
            "Data/model histograms test DF marginals. Without an analytic "
            "6D score or acceleration truth, ensemble and Stein metrics "
            "measure repeatability/consistency, not absolute score accuracy."
        ),
    }
    result = _json_safe(result)
    (output_dir / "df_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    diagnostics: dict[str, np.ndarray] = {
        "radial_edges": radial_edges,
        "reference_density": reference_profile["density"],
        "model_density": model_density,
        "model_density_by_model": model_density_by_model,
        "reference_shell_probability": reference_profile["shell_probability"],
        "model_shell_probability": model_shell_probability,
        "model_shell_probability_by_model": (
            model_shell_probability_by_model
        ),
        "score_indices": score_indices,
        "score_eta": score_eta,
        "scores": scores,
        "model_labels": np.asarray(model_labels),
        "conditional_velocity_edges": np.asarray(
            conditional["velocity_edges"]
        ),
        "spatial_phi_edges": np.asarray(spatial["phi_edges"]),
        "spatial_r_edges": np.asarray(
            spatial["cylindrical_radius_edges"]
        ),
        "spatial_z_edges": np.asarray(spatial["z_edges"]),
        "spatial_cell_volume": np.asarray(spatial["cell_volume"]),
        "spatial_reference_density": np.asarray(
            spatial["reference_density"]
        ),
        "spatial_model_density": np.asarray(spatial["model_density"]),
        "spatial_model_median_density": np.asarray(
            spatial["model_median_density"]
        ),
        "spatial_reference_count": np.asarray(spatial["reference_count"]),
        "spatial_model_count": np.asarray(spatial["model_count"]),
        "spatial_log10_rmse_by_model_phi": np.asarray(
            spatial["log10_rmse_by_model_phi"]
        ),
        "spatial_min_cell_count": np.asarray(spatial["min_cell_count"]),
    }
    for coordinate_name in ("r", "theta", "phi"):
        values = conditional[coordinate_name]
        for key, value in values.items():
            diagnostics[f"conditional_{coordinate_name}_{key}"] = np.asarray(
                value
            )
    np.savez_compressed(
        output_dir / "df_diagnostics.npz",
        **diagnostics,
    )
    np.savez_compressed(
        output_dir / "df_samples.npz",
        reference_eta=np.asarray(snapshot.eta),
        model_eta_by_model=samples_by_model,
        model_labels=np.asarray(model_labels),
    )
    return result
