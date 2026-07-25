"""Evaluate Halo DF mass density and 6D score repeatability on a server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dpjax.data import preprocess_eta, require_physics_compatible_transform
from dpjax.datasets.auriga import load_auriga_snapshot
from dpjax.evaluation import (
    density_profile_metrics,
    score_ensemble_metrics,
    spherical_density_profile,
    stein_score_metrics,
)
from dpjax.flows.api import load_df, sample_apply, score_apply


DEFAULT_RADIAL_EDGES = np.array(
    [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 75.0],
    dtype=np.float64,
)


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


def evaluate_auriga_df(
    data_path: str | Path,
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    n_samples_per_model: int = 262_144,
    n_score_points: int = 32_768,
    score_batch_size: int = 1_024,
    seed: int = 42,
    radial_edges: np.ndarray = DEFAULT_RADIAL_EDGES,
) -> dict:
    """Run mass-density and ensemble-score acceptance diagnostics."""
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two independent DF run directories.")
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
            operation="Auriga physical score evaluation",
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

    samples = np.concatenate(all_samples, axis=0)
    scores = np.stack(all_scores, axis=0)
    reference_profile = spherical_density_profile(
        snapshot.eta[:, :3],
        weights=np.asarray(target_weights),
        edges=radial_edges,
    )
    model_profile = spherical_density_profile(
        samples[:, :3],
        edges=radial_edges,
    )
    density_metrics = density_profile_metrics(
        model_profile["density"],
        reference_profile["density"],
    )
    ensemble_metrics = score_ensemble_metrics(scores)
    median_score = np.median(scores, axis=0)
    stein_metrics = stein_score_metrics(
        score_eta,
        median_score,
        weights=score_weights,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "data": str(Path(data_path).resolve()),
        "run_dirs": [str(Path(path).resolve()) for path in run_dirs],
        "n_data": snapshot.n_particles,
        "n_flow_samples": int(samples.shape[0]),
        "density_target": (
            "tracer_weight"
            if snapshot.tracer_weight is not None
            else "mass"
            if snapshot.mass is not None
            else "particle_count"
        ),
        "density_profile": density_metrics,
        "score_ensemble": ensemble_metrics,
        "score_stein_consistency": stein_metrics,
        "interpretation": (
            "Auriga has no analytic 6D score truth: ensemble and Stein metrics "
            "measure repeatability/consistency, not absolute score accuracy."
        ),
    }
    (output_dir / "auriga_df_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "auriga_df_diagnostics.npz",
        radial_edges=radial_edges,
        reference_density=reference_profile["density"],
        model_density=model_profile["density"],
        reference_shell_probability=reference_profile["shell_probability"],
        model_shell_probability=model_profile["shell_probability"],
        score_indices=score_indices,
        score_eta=score_eta,
        scores=scores,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Halo12 mass density and 6D score stability."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Independent DF run directory; repeat at least twice.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples-per-model", type=int, default=262_144)
    parser.add_argument("--n-score-points", type=int, default=32_768)
    parser.add_argument("--score-batch-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = evaluate_auriga_df(
        args.data,
        args.run_dir,
        args.output_dir,
        n_samples_per_model=args.n_samples_per_model,
        n_score_points=args.n_score_points,
        score_batch_size=args.score_batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
