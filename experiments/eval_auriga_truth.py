"""Evaluate a trained static potential directly against Auriga mock truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from dpjax.data import (
    load_run_preprocessing,
    require_physics_compatible_transform,
)
from dpjax.datasets.auriga import load_auriga_snapshot
from dpjax.evaluation import (
    acceleration_error_metrics,
    potential_error_metrics,
    radial_acceleration_profile,
)
from dpjax.models.potential import grad_phi_apply, load_phi, phi_apply
from dpjax.paths import ensure_dir, resolve_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_eval_auriga_truth(
    data_path: str | Path,
    df_run_dir: str | Path,
    phi_run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    n_eval: int | None = 65536,
    batch_size: int = 4096,
    seed: int = 0,
    truth_potential_scale: float = 1.0,
    truth_acceleration_scale: float = 1.0,
    radial_bins: int = 12,
) -> dict[str, Any]:
    """Compute additive-offset-safe potential and vector force errors."""
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)
    out_dir = ensure_dir(out_dir or phi_run_dir / "eval" / "auriga_truth")

    snapshot = load_auriga_snapshot(data_path)
    if snapshot.potential is None and snapshot.acceleration is None:
        raise ValueError(
            "Auriga truth evaluation requires a potential and/or acceleration "
            "dataset in the prepared mock file."
        )

    normalizer, coordinate_transform = load_run_preprocessing(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Auriga simulator-truth evaluation",
    )
    phi_model, phi_params, _ = load_phi(phi_run_dir)

    indices = np.arange(snapshot.n_particles, dtype=np.int64)
    if n_eval is not None:
        if n_eval <= 0:
            raise ValueError("n_eval must be positive or None.")
        if indices.size > int(n_eval):
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(indices, size=int(n_eval), replace=False)
            )
    evaluated = snapshot.subset(indices)

    positions = np.asarray(evaluated.eta[:, :3], dtype=np.float32)
    mean_x = np.asarray(normalizer.mean[:3], dtype=np.float32)
    std_x = np.asarray(normalizer.std[:3], dtype=np.float32)
    positions_std = (positions - mean_x[None, :]) / std_x[None, :]

    predicted_potential_parts: list[np.ndarray] = []
    predicted_acceleration_parts: list[np.ndarray] = []
    for start in range(0, positions_std.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), positions_std.shape[0])
        batch = jnp.asarray(positions_std[start:stop])
        predicted_potential_parts.append(
            np.asarray(phi_apply(phi_model, phi_params, batch), dtype=np.float32)
        )
        gradient_std = np.asarray(
            grad_phi_apply(phi_model, phi_params, batch),
            dtype=np.float32,
        )
        predicted_acceleration_parts.append(-gradient_std / std_x[None, :])

    predicted_potential = np.concatenate(predicted_potential_parts)
    predicted_acceleration = np.concatenate(predicted_acceleration_parts)
    metrics: dict[str, Any] = {
        "schema": "dpjax.auriga.truth-eval.v1",
        "n_eval": int(evaluated.n_particles),
        "seed": int(seed),
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
    }

    aligned_predicted_potential = predicted_potential
    truth_potential = None
    if evaluated.potential is not None:
        truth_potential = (
            np.asarray(evaluated.potential, dtype=np.float64)
            * float(truth_potential_scale)
        )
        potential_metrics, aligned_predicted_potential = (
            potential_error_metrics(predicted_potential, truth_potential)
        )
        metrics["potential"] = potential_metrics
        metrics["truth_potential_scale"] = float(truth_potential_scale)

    truth_acceleration = None
    if evaluated.acceleration is not None:
        truth_acceleration = (
            np.asarray(evaluated.acceleration, dtype=np.float64)
            * float(truth_acceleration_scale)
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
        metrics["truth_acceleration_scale"] = float(
            truth_acceleration_scale
        )

    metrics_path = out_dir / "auriga_truth_metrics.json"
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / "auriga_truth_predictions.npz",
        row_index=indices,
        source_index=np.asarray(evaluated.source_index),
        position=positions,
        predicted_potential=predicted_potential,
        aligned_predicted_potential=aligned_predicted_potential,
        truth_potential=(
            np.asarray([], dtype=np.float32)
            if truth_potential is None
            else truth_potential
        ),
        predicted_acceleration=predicted_acceleration,
        truth_acceleration=(
            np.empty((0, 3), dtype=np.float32)
            if truth_acceleration is None
            else truth_acceleration
        ),
    )
    print(json.dumps(_json_safe(metrics), indent=2))
    print(f"Wrote {metrics_path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate learned Phi and acceleration against a prepared Auriga "
            "mock with simulator truth."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--df-run-dir", type=Path, required=True)
    parser.add_argument("--phi-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-eval", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth-potential-scale", type=float, default=1.0)
    parser.add_argument("--truth-acceleration-scale", type=float, default=1.0)
    parser.add_argument("--radial-bins", type=int, default=12)
    args = parser.parse_args()

    run_eval_auriga_truth(
        args.data,
        args.df_run_dir,
        args.phi_run_dir,
        out_dir=args.out_dir,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        seed=args.seed,
        truth_potential_scale=args.truth_potential_scale,
        truth_acceleration_scale=args.truth_acceleration_scale,
        radial_bins=args.radial_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
