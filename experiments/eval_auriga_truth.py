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
    binned_potential_truth_by_phi,
    potential_error_metrics,
    radial_acceleration_profile,
)
from dpjax.models.potential import grad_phi_apply, load_phi, phi_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.plotting.diagnostics import (
    plot_auriga_potential_comparison,
    plot_potential_rz_by_phi,
)


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
    plots_dir: str | Path | None = None,
    n_eval: int | None = 65536,
    batch_size: int = 4096,
    seed: int = 0,
    truth_potential_scale: float = 1.0,
    truth_acceleration_scale: float = 1.0,
    radial_bins: int = 12,
    plot_truth: bool = True,
    slice_phi_bins: int = 6,
    slice_r_bins: int = 40,
    slice_z_bins: int = 40,
    slice_min_count: int = 3,
    slice_r_max: float | None = None,
    slice_z_max: float | None = None,
    fig_fmt: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 180,
) -> dict[str, Any]:
    """Compute simulator-truth metrics and render potential field diagnostics."""
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    phi_run_dir = resolve_path(phi_run_dir)
    out_dir = ensure_dir(out_dir or phi_run_dir / "eval" / "auriga_truth")
    plots_dir = ensure_dir(plots_dir or out_dir)

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

    if truth_potential is not None and plot_truth:
        plot_auriga_potential_comparison(
            positions,
            truth_potential,
            aligned_predicted_potential,
            metrics=metrics["potential"],
            fig_dir=plots_dir,
            fig_fmt=fig_fmt,
            dpi=int(dpi),
        )

        if (
            int(slice_phi_bins) < 1
            or int(slice_r_bins) < 1
            or int(slice_z_bins) < 1
        ):
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
        phi_edges = np.linspace(
            -np.pi,
            np.pi,
            int(slice_phi_bins) + 1,
        )
        cylindrical_radius_edges = np.linspace(
            0.0,
            r_max,
            int(slice_r_bins) + 1,
        )
        z_edges = np.linspace(
            -z_max,
            z_max,
            int(slice_z_bins) + 1,
        )
        truth_slices = binned_potential_truth_by_phi(
            positions,
            truth_potential,
            phi_edges=phi_edges,
            cylindrical_radius_edges=cylindrical_radius_edges,
            z_edges=z_edges,
            min_cell_count=int(slice_min_count),
        )

        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        radial_centers = 0.5 * (
            cylindrical_radius_edges[:-1]
            + cylindrical_radius_edges[1:]
        )
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        radial_grid, z_grid = np.meshgrid(
            radial_centers,
            z_centers,
            indexing="ij",
        )
        model_potential_grid = np.empty(
            (
                int(slice_phi_bins),
                int(slice_r_bins),
                int(slice_z_bins),
            ),
            dtype=np.float64,
        )
        fitted_offset = float(
            metrics["potential"]["fitted_additive_offset"]
        )
        for phi_index, phi_center in enumerate(phi_centers):
            grid_positions = np.column_stack(
                [
                    radial_grid.ravel() * np.cos(phi_center),
                    radial_grid.ravel() * np.sin(phi_center),
                    z_grid.ravel(),
                ]
            ).astype(np.float32)
            grid_std = (
                grid_positions - mean_x[None, :]
            ) / std_x[None, :]
            model_parts: list[np.ndarray] = []
            for start in range(0, grid_std.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), grid_std.shape[0])
                model_parts.append(
                    np.asarray(
                        phi_apply(
                            phi_model,
                            phi_params,
                            jnp.asarray(grid_std[start:stop]),
                        ),
                        dtype=np.float64,
                    )
                )
            model_potential_grid[phi_index] = (
                np.concatenate(model_parts).reshape(radial_grid.shape)
                + fitted_offset
            )

        slice_path = out_dir / "potential_rz_by_phi.npz"
        np.savez_compressed(
            slice_path,
            phi_edges=phi_edges,
            cylindrical_radius_edges=cylindrical_radius_edges,
            z_edges=z_edges,
            model_potential=model_potential_grid,
            truth_potential=truth_slices["truth_median"],
            truth_count=truth_slices["count"],
            min_cell_count=truth_slices["min_cell_count"],
            fitted_additive_offset=np.asarray(fitted_offset),
        )
        plot_potential_rz_by_phi(
            phi_edges,
            cylindrical_radius_edges,
            z_edges,
            model_potential_grid,
            truth_potential=truth_slices["truth_median"],
            truth_count=truth_slices["count"],
            min_cell_count=int(slice_min_count),
            fig_dir=plots_dir,
            fig_fmt=fig_fmt,
            dpi=int(dpi),
        )
        supported = truth_slices["count"] >= int(slice_min_count)
        metrics["potential_slices"] = {
            "phi_bins": int(slice_phi_bins),
            "cylindrical_radius_bins": int(slice_r_bins),
            "z_bins": int(slice_z_bins),
            "min_cell_count": int(slice_min_count),
            "supported_cell_fraction": float(np.mean(supported)),
            "r_max": r_max,
            "z_max": z_max,
        }

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
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--slice-phi-bins", type=int, default=6)
    parser.add_argument("--slice-r-bins", type=int, default=40)
    parser.add_argument("--slice-z-bins", type=int, default=40)
    parser.add_argument("--slice-min-count", type=int, default=3)
    parser.add_argument("--slice-r-max", type=float, default=None)
    parser.add_argument("--slice-z-max", type=float, default=None)
    parser.add_argument("--fig-formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
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
        plot_truth=not args.no_plots,
        slice_phi_bins=args.slice_phi_bins,
        slice_r_bins=args.slice_r_bins,
        slice_z_bins=args.slice_z_bins,
        slice_min_count=args.slice_min_count,
        slice_r_max=args.slice_r_max,
        slice_z_max=args.slice_z_max,
        fig_fmt=tuple(args.fig_formats),
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
