"""Evaluate one Halo DF score against row-aligned simulator acceleration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from dpjax.data import (
    fit_normalizer,
    preprocess_eta,
    require_physics_compatible_transform,
)
from dpjax.datasets.auriga import AurigaSnapshot, load_auriga_snapshot
from dpjax.evaluation import (
    stein_score_metrics,
    truth_cbe_radial_profile,
    truth_cbe_score_metrics,
)
from dpjax.flows.api import load_df, score_apply
from dpjax.paths import ensure_dir, resolve_path
from dpjax.plotting.auriga_score import (
    COMPONENT_LABELS,
    plot_truth_cbe_diagnostics,
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


def _target_weights(
    snapshot: AurigaSnapshot,
    config: dict[str, Any],
) -> tuple[np.ndarray, str]:
    weight_dataset = config.get("data", {}).get("weight_dataset")
    if weight_dataset:
        aliases = {
            "tracer_weight": snapshot.tracer_weight,
            "TracerWeight": snapshot.tracer_weight,
            "weights": snapshot.tracer_weight,
            "mass": snapshot.mass,
            "Masses": snapshot.mass,
        }
        values = aliases.get(str(weight_dataset))
        if values is None:
            raise ValueError(
                f"DF config requests weight dataset {weight_dataset!r}, but the "
                "prepared Auriga file does not expose that row-aligned field."
            )
        return np.asarray(values, dtype=np.float64), str(weight_dataset)
    if snapshot.tracer_weight is not None:
        return np.asarray(snapshot.tracer_weight, dtype=np.float64), "tracer_weight"
    if snapshot.mass is not None:
        return np.asarray(snapshot.mass, dtype=np.float64), "mass"
    return np.ones(snapshot.n_particles, dtype=np.float64), "particle_count"


def _split_indices(
    snapshot: AurigaSnapshot,
    config: dict[str, Any],
    weights: np.ndarray,
    *,
    subset: str,
) -> np.ndarray:
    """Reconstruct the DF train/validation split, including sigma clipping."""
    if subset not in {"all", "train", "validation"}:
        raise ValueError(f"Unknown subset {subset!r}.")
    eligible = np.arange(snapshot.n_particles, dtype=np.int64)
    if subset == "all":
        return eligible

    data_cfg = config.get("data", {})
    clip_sigma = float(data_cfg.get("clip_sigma", 0.0))
    if clip_sigma > 0:
        clip_normalizer = fit_normalizer(snapshot.eta, weights=weights)
        clip_std = np.maximum(clip_normalizer.std, 1.0e-6)
        keep = np.all(
            np.abs(snapshot.eta - clip_normalizer.mean) < clip_sigma * clip_std,
            axis=1,
        )
        eligible = eligible[keep]

    val_frac = float(np.clip(data_cfg.get("val_frac", 0.1), 0.0, 0.5))
    n_val = int(round(eligible.size * val_frac))
    n_val = min(max(n_val, 0), max(eligible.size - 1, 0))
    split_seed = int(data_cfg.get("split_seed", config.get("seed", 0)))
    split_order = np.random.default_rng(split_seed).permutation(eligible.size)
    if subset == "validation":
        if n_val == 0:
            raise ValueError(
                "The DF config has no validation rows; use --subset all or train."
            )
        return eligible[split_order[:n_val]]
    return eligible[split_order[n_val:]]


def _physical_score(
    model,
    params,
    normalizer,
    flow_cfg: dict[str, Any],
    eta: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    eta_std = preprocess_eta(eta, normalizer, None)
    parts: list[np.ndarray] = []
    for start in range(0, eta.shape[0], int(batch_size)):
        score_std = np.asarray(
            score_apply(
                model,
                params,
                jnp.asarray(eta_std[start : start + int(batch_size)]),
                flow_cfg,
            ),
            dtype=np.float64,
        )
        parts.append(score_std / normalizer.std[None, :])
    return np.concatenate(parts, axis=0)


def run_eval_auriga_df_acceleration(
    data_path: str | Path,
    df_run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    subset: str = "validation",
    n_eval: int | None = 65_536,
    batch_size: int = 1_024,
    seed: int = 0,
    truth_acceleration_scale: float = 1.0,
    normalization_floor_fraction: float = 1.0e-6,
    radial_bins: int = 12,
    dpi: int = 180,
) -> dict[str, Any]:
    """Evaluate a single physical DF score with simulator-acceleration CBE."""
    data_path = resolve_path(data_path)
    df_run_dir = resolve_path(df_run_dir)
    out_dir = ensure_dir(
        out_dir or df_run_dir / "eval" / "auriga_df_acceleration"
    )

    snapshot = load_auriga_snapshot(data_path)
    if snapshot.acceleration is None:
        raise ValueError(
            "Auriga DF acceleration evaluation requires a row-aligned "
            "acceleration/Acceleration/GravAcceleration dataset."
        )
    if truth_acceleration_scale <= 0:
        raise ValueError("truth_acceleration_scale must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model, params, normalizer, config, coordinate_transform = load_df(df_run_dir)
    require_physics_compatible_transform(
        coordinate_transform,
        operation="Auriga DF truth-acceleration CBE evaluation",
    )
    weights, weight_target = _target_weights(snapshot, config)
    indices = _split_indices(
        snapshot,
        config,
        weights,
        subset=subset,
    )
    if n_eval is not None:
        if n_eval <= 0:
            raise ValueError("n_eval must be positive or None.")
        if indices.size > int(n_eval):
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(indices, size=int(n_eval), replace=False)
            )
    evaluated = snapshot.subset(indices)
    evaluated_weights = weights[indices]
    acceleration = (
        np.asarray(evaluated.acceleration, dtype=np.float64)
        * float(truth_acceleration_scale)
    )
    score = _physical_score(
        model,
        params,
        normalizer,
        config.get("flow", {}),
        np.asarray(evaluated.eta),
        batch_size=int(batch_size),
    )

    cbe_metrics, diagnostics = truth_cbe_score_metrics(
        evaluated.eta,
        score,
        acceleration,
        weights=evaluated_weights,
        normalization_floor_fraction=float(normalization_floor_fraction),
    )
    finite = diagnostics["finite_mask"]
    radial_profile = truth_cbe_radial_profile(
        np.asarray(evaluated.eta)[finite, :3],
        diagnostics["residual"],
        diagnostics["normalized_residual"],
        weights=evaluated_weights[finite],
        n_bins=int(radial_bins),
    )
    stein_metrics = stein_score_metrics(
        evaluated.eta,
        score,
        weights=evaluated_weights,
    )

    component_metrics: dict[str, Any] = {}
    if evaluated.component is not None:
        for value, label in COMPONENT_LABELS.items():
            mask = np.asarray(evaluated.component) == value
            if np.any(mask):
                component_metrics[label], _ = truth_cbe_score_metrics(
                    np.asarray(evaluated.eta)[mask],
                    score[mask],
                    acceleration[mask],
                    weights=evaluated_weights[mask],
                    normalization_floor_fraction=float(
                        normalization_floor_fraction
                    ),
                )

    metrics: dict[str, Any] = {
        "schema": "dpjax.auriga.df-acceleration-cbe-eval.v1",
        "data_path": str(data_path),
        "df_run_dir": str(df_run_dir),
        "subset": subset,
        "n_eval": int(evaluated.n_particles),
        "seed": int(seed),
        "weight_target": weight_target,
        "truth_acceleration_scale": float(truth_acceleration_scale),
        "truth_units": {
            key: evaluated.attrs.get(key, "unknown")
            for key in (
                "length_unit",
                "velocity_unit",
                "acceleration_unit",
            )
        },
        "cbe_truth_acceleration": cbe_metrics,
        "score_stein_consistency": stein_metrics,
        "radial_profile": radial_profile,
        "component_metrics": component_metrics,
        "interpretation": (
            "This evaluates the learned score projected along the simulator "
            "Hamiltonian flow under a stationary-CBE assumption. It is not "
            "six independent score-component truth."
        ),
    }
    metrics_path = out_dir / "auriga_df_acceleration_metrics.json"
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / "auriga_df_acceleration_diagnostics.npz",
        row_index=indices,
        source_index=np.asarray(evaluated.source_index),
        eta=np.asarray(evaluated.eta),
        score=score,
        truth_acceleration=acceleration,
        tracer_weight=evaluated_weights,
        finite_row_index=indices[finite],
        transport_term=diagnostics["transport_term"],
        acceleration_term=diagnostics["acceleration_term"],
        residual=diagnostics["residual"],
        term_amplitude=diagnostics["term_amplitude"],
        normalized_residual=diagnostics["normalized_residual"],
        component=(
            np.asarray([], dtype=np.int8)
            if evaluated.component is None
            else np.asarray(evaluated.component)
        ),
    )
    plot_paths = plot_truth_cbe_diagnostics(
        diagnostics,
        cbe_metrics,
        radial_profile,
        component=(
            None
            if evaluated.component is None
            else np.asarray(evaluated.component)[finite]
        ),
        fig_dir=out_dir / "plots",
        dpi=int(dpi),
    )
    metrics["outputs"] = {
        "metrics": str(metrics_path),
        "diagnostics": str(
            out_dir / "auriga_df_acceleration_diagnostics.npz"
        ),
        "plots": [str(path) for path in plot_paths],
    }
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(metrics), indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Halo DF score against row-aligned simulator "
            "acceleration through the stationary CBE."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--df-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--subset",
        choices=("validation", "train", "all"),
        default="validation",
        help=(
            "Reconstruct the saved DF split; validation assumes --data has "
            "the same row order as the training file."
        ),
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=65_536,
        help="Maximum number of rows; use 0 to evaluate the full subset.",
    )
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truth-acceleration-scale", type=float, default=1.0)
    parser.add_argument(
        "--normalization-floor-fraction",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    run_eval_auriga_df_acceleration(
        args.data,
        args.df_run_dir,
        out_dir=args.out_dir,
        subset=args.subset,
        n_eval=None if args.n_eval == 0 else args.n_eval,
        batch_size=args.batch_size,
        seed=args.seed,
        truth_acceleration_scale=args.truth_acceleration_scale,
        normalization_floor_fraction=args.normalization_floor_fraction,
        radial_bins=args.radial_bins,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
