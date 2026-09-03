"""Pure input-data velocity validation outside any trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.datasets.auriga import load_auriga_snapshot
from experiments.datasets.phase_space import sigma_clip_mask
from experiments.diagnostics.evaluation import input_velocity_diagnostics
from experiments.paths import ensure_dir, resolve_path
from experiments.plotting.df_diagnostics import plot_input_velocity_distributions
from experiments.workflows.config import load_run_spec
from experiments.workflows.evaluation.df import DEFAULT_RADIAL_EDGES

DEFAULT_OUTPUT_DIR = Path("runs/validation/input-data-check")
FIGURE_NAMES = {
    "r": "input_velocity_by_r.png",
    "theta": "input_velocity_by_theta.png",
    "phi": "input_velocity_by_phi.png",
}


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


def apply_input_preprocessing(
    eta: np.ndarray,
    weights: np.ndarray,
    *,
    clip_sigma: float,
    r_center_cut: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the data-level cuts a run config would apply before training.

    Only row-removal cuts are reproduced (sigma clip and center hole); training
    augmentations such as jitter or boundary padding are deliberately skipped
    because they do not describe the input data.
    """
    kept = np.ones(eta.shape[0], dtype=bool)
    removed_sigma = 0
    if clip_sigma > 0.0:
        mask = sigma_clip_mask(eta, float(clip_sigma), weights=weights)
        removed_sigma = int(np.count_nonzero(~mask))
        kept &= mask
    removed_center = 0
    if r_center_cut > 0.0:
        radius = np.linalg.norm(eta[:, :3], axis=-1)
        center_mask = radius >= float(r_center_cut)
        removed_center = int(np.count_nonzero(kept & ~center_mask))
        kept &= center_mask
    summary = {
        "n_input": int(eta.shape[0]),
        "n_kept": int(np.count_nonzero(kept)),
        "n_removed_sigma_clip": removed_sigma,
        "n_removed_center_cut": removed_center,
    }
    return eta[kept], weights[kept], summary


def run(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    radial_edges: np.ndarray,
    theta_edges: np.ndarray,
    phi_edges: np.ndarray,
    n_velocity_bins: int = 64,
) -> dict[str, Any]:
    """Compute and persist input-data velocity diagnostics and figures."""
    data_path = resolve_path(data_path)
    output_dir = ensure_dir(output_dir)
    snapshot = load_auriga_snapshot(data_path)
    weights = (
        snapshot.tracer_weight
        if snapshot.tracer_weight is not None
        else snapshot.mass
    )
    if weights is None:
        weights = np.ones(snapshot.n_particles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    eta = np.asarray(snapshot.eta)
    preprocessing_summary: dict[str, Any] = {"applied": False}
    if config_path is not None:
        spec = load_run_spec(resolve_path(config_path))
        preprocessing = dict(spec.data_config.get("preprocessing", {}))
        eta, weights, preprocessing_summary = apply_input_preprocessing(
            eta,
            weights,
            clip_sigma=float(preprocessing.get("clip_sigma", 0.0)),
            r_center_cut=float(preprocessing.get("r_center_cut", 0.0)),
        )
        preprocessing_summary["applied"] = True
        preprocessing_summary["config"] = str(resolve_path(config_path))

    diagnostics = input_velocity_diagnostics(
        eta,
        weights=weights,
        conditioning_edges={
            "r": radial_edges,
            "theta": theta_edges,
            "phi": phi_edges,
        },
        n_velocity_bins=int(n_velocity_bins),
    )
    np.savez_compressed(
        output_dir / "input_velocity_diagnostics.npz",
        **diagnostics,
    )

    import matplotlib.pyplot as plt

    summary: dict[str, Any] = {
        "data": str(data_path),
        "n_particles": int(eta.shape[0]),
        "preprocessing": preprocessing_summary,
        "n_velocity_bins": int(n_velocity_bins),
        "coordinates": {},
    }
    for coordinate in ("r", "theta", "phi"):
        prefix = f"input_{coordinate}_"
        count = np.asarray(diagnostics[f"{prefix}count"])
        effective = np.asarray(diagnostics[f"{prefix}effective_count"])
        skewness = np.asarray(diagnostics[f"{prefix}skewness"])
        excess_kurtosis = np.asarray(diagnostics[f"{prefix}excess_kurtosis"])
        summary["coordinates"][coordinate] = {
            "n_bins": int(count.size),
            "min_count": int(np.min(count)),
            "min_effective_count": float(np.min(effective)),
            "median_skewness_by_velocity": {
                name: float(np.nanmedian(skewness[:, index]))
                for index, name in enumerate(("v_r", "v_theta", "v_phi"))
            },
            "median_excess_kurtosis_by_velocity": {
                name: float(np.nanmedian(excess_kurtosis[:, index]))
                for index, name in enumerate(("v_r", "v_theta", "v_phi"))
            },
        }
        figure = plot_input_velocity_distributions(
            diagnostics,
            coordinate,
            dpi=200,
        )
        figure.savefig(
            output_dir / FIGURE_NAMES[coordinate],
            bbox_inches="tight",
        )
        plt.close(figure)
    (output_dir / "input_velocity_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate raw input data: per spatial bin, show the distribution "
            "and shape of the three spherical velocity components without "
            "any trained model."
        )
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to the snapshot H5 file"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional run config YAML; its data-level cuts (clip_sigma, "
            "r_center_cut) are applied before validation."
        ),
    )
    parser.add_argument(
        "--radial-edges",
        type=str,
        default=None,
        help=(
            "Comma-separated radial bin edges in kpc. Default mirrors the "
            "DF evaluation radial grid."
        ),
    )
    parser.add_argument("--n-theta-bins", type=int, default=6)
    parser.add_argument("--n-phi-bins", type=int, default=8)
    parser.add_argument("--n-velocity-bins", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.radial_edges is not None:
        radial_edges = np.asarray(
            [float(value) for value in args.radial_edges.split(",")],
            dtype=np.float64,
        )
        if radial_edges.size < 2 or np.any(np.diff(radial_edges) <= 0):
            raise ValueError("--radial-edges must be strictly increasing.")
    else:
        radial_edges = np.asarray(DEFAULT_RADIAL_EDGES, dtype=np.float64)
    theta_edges = np.arccos(
        np.linspace(1.0, -1.0, int(args.n_theta_bins) + 1)
    )
    phi_edges = np.linspace(
        -np.pi, np.pi, int(args.n_phi_bins) + 1
    )
    if args.n_phi_bins < 1 or args.n_theta_bins < 1:
        raise ValueError("Bin counts must be positive.")
    summary = run(
        args.data,
        args.output_dir,
        config_path=args.config,
        radial_edges=radial_edges,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        n_velocity_bins=int(args.n_velocity_bins),
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote input-data validation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
