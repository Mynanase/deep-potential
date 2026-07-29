"""Render single-model or ensemble DF diagnostics from eval outputs.

Reads the ``auriga_df_metrics.json`` + ``auriga_df_diagnostics.npz`` produced
by :mod:`experiments.eval_auriga_df` and writes the following figures to the
evaluation directory (or a user-specified ``--out-dir``):

* ``density_profile.png``      – radial ρ(r) data vs flow samples + fractional error
* ``score_consistency.png``    – per-dim MAD bars + pairwise cosine histogram
* ``score_per_dim_hist.png``   – per-dimension score distribution across seeds
* ``df_ensemble_training.png`` – 4-seed train/val NLL overlay (when --run-dir is provided)
* ``df_spatial_rz_by_phi.png`` – stellar-tracer R-z density by azimuth wedge
* ``velocity_marginals_by_*.png`` – data/model spherical-velocity histograms

Examples
--------
    python -m experiments.plot_auriga_df \
        --eval-dir runs/halo_12/df_ffjord_v23_mass/ensemble_evaluation \
        --run-dir  runs/halo_12/df_ffjord_v23_mass/seed_42 \
        --run-dir  runs/halo_12/df_ffjord_v23_mass/seed_43 \
        --run-dir  runs/halo_12/df_ffjord_v23_mass/seed_44 \
        --run-dir  runs/halo_12/df_ffjord_v23_mass/seed_45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpjax.plotting import plot_auriga_df_ensemble


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Auriga DF diagnostics from eval outputs.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory containing auriga_df_metrics.json and "
        "auriga_df_diagnostics.npz (output of experiments.eval_auriga_df).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to save figures; defaults to --eval-dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        action="append",
        default=None,
        help="Per-seed DF training run directory (for training-curve overlay). "
        "Pass one --run-dir per seed; order is preserved.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="auriga_df_metrics.json",
    )
    parser.add_argument(
        "--diagnostics-name",
        type=str,
        default="auriga_df_diagnostics.npz",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--fig-fmt",
        type=str,
        action="append",
        default=None,
        help="Output format(s), e.g. png or pdf; defaults to png.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    metrics_path = eval_dir / args.metrics_name
    diagnostics_path = eval_dir / args.diagnostics_name
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing {diagnostics_path}")

    result = plot_auriga_df_ensemble(
        metrics_json=metrics_path,
        diagnostics_npz=diagnostics_path,
        fig_dir=args.out_dir or eval_dir,
        run_dirs=args.run_dir,
        fig_fmt=tuple(args.fig_fmt or ["png"]),
        dpi=args.dpi,
    )

    print("Wrote DF diagnostic figures:")
    for key, val in result.items():
        print(f"  {key}: {val}")

    # Re-print the metrics JSON for convenience.
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
