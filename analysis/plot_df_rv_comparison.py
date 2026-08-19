"""Radial-speed (r, v) comparison: analytic Plummer reference vs model samples.

Uses the project's plot_radial_speed_comparison with the exact analytic
probability mass from plummer_rv_ideal_grid and the E = 0 escape curve.

Usage:
  # GPU (full model sampling):
  python -u analysis/plot_df_rv_comparison.py --df-run runs/plummer_rcut/full/trial_00/df
  # CPU validation (real data as "model" samples):
  python -u analysis/plot_df_rv_comparison.py --data data/plummer_n524288_train.h5 --subsample 131072
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/localdisk/kosmos/my-deep-potential")

import matplotlib
matplotlib.use("Agg")

ROOT = Path("/localdisk/kosmos/my-deep-potential")
DATA = ROOT / "data/plummer_n524288_train.h5"
DF_RUN = ROOT / "runs/plummer_rcut/full/trial_00/df"
OUT = ROOT / "runs/plummer_rcut/full/eval/plots/compare"
OUT.mkdir(parents=True, exist_ok=True)

R_LIM = (0.0, 5.0)
V_LIM = (0.0, 1.5)
BINS = (48, 48)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-run", type=str, default=None,
                        help="Path to a trained DF run dir; if given, sample it as the model.")
    parser.add_argument("--data", type=str, default=str(DATA),
                        help="Path to real data h5 (used as reference fallback).")
    parser.add_argument("--subsample", type=int, default=131_072,
                        help="Number of samples to use.")
    parser.add_argument("--tag", type=str, default="full",
                        help="Output tag, e.g. 'full' or 'cut'.")
    args = parser.parse_args()

    from experiments.validation.plummer import plummer_rv_ideal_grid

    ideal = plummer_rv_ideal_grid(R_LIM, V_LIM, BINS)
    probability_mass = ideal["probability_mass"]  # (N_r, N_v)

    if args.df_run is not None:
        import jax

        from dpjax.flows.api import sample_apply
        from experiments.workflows.artifacts import load_df

        print(f"[rv] loading DF from {args.df_run}")
        df_model, df_params, normalizer, df_cfg, _ = load_df(args.df_run)
        key = jax.random.PRNGKey(42)
        n = args.subsample
        eta_std = np.asarray(
            sample_apply(df_model, df_params, key, n, flow_cfg=df_cfg.get("flow")),
            dtype=np.float32,
        )
        model_eta = np.asarray(normalizer.inverse(eta_std), dtype=np.float64)
        model_label = "DF model sample"
    else:
        from experiments.datasets.phase_space import load_eta_h5

        print(f"[rv] loading real data from {args.data}")
        real = load_eta_h5(args.data)
        rng = np.random.default_rng(0)
        idx = rng.choice(real.shape[0], size=min(args.subsample, real.shape[0]), replace=False)
        model_eta = real[idx].astype(np.float64)
        model_label = "data sample (validation)"

    r_curve = np.linspace(max(R_LIM[0], 1e-3), R_LIM[1], 300)
    v_esc = np.sqrt(2.0 / np.sqrt(1.0 + r_curve**2))
    escape_curve = (r_curve, v_esc)

    from experiments.plotting.df_diagnostics import plot_radial_speed_comparison

    fig = plot_radial_speed_comparison(
        reference_eta=None,
        model_eta=model_eta,
        reference_probability_mass=probability_mass,
        radius_range=R_LIM,
        speed_range=V_LIM,
        bins=BINS,
        escape_curve=escape_curve,
        reference_label="Analytic Plummer",
        model_label=model_label,
        radius_label=r"$r$",
        speed_label=r"$v$",
        unbound_label=r"$E>0$",
        title=f"Radial-speed plane ({args.tag}): analytic Plummer vs model",
    )
    out_path = OUT / f"df_rv_comparison_{args.tag}.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[rv] wrote {out_path}")


if __name__ == "__main__":
    main()