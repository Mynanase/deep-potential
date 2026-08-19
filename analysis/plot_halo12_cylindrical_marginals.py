"""Halo12 cylindrical-coordinate DF marginals binned by cylindrical radius R.

Layout: one row per R bin, five columns for the five non-R components
(phi, z, vR, vphi, vz).  Each panel shows the weighted marginal of that
component for the bin, real data (solid black) vs the best-fit trained DF
sample (dashed colored).

Best-fit model: runs/halo_12/df_ffjord_v23_mass/seed_43  (from the ensemble
evaluation, model_labels = ['seed_43']).

Usage:
  # full (needs GPU for FFJORD sampling):
  python -u analysis/plot_halo12_cylindrical_marginals.py
  # CPU validation (real data subsampled twice):
  python -u analysis/plot_halo12_cylindrical_marginals.py --cpu-validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/localdisk/kosmos/my-deep-potential")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/localdisk/kosmos/my-deep-potential")
DATA = ROOT / "data/auriga/halo12_all_mass.h5"
DF_RUN = ROOT / "runs/halo_12/df_ffjord_v23_mass/seed_43"
OUT = ROOT / "runs/halo_12/df_ffjord_v23_mass/seed_43/plots/df_marginals_cyl"
OUT.mkdir(parents=True, exist_ok=True)

COMPONENTS = [r"$\phi$", "$z$", "$v_R$", "$v_\phi$", "$v_z$"]
COL_INDICES = [1, 2, 3, 4, 5]  # cylindrical phase-space column order


def cartesian_to_cylindrical_phase_space(eta: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,vx,vy,vz] rows to [R, phi, z, vR, vphi, vz].

    ``phi`` is the azimuth in [-pi, pi].  The velocity basis is orthonormal.
    """
    eta = np.asarray(eta, dtype=np.float64)
    x, y, z, vx, vy, vz = eta.T
    cylindrical_radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    v_r = vx * cos_phi + vy * sin_phi
    v_phi = -vx * sin_phi + vy * cos_phi
    return np.column_stack(
        [cylindrical_radius, phi, z, v_r, v_phi, vz]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DATA))
    parser.add_argument("--df-run", type=str, default=str(DF_RUN))
    parser.add_argument("--n-model", type=int, default=262_144,
                        help="Number of DF samples to draw (GPU).")
    parser.add_argument("--n-bins", type=int, default=11,
                        help="Number of quantile R bins (rows).")
    parser.add_argument("--n-real", type=int, default=400_000,
                        help="Max real particles used (weighted).")
    parser.add_argument("--n-bins-hist", type=int, default=64,
                        help="Histogram bins per component.")
    parser.add_argument("--cpu-validation", action="store_true",
                        help="Use real data as both reference and 'model'.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from experiments.datasets.phase_space import load_eta_h5, load_h5_vector

    print("[cyl] loading real data ...")
    real = load_eta_h5(args.data)
    weights = load_h5_vector(args.data, "tracer_weight")
    rng = np.random.default_rng(args.seed)
    if real.shape[0] > args.n_real:
        idx = rng.choice(real.shape[0], size=args.n_real, replace=False)
        real = real[idx]
        weights = weights[idx]

    if args.cpu_validation:
        model = real.copy()
    else:
        import jax

        from dpjax.flows.api import sample_apply
        from experiments.workflows.artifacts import load_df

        print(f"[cyl] loading best-fit DF from {args.df_run} ...")
        df_model, df_params, normalizer, df_cfg, _ = load_df(args.df_run)
        print(f"[cyl] sampling {args.n_model} points ...")
        key = jax.random.PRNGKey(args.seed)
        eta_std = np.asarray(
            sample_apply(
                df_model,
                df_params,
                key,
                args.n_model,
                flow_cfg=df_cfg.get("flow"),
            ),
            dtype=np.float32,
        )
        model = np.asarray(normalizer.inverse(eta_std), dtype=np.float64)

    real_cyl = cartesian_to_cylindrical_phase_space(real)
    model_cyl = cartesian_to_cylindrical_phase_space(model)
    print("[cyl] real R:", real_cyl[:, 0].min(), real_cyl[:, 0].max(),
          "model R:", model_cyl[:, 0].min(), model_cyl[:, 0].max())

    # quantile R bins (same edges for both)
    r_edges = np.quantile(
        np.concatenate([real_cyl[:, 0], model_cyl[:, 0]]),
        np.linspace(0.0, 1.0, args.n_bins + 1),
    )
    r_edges[0] = -np.inf
    r_edges[-1] = np.inf
    print("[cyl] R bins:", np.round(r_edges[1:-1], 3))

    fig, axes = plt.subplots(
        args.n_bins, 5,
        figsize=(18, 2.6 * args.n_bins),
        dpi=150,
        sharex=False,
    )
    for b in range(args.n_bins):
        r_lo, r_hi = r_edges[b], r_edges[b + 1]
        r_mask = (real_cyl[:, 0] > r_lo) & (real_cyl[:, 0] <= r_hi)
        m_mask = (model_cyl[:, 0] > r_lo) & (model_cyl[:, 0] <= r_hi)
        for col, (name, idx) in enumerate(zip(COMPONENTS, COL_INDICES)):
            ax = axes[b, col]
            if r_mask.sum() == 0 or m_mask.sum() == 0:
                ax.axis("off")
                continue
            w = weights[r_mask] / weights[r_mask].sum()
            ax.hist(real_cyl[r_mask, idx], bins=args.n_bins_hist,
                    weights=w, histtype="step", color="black", lw=1.4,
                    label="real")
            ax.hist(model_cyl[m_mask, idx], bins=args.n_bins_hist,
                    density=True, histtype="step", color="C1", ls="--",
                    lw=1.2, label="DF")
            ax.grid(alpha=0.2)
            if b == 0:
                ax.set_title(name)
            if col == 0:
                r_lo_s = "" if np.isneginf(r_lo) else f"{r_lo:.2f}"
                r_hi_s = "" if np.isposinf(r_hi) else f"{r_hi:.2f}"
                ax.set_ylabel(f"R∈({r_lo_s},{r_hi_s}]")
            if b == args.n_bins - 1:
                ax.set_xlabel(name)
            if col == 4 and b == 0:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Halo12 cylindrical DF marginals per R bin "
        "(best-fit seed_43; solid = real, dashed = DF)"
    )
    fig.tight_layout()
    out_path = OUT / "df_marginals_cylindrical_by_R_row_bins.png"
    fig.savefig(out_path, bbox_inches="tight")
    print("[cyl] wrote", out_path)


if __name__ == "__main__":
    main()