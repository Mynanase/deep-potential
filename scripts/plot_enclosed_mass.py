#!/usr/bin/env python
"""Enclosed-mass comparison: models vs particle truth, with confidence bands.

Generic rewrite of the auriga enclosed-mass display, reduced to the two
data kinds that matter routinely: trained Phi models (flux mass M(<r) from
the sphere-averaged radial gradient, validate_enclosed_mass machinery) and
the particle truth (shell-mass table from the grid product, or the legacy
60-shell hdf5).

Figure layout (top / bottom, shared log-r axis):
  top     M(<r): truth line with its Poisson band, model lines with
          bootstrap confidence intervals (resampled Sobol directions);
  bottom  relative mass error (M_model - M_truth)/M_truth in %, zero line,
          +-5% reference band, model CIs as error bars.

Truth-side options: --truth-grids (grid product h5; the shell table is
built or reused through auriga/truth_products.build-shell-mass, --cache
controls where it lands) or --truth-60 (legacy 60-shell hdf5, M binds to
outer edges).  Model side: M_flux evaluated at the truth radial edges via
sobol_directions; the CI bootstrap-resamples directions (--n-boot draws,
--ci percentile).

Data-level conventions follow the rbin/adjudication family: r in kpc,
masses in Msun, figures + npz to --fig-dir.

Run from the repo root:
  python scripts/plot_enclosed_mass.py \
      --input data/auriga/halo12.h5 \
      --model S1=runs/halo12-cap-w1024 \
      --truth-grids data/auriga/halo12_particle_truth_grids.h5 \
      --fig-dir runs/enclosed-mass/figs
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "scripts"
AURIGA = HERE / "auriga"
for p in (str(HERE), str(AURIGA)):
    if p not in sys.path:
        sys.path.insert(0, p)

from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, sobol_directions, m_flux_at_radius, make_radial_nodes)
import orx_figstyle as ofs  # noqa: E402
from truth_products import shell_mass_from_grids, _cache_guard, \
    _open_with_lineage, SHELL_SCHEMA, sha256_file  # noqa: E402


def load_units(input_fname):
    with h5py.File(input_fname, "r") as f:
        attrs = dict(f.attrs)
    return (float(attrs["length_scale_kpc"]),
            float(attrs["velocity_scale_kms"]))


def load_phi_f32(run_dir):
    import jax
    import fit_all
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        model = fit_all.load_potential(Path(run_dir) / "models" / "Phi",
                                       checkpoint_index=-1)
        return model.phi_model
    finally:
        jax.config.update("jax_enable_x64", prev)


def m_flux_profile(phi, r_nodes, dirs, L, V, chunk=128):
    """M_flux(<r) at each node; per-direction values for the bootstrap."""
    import jax
    import jax.numpy as jnp
    per_dir = np.empty((r_nodes.size, dirs.shape[0]))
    # per-direction M: same formula without the direction mean
    for j, r in enumerate(r_nodes):
        q = dirs * (r / L)
        dn = jax.vmap(lambda q_, n_: jnp.dot(jax.grad(phi)(q_), n_))
        vals = np.asarray(dn(jnp.asarray(q), jnp.asarray(dirs)))
        per_dir[j] = vals * r ** 2 * V ** 2 / (G_KPC_KMS2_MSUN * L)
    return per_dir.mean(axis=1), per_dir


def bootstrap_ci(per_dir, n_boot, ci, rng):
    """Percentile CI of the direction-mean at each node via direction resampling."""
    n_dir = per_dir.shape[1]
    out = np.empty((per_dir.shape[0], 2))
    alpha = (1.0 - ci) / 2.0
    idx = rng.integers(0, n_dir, size=(n_boot, n_dir))
    means = per_dir[:, idx].mean(axis=2)  # (n_nodes, n_boot)
    out[:, 0] = np.percentile(means, 100 * alpha, axis=1)
    out[:, 1] = np.percentile(means, 100 * (1 - alpha), axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="training/eta h5 (attrs carry the physical units)")
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=RUN_DIR")
    ap.add_argument("--truth-grids", type=Path, default=None,
                    help="particle-truth grid product h5")
    ap.add_argument("--truth-60", type=Path, default=None,
                    help="legacy 60-shell truth hdf5 (M binds to outer edges)")
    ap.add_argument("--r-inner", type=float, default=1.0906,
                    help="inner edge of the comparison domain [kpc]")
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--ci", type=float, default=0.68,
                    help="confidence level (0.68 -> 16-84 percentiles)")
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--cache", type=Path,
                    default=Path("data/auriga/halo12_shell_mass.h5"),
                    help="shell-mass product path (truth_products build-shell-mass)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--fig-fmt", default="png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax

    if (args.truth_grids is None) == (args.truth_60 is None):
        ap.error("give exactly one of --truth-grids / --truth-60")

    t_start = time.time()
    L, V = load_units(args.input)
    print(f"units from {args.input}: L={L} kpc, V={V} km/s")

    # ---------------- truth side ----------------------------------------
    if args.truth_grids is not None:
        r_edges = np.unique(np.concatenate((
            [args.r_inner, args.r_outer],
            np.geomspace(args.r_inner, args.r_outer, 21))))
        r_edges = np.sort(r_edges[(r_edges >= args.r_inner)
                                  & (r_edges <= args.r_outer)])
        lineage = {"grids_sha256": sha256_file(args.truth_grids),
                   "r_edges": r_edges.tolist(),
                   "r_nodes": r_edges.tolist()}  # nodes == edges here
        if _cache_guard(args.cache, SHELL_SCHEMA, lineage, args.force):
            with h5py.File(args.cache, "r") as f:
                r_q = np.asarray(f["r_nodes"][:])
                m_cum = np.asarray(f["M_cum"][:])
                m_err = np.asarray(f["M_cum_err"][:])
        else:
            tab = shell_mass_from_grids(args.truth_grids, r_edges, r_edges)
            m_cum, m_err = tab["M_cum"], tab["M_cum_err"]
            r_q = r_edges
            with _open_with_lineage(args.cache, SHELL_SCHEMA, lineage) as f:
                f.attrs.update({
                    "source_grids": str(args.truth_grids),
                    "M_total_msun": tab["M_total"],
                    "n_particles": tab["n_particles"],
                    "method": "rho3d cell-centre binning; Poisson CI from "
                              "effective counts",
                })
                f["r_edges"] = tab["r_edges"]
                f["M_shell"] = tab["M_shell"]
                f["M_shell_err"] = tab["M_shell_err"]
                f["N_eff_shell"] = tab["N_eff_shell"]
                f["r_nodes"] = tab["r_nodes"]
                f["M_cum"] = tab["M_cum"]
                f["M_cum_err"] = tab["M_cum_err"]
            print(f"wrote {args.cache} ({args.cache.stat().st_size/1e6:.2f} MB)")
        m_true = m_cum - m_cum[0]           # Delta M(r; r_inner)
        m_true_err = np.sqrt(np.maximum(m_err ** 2 - m_err[0] ** 2, 0.0))
        truth_label = "truth (grid product)"
    else:
        from validate_enclosed_mass import load_truth, truth_delta_mass
        truth = load_truth(args.truth_60)
        r_q, dM = truth_delta_mass(truth, args.r_inner)
        keep = r_q <= args.r_outer
        r_q, m_true = r_q[keep], dM[keep]
        m_true_err = np.full_like(m_true, np.nan)  # no CI from the 60-shell
        truth_label = "truth (60-shell)"

    # ---------------- model side ----------------------------------------
    models = []
    for spec in args.model:
        label, _, run_dir = spec.partition("=")
        models.append((label.strip(), Path(run_dir.strip())))

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    rng = np.random.default_rng(args.boot_seed)
    prof, lo, hi = {}, {}, {}
    for label, run_dir in models:
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        m_nodes, per_dir = m_flux_profile(phi, r_q, dirs, L, V)
        ci = bootstrap_ci(per_dir, args.n_boot, args.ci, rng)
        prof[label] = m_nodes - m_nodes[0]
        lo[label] = ci[:, 0] - m_nodes[0]
        hi[label] = ci[:, 1] - m_nodes[0]
        print(f"[{label}] flux profile on {r_q.size} nodes "
              f"({time.time()-t0:.0f}s); CI {args.ci:.0%} over "
              f"{args.n_boot} bootstraps")
        del phi
        jax.clear_caches()

    # ---------------- figure ---------------------------------------------
    ofs.use_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = ofs.figure_grid(2, 1, width=ofs.TEXT, ratio=0.78)
    ax1.errorbar(r_q, m_true, yerr=m_true_err, color="0.2", lw=1.6,
                 label=truth_label)
    for j, (label, _) in enumerate(models):
        c = ofs.CYCLE[j % len(ofs.CYCLE)]
        ax1.plot(r_q, prof[label], color=c, lw=1.2, label=label)
        ax1.fill_between(r_q, lo[label], hi[label], color=c, alpha=0.18, lw=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("r [kpc]")
    ax1.set_ylabel(r"$\Delta M(r;\,r_{\rm in})$ [$M_\odot$]")
    ax1.legend(frameon=False, fontsize=7)

    ax2.axhline(0.0, color="0.2", lw=0.8)
    ax2.axhspan(-5.0, 5.0, color="0.92", zorder=0)
    for j, (label, _) in enumerate(models):
        c = ofs.CYCLE[j % len(ofs.CYCLE)]
        rel = 100.0 * (prof[label] - m_true) / m_true
        rel_lo = 100.0 * (lo[label] - m_true) / m_true
        rel_hi = 100.0 * (hi[label] - m_true) / m_true
        yerr = np.stack([rel - rel_lo, rel_hi - rel])
        ax2.errorbar(r_q[1:], rel[1:], yerr=yerr[:, 1:], color=c, lw=1.2,
                     marker="o", ms=2.5, label=label)
    ax2.set_xscale("log")
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel("relative error of $\\Delta M$  [%]")
    ax2.set_xlim(r_q[1] * 0.9, r_q[-1] * 1.1)
    ofs.panel_labels([ax1, ax2])
    stem = args.fig_dir / "enclosed_mass"
    for fmt in args.fig_fmt.split(","):
        path = f"{stem}.{fmt.strip()}"
        fig.savefig(path, dpi=200 if fmt.strip() == "png" else None)
        print("saved", path)

    np.savez(args.fig_dir / "enclosed_mass.npz",
             r_nodes=r_q, m_true=m_true, m_true_err=m_true_err,
             **{f"{l}_{k}": v for l, _ in models
                for k, v in (("m_flux", prof[l]), ("lo", lo[l]), ("hi", hi[l]))})
    with open(args.fig_dir / "summary.json", "w") as f:
        json.dump({
            "models": [l for l, _ in models],
            "truth": str(args.truth_grids or args.truth_60),
            "r_inner": args.r_inner, "r_outer": args.r_outer,
            "n_dirs": args.n_dirs, "n_boot": args.n_boot, "ci": args.ci,
            "median_rel_pct_last_half": {
                l: float(np.median(
                    100.0 * (prof[l][r_q > r_q[r_q.size // 2]]
                             - m_true[r_q > r_q[r_q.size // 2]])
                    / m_true[r_q > r_q[r_q.size // 2]]))
                for l, _ in models},
        }, f, indent=1)
    print(f"PLOT_ENCLOSED_MASS_DONE total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
