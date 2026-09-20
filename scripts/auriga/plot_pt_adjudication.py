#!/usr/bin/env python
"""Flux enclosed-mass and outer angular negative-density vs PARTICLE truth.

Grid-prior pt-adjudication: the three decision-relevant w1024 clean+smooth
Phis - base (production co-winner), S1 (gridprior parent) and gridprior
(run 09faad30) - against the particle-monopole truth of the grid product.
Methods transplanted unchanged from the frozen pt-adjudication line (node
fd47ea26, commit f892ffc):
  * Gauss-flux dM(r) band adjudication from compare_phi_truth.py (nodes
    532eada5 / b08af807): M_flux(<r) = r^2 V^2/(G L) <n.grad phi>_Omega on
    2048 Sobol dirs, annulus dM(r; 1.09 kpc), median |rel err| per band
    (2-10/10-30/30-50/50-70 kpc). Truth: M_true(<r) = vc2(r) r / G with
    vc2 = r |d<Phi_true>_Omega/dln r| (monopole theorem = G M(<r)/r).
  * Sobol-direction angular density of plot_s1_density_angular.py (node
    79117b4b) + the non-positive Laplacian diagnostic of plot_s1_density_2d
    (node c9b363df), resolved per radius: frac(rho <= 0) over directions,
    focused on the outer bands.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, load_truth, truth_delta_mass, sobol_directions,
    m_flux_at_radius, rho_from_phi)
from plot_particle_truth_2d import load_phi_f32  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "S1", "gridprior")
COLOR = {"base": "blue", "S1": "red", "gridprior": "green"}
BANDS = ((2.0, 10.0), (10.0, 30.0), (30.0, 50.0), (50.0, 70.0))
# Published 60-shell adjudication anchors, identical estimator, for
# side-by-side comparison only - not used in any computation here.
# base / S1: run 0454cb6b @86a3ef7 (node 532eada5).
# gridprior: run 2b4af62c @8747998 (node 658d75bd).
REF_60SHELL = {"base": (2.34, 6.64, 9.11, 7.46),
               "S1": (1.84, 6.54, 9.32, 11.40),
               "gridprior": (2.87, 6.86, 9.15, 7.93)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--truth", type=Path, default=None,
                    help="60-shell truth hdf5 (cross-check of the monopole)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/pt-adjudication-gridprior"))
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    t_start = time.time()
    with h5py.File(args.grids, "r") as f:
        r_nodes = np.asarray(f["potential/r_nodes"][:], dtype=float)
        phi_prof = np.asarray(f["potential/phi_sphere_mean"][:], dtype=float)
    print(f"grids product read in {time.time()-t_start:.2f}s; "
          f"{r_nodes.size} monopole nodes {r_nodes[0]:.3f}-{r_nodes[-1]:.1f} kpc")

    vc2 = np.abs(np.gradient(phi_prof, np.log(r_nodes)))
    m_true = vc2 * r_nodes / G_KPC_KMS2_MSUN
    dm_true = m_true - m_true[0]
    print(f"M_true mono: {m_true[0]:.3e} ... {m_true[-1]:.3e} Msun "
          f"(asset total 5.934e11 within 75 kpc)")

    if args.truth is not None and Path(args.truth).is_file():
        truth = load_truth(args.truth)
        edges = np.asarray(truth["r_edges"], dtype=float)
        r_in = float(edges[np.argmin(np.abs(edges - r_nodes[0]))])
        r_true, dm_60 = truth_delta_mass(truth, r_in)
        keep = r_true <= float(r_nodes[-1])
        r_true, dm_60 = r_true[keep], dm_60[keep]
        ours = np.interp(r_true, r_nodes, dm_true)
        rel = np.abs(ours / dm_60 - 1.0)
        i_max = int(np.argmax(rel))
        print(f"monopole dM vs 60-shell dM (annuli from {r_in:.3f} kpc): "
              f"median {np.median(rel)*100:.2f}%, max {rel.max()*100:.2f}% "
              f"(at r={r_true[i_max]:.2f} kpc) over {r_true.size} shell edges")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    m_flux, dm_fluxes, rel_node, frac_neg = {}, {}, {}, {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        mf = np.array([m_flux_at_radius(phi, r, dirs, L_KPC, V_KMS)
                       for r in r_nodes])
        dm_flux = mf - mf[0]
        rel = (dm_flux - dm_true) / dm_true
        m_flux[label] = mf
        dm_fluxes[label] = dm_flux
        rel_node[label] = rel
        fn = np.empty(r_nodes.size)
        for j, r in enumerate(r_nodes):
            rho_d = np.asarray(rho_from_phi(phi, dirs * (r / L_KPC),
                                            L_KPC, V_KMS))
            fn[j] = float(np.mean(rho_d <= 0.0))
        frac_neg[label] = fn
        print(f"[{label}] flux + directions in {time.time()-t0:.0f}s; "
              f"M_flux(70)/M_true(70) = {mf[-1]/m_true[-1]:.4f}; "
              f"frac(rho<=0) at r~40/60 kpc = "
              f"{fn[np.argmin(np.abs(r_nodes-40))]*100:.1f}%/"
              f"{fn[np.argmin(np.abs(r_nodes-60))]*100:.1f}%")

    print("=== median |rel err| of FLUX dM per radial band [%] "
          "(this run / published 60-shell: base,S1 = 0454cb6b, "
          "gridprior = 2b4af62c) ===")
    band_med = {}
    for bi, (lo, hi) in enumerate(BANDS):
        m = (r_nodes > lo) & (r_nodes <= hi)
        row = f"  {lo:4.0f}-{hi:4.0f} kpc ({int(m.sum())} nodes): "
        for l in MODELS:
            med = 100.0 * float(np.median(np.abs(rel_node[l][m])))
            band_med.setdefault(l, []).append(med)
            row += f"{l} {med:5.2f}/{REF_60SHELL[l][bi]:5.2f}   "
        print(row)
    print("  signed per-node rel err at r >= 45 kpc [%] "
          "(band medians above use only the nodes listed):")
    for j in range(r_nodes.size):
        if r_nodes[j] >= 45.0:
            print(f"    {r_nodes[j]:6.2f} kpc: "
                  + "  ".join(f"{l} {100*rel_node[l][j]:+6.2f}" for l in MODELS))
    print("  note: the published 60-shell runs used 37 shell edges (last "
          "67.28 kpc); their outermost edge had base +13.90 / S1 -2.44 / "
          "gridprior +5.53% (gridprior from run 2b4af62c) - the same "
          "outer-edge pattern seen here; band-median ordering is node-set "
          "sensitive at 2-3 nodes per outer band.")

    outer = r_nodes >= 30.0
    print("=== outer angular negative density (30-70 kpc nodes) ===")
    for l in MODELS:
        vals = frac_neg[l][outer]
        print(f"  {l}: frac(rho<=0) {vals.min()*100:.1f}-{vals.max()*100:.1f}% "
              f"(mean {vals.mean()*100:.1f}%); whole-domain midplane cells "
              "were 17.0/15.6/3.6% base/S1/gridprior (node 658d75bd)")

    ofs.use_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Enclosed-mass panels follow artifacts/halo12-potential/figures/
    # truth-compare.py (truth-compare-baseline-vs-s1.svg, node b08af807):
    # (a) absolute annulus mass, log-log, truth as a black line, models with
    #     small circle markers; (b) signed rel err of dM with a zero line and
    #     a +/-5% band, log x; (c) outer negative-direction fraction.
    fig, axes = ofs.figure_grid(1, 3, width=ofs.WIDE, ratio=0.36)
    ax = axes[0]
    ax.plot(r_nodes[1:], dm_true[1:], color="k", lw=1.8,
            label="particle truth", zorder=3)
    for l in MODELS:
        ax.plot(r_nodes[1:], dm_fluxes[l][1:], color=ofs.PALETTE[COLOR[l]],
                lw=1.2, marker="o", ms=2.5, label=l + " (flux)", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("annulus mass $\\Delta M$(r; 1.09 kpc)  [Msun]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]
    for l in MODELS:
        ax.plot(r_nodes[1:], 100.0 * rel_node[l][1:],
                color=ofs.PALETTE[COLOR[l]], lw=1.2, marker="o", ms=2.5,
                label=l)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axhspan(-5, 5, color="0.92", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("relative error of $\\Delta M$  [%]")
    ax.set_xlim(r_nodes[1] * 0.9, r_nodes[-1] * 1.1)
    ax = axes[2]
    ax.axvspan(30.0, 70.0, color="0.92", zorder=0)
    for l in MODELS:
        ax.plot(r_nodes, 100.0 * frac_neg[l], color=ofs.PALETTE[COLOR[l]], lw=1.3)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"directions with $\rho\leq0$ [%]")
    ax.set_ylim(0, None)
    ofs.panel_labels(list(axes))
    np.savez(args.output_dir / "pt_adjudication_gridprior.npz",
             r_nodes=r_nodes, m_true=m_true, vc2=vc2,
             dm_true=dm_true,
             bands=np.asarray(BANDS),
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("m_flux", m_flux[l]), ("rel", rel_node[l]),
                             ("frac_neg", frac_neg[l]),
                             ("dm_flux", dm_fluxes[l]))})
    for p in ofs.save(fig, str(args.output_dir / "pt-adjudication-gridprior")):
        print("saved", p)
    print(f"TOTAL WALL {time.time()-t_start:.1f}s")
    print("PT_ADJUDICATION_DONE")


if __name__ == "__main__":
    sys.exit(main())
