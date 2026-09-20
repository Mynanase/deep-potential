#!/usr/bin/env python
"""Cross-generation flux dM under ONE particle truth: width x cleaning.

Evaluates arbitrary labeled checkpoints (w128raw / w128S1raw / w128clean /
w1024raw / w1024csm ...) with the audited Gauss-flux annulus estimator
against the particle monopole M_true = vc2 r / G from the grid product.
Figure follows the truth-compare-baseline-vs-s1 style: (a) absolute dM
log-log, (b) signed rel err with a +/-5% band.
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
    G_KPC_KMS2_MSUN, sobol_directions, m_flux_at_radius)
from plot_particle_truth_2d import load_phi_f32  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
BANDS = ((2.0, 10.0), (10.0, 30.0), (30.0, 50.0), (50.0, 70.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/pt-generations"))
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
    vc2 = np.abs(np.gradient(phi_prof, np.log(r_nodes)))
    m_true = vc2 * r_nodes / G_KPC_KMS2_MSUN
    dm_true = m_true - m_true[0]
    print(f"grids read {time.time()-t_start:.2f}s; M_true(70)={m_true[-1]:.4e} Msun")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    labels, dm_fluxes, rels = [], {}, {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        mf = np.array([m_flux_at_radius(phi, r, dirs, L_KPC, V_KMS)
                       for r in r_nodes])
        dmf = mf - mf[0]
        rel = (dmf - dm_true) / dm_true
        labels.append(label)
        dm_fluxes[label] = dmf
        rels[label] = rel
        print(f"[{label}] flux in {time.time()-t0:.0f}s; "
              f"dM rel err at 20/50/70 kpc = "
              + "/".join(f"{100*rel[np.argmin(np.abs(r_nodes-r))]:+.2f}%"
                         for r in (20.0, 50.0, 70.0)))

    print("=== median |rel err| of FLUX dM per radial band [%] ===")
    header = f"{'band':>14} ({r_nodes.size} log nodes)"
    print(f"{header:>28} " + " ".join(f"{l:>11}" for l in labels))
    for lo, hi in BANDS:
        m = (r_nodes > lo) & (r_nodes <= hi)
        row = f"{lo:5.0f}-{hi:4.0f} kpc ({int(m.sum())} nodes): "
        row += " ".join(f"{100*np.median(np.abs(rels[l][m])):10.2f}" for l in labels)
        print(row)
    print("signed per-node rel err at r >= 45 kpc [%]:")
    for j in range(r_nodes.size):
        if r_nodes[j] >= 45.0:
            print(f"  {r_nodes[j]:6.2f} kpc: "
                  + "  ".join(f"{l} {100*rels[l][j]:+6.2f}" for l in labels))

    ofs.use_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = ofs.figure_grid(1, 2, width=ofs.WIDE, ratio=0.42)
    ax = axes[0]
    ax.plot(r_nodes[1:], dm_true[1:], color="k", lw=1.8,
            label="particle truth", zorder=3)
    palette = ["blue", "orange", "red", "green", "purple", "cyan"]
    for i, l in enumerate(labels):
        ax.plot(r_nodes[1:], dm_fluxes[l][1:], color=ofs.PALETTE[palette[i % 6]],
                lw=1.2, marker="o", ms=2.5, label=l, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("annulus mass $\\Delta M$(r; 1.09 kpc)  [Msun]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]
    for i, l in enumerate(labels):
        ax.plot(r_nodes[1:], 100.0 * rels[l][1:],
                color=ofs.PALETTE[palette[i % 6]], lw=1.2, marker="o",
                ms=2.5, label=l)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axhspan(-5, 5, color="0.92", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("relative error of $\\Delta M$  [%]")
    ax.set_xlim(r_nodes[1] * 0.9, r_nodes[-1] * 1.1)
    ofs.panel_labels(list(axes))
    np.savez(args.output_dir / "pt_generations.npz",
             r_nodes=r_nodes, m_true=m_true, dm_true=dm_true,
             **{f"{l}_{k}": v for l in labels
                for k, v in (("dm_flux", dm_fluxes[l]), ("rel", rels[l]))})
    for p in ofs.save(fig, str(args.output_dir / "pt-generations")):
        print("saved", p)
    print(f"TOTAL WALL {time.time()-t_start:.1f}s")
    print("PT_GENERATIONS_DONE")


if __name__ == "__main__":
    sys.exit(main())
