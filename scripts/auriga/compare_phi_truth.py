#!/usr/bin/env python
"""Truth comparison: enclosed-mass profiles of frozen Phi models vs the
Auriga Halo12 total-matter truth (60-shell HDF5).

Judgement design:
  * annulus masses dM(r; r_anchor) with r_anchor = first truth shell edge
    >= r_inner_min (default 1 kpc), so no untrained interior extrapolation
    enters either side of the comparison;
  * primary estimator: FLUX path  M(<r) = r^2 V^2 / (G L) <n.grad_q phi>_Omega
    (Gauss theorem; no second-derivative noise);
  * secondary estimator: RHO path  4 pi int s^2 <rho>_Omega ds (log-r trapezoid);
  * quadrature, unit conventions and truth handling are REUSED from
    validate_enclosed_mass.py (covered by its analytic tests);
  * models are compared at every truth shell edge up to r_outer.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, load_truth, truth_delta_mass, sobol_directions,
    make_radial_nodes, angular_mean_at_nodes, cumulative_volume_mass,
    m_flux_at_radius, rho_from_phi)

L_KPC, V_KMS = 10.0, 100.0  # fixed by the Halo12 adapter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="label=run_dir (run_dir contains models/Phi)")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--r-inner-min", type=float, default=1.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--nodes-per-interval", type=int, default=8)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--phi-ckpt", type=int, default=-1)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/truth_compare"))
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import fit_all  # noqa: F401  (path setup done by the import above)

    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"constants: G={G_KPC_KMS2_MSUN} L={L_KPC} kpc V={V_KMS} km/s")

    truth = load_truth(args.truth)
    edges = np.asarray(truth["r_edges"], dtype=float)
    r_in = float(edges[edges >= args.r_inner_min][0])
    edges_in = edges[(edges > r_in) & (edges <= args.r_outer)]
    nodes = make_radial_nodes(r_in, args.r_outer, args.nodes_per_interval,
                              boundaries=edges_in.tolist())
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    r_true, dM_true_full = truth_delta_mass(truth, r_in)
    dM_true = np.interp(edges_in, r_true, dM_true_full)
    print(f"anchor r={r_in:.4f} kpc; {len(edges_in)} truth edges to "
          f"{args.r_outer} kpc; {len(nodes)} radial nodes x {args.n_dirs} dirs")

    def load_phi_f32(run_dir, ckpt):
        # checkpoints are f32; loading under x64 makes the empty model f64 and
        # eqx refuses the dtype change. Load with x64 off, restore afterwards
        # (same pattern as validate_enclosed_mass._load_phi_smoke).
        prev = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", False)
        try:
            model = fit_all.load_potential(Path(run_dir) / "models" / "Phi",
                                           checkpoint_index=ckpt)
            return model.phi_model
        finally:
            jax.config.update("jax_enable_x64", prev)

    labels, results = [], {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        labels.append(label)
        phi = load_phi_f32(run_dir, args.phi_ckpt)
        t0 = time.time()
        m_flux = np.array([m_flux_at_radius(phi, float(r), dirs, L_KPC, V_KMS)
                           for r in nodes])
        dM_flux = m_flux - m_flux[0]
        rho_mean = angular_mean_at_nodes(
            lambda q: rho_from_phi(phi, q, L_KPC, V_KMS), nodes, dirs, L_KPC)
        dM_rho = cumulative_volume_mass(nodes, rho_mean, r_in)
        results[label] = dict(
            dM_flux=np.interp(edges_in, nodes, dM_flux),
            dM_rho=np.interp(edges_in, nodes, dM_rho))
        print(f"[{label}] {len(nodes)} nodes evaluated in {time.time()-t0:.0f}s")

    def rel(x):
        return 100.0 * (x - dM_true) / np.abs(dM_true)

    print()
    print("=== dM(r; {:.2f} kpc) at truth shell edges  [Msun] ===".format(r_in))
    header = f"{'r_kpc':>7} {'truth':>13} " + " ".join(
        f"{l+'_flux':>13} {'rel%':>7} {l+'_rho':>13}" for l in labels)
    print(header)
    for i, r in enumerate(edges_in):
        row = f"{r:7.2f} {dM_true[i]:13.4e} "
        for l in labels:
            row += (f"{results[l]['dM_flux'][i]:13.4e} "
                    f"{rel(results[l]['dM_flux'][i])[i]:7.2f} "
                    f"{results[l]['dM_rho'][i]:13.4e}")
        print(row)

    bands = [(2.0, 10.0), (10.0, 30.0), (30.0, 50.0), (50.0, 70.0)]
    print()
    print("=== median |rel err| of FLUX dM per radial band [%] ===")
    print(f"{'band':>12} " + " ".join(f"{l:>12}" for l in labels))
    summary = {}
    for lo, hi in bands:
        m = (edges_in >= lo) & (edges_in < hi)
        if not m.any():
            continue
        vals = {}
        for l in labels:
            vals[l] = float(np.median(np.abs(rel(results[l]["dM_flux"])[m])))
            summary.setdefault(l, {})[f"{lo:g}-{hi:g}"] = vals[l]
        print(f"{lo:5.0f}-{hi:3.0f} " + " ".join(f"{vals[l]:12.2f}" for l in labels))
    outer = "50-70"
    if all(outer in summary.get(l, {}) for l in labels):
        best = min(labels, key=lambda l: summary[l][outer])
        print(f"OUTER VERDICT ({outer} kpc, flux): best = {best}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = dict(anchor_kpc=r_in, r_edges=edges_in.tolist(),
               dM_true=dM_true.tolist(),
               models={l: {k: v.tolist() for k, v in results[l].items()} for l in labels},
               median_abs_rel_flux_per_band=summary)
    with open(args.output_dir / "truth_compare.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.output_dir / 'truth_compare.json'}")
    print("TRUTH_COMPARE_DONE")


if __name__ == "__main__":
    sys.exit(main())

