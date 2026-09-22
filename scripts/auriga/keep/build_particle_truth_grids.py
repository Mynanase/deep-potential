#!/usr/bin/env python
"""Build the particle-truth grid data product (density + potential, one h5).

Source chain: Gadget snapshot halo12_raw/snapdir_127 (~2.6 GB, Mpc/h units)
  -> halo12_total_matter_particles_starframe.h5 (node f411ae24: FoF GroupPos
     centre + reflection rotation into the stellar principal-axis frame,
     PartType0/1/4, r < 75 kpc, validated vs the 60-shell truth to 3.2% in
     8-75 kpc)
  -> THIS product: small gridded h5 committed under data/auriga/ so later
     figures never reload 3.9M particles or redo GPU direct summation.

Everything here is either a plain histogram (density: no integration, no
smoothing kernel; cell/shell volumes are exact geometry) or direct-summation
potential on fixed grids (phi_direct, unit-calibrated in run e1801a91).
Potential grid parameters reproduce the handoff figure exactly: n_grid=240
y=0 slice over [-70,70]^2, Sobol(20260917) 2048 directions, 24 radial nodes
make_radial_nodes(1.0906, 70, 23), zero convention Phi_sphere_mean(70)=0.
"""
import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, sobol_directions, make_radial_nodes, load_truth)
from particle_truth import load_particles, phi_direct  # noqa: E402

SCHEMA = "dpjax.particle-truth-grids.v1"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_by_type(asset):
    out = {}
    with h5py.File(asset, "r") as f:
        for ptype in ("PartType0", "PartType1", "PartType4"):
            out[ptype] = (int(f[ptype]["mass"].shape[0]),
                          float(f[ptype]["mass"][:].sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--truth", type=Path, default=None,
                    help="60-shell truth hdf5 for the radial cross-check")
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-sphere-nodes", type=int, default=24)
    ap.add_argument("--rho3d-n", type=int, default=96)
    ap.add_argument("--rho3d-extent", type=float, default=75.0)
    ap.add_argument("--slab-n", type=int, default=480)
    ap.add_argument("--slab-half-width", type=float, default=0.5)
    ap.add_argument("--rho-r-nbins", type=int, default=240)
    ap.add_argument("--rho-r-min", type=float, default=0.5)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"asset: {args.asset}")
    t0 = time.time()
    asset_sha = sha256_file(args.asset)
    print(f"asset sha256: {asset_sha} ({time.time()-t0:.0f}s)")
    by_type = count_by_type(args.asset)
    for k, (n, m) in by_type.items():
        print(f"  {k}: n={n}, M={m:.6e} Msun")

    xyz, m = load_particles(args.asset)
    print(f"particles loaded: n={m.size}, M={m.sum():.6e} Msun")

    print("--- unit calibration (single particle anchor) ---")
    one_xyz = np.array([[10.0, 0.0, 0.0]])
    one_m = np.array([1.0e5])
    v = float(phi_direct(one_xyz, one_m, np.array([[0.0, 0.0, 0.0]]))[0])
    expect = -G_KPC_KMS2_MSUN * 1.0e5 / 10.0
    print(f"single particle m=1e5 at 10 kpc, query origin: {v} expected {expect}")
    assert abs(v - expect) <= 1e-6 * abs(expect), "unit calibration failed"

    print("=== potential grids (direct summation, handoff parameters) ===")
    r_anchor = 1.0906
    r_nodes = np.asarray(make_radial_nodes(r_anchor, args.r_outer,
                                           args.n_sphere_nodes - 1), dtype=float)
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    t0 = time.time()
    q_sph = (dirs[None, :, :] * r_nodes[:, None, None]).reshape(-1, 3)
    phi_true_nodes = phi_direct(xyz, m, q_sph).reshape(r_nodes.size, -1)
    phi_70 = float(phi_direct(xyz, m, dirs * args.r_outer).mean())
    phi_nodes = phi_true_nodes - phi_70
    phi_prof = phi_nodes.mean(axis=1)
    sig = {}
    for r in (5.0, 20.0):
        row = phi_nodes[np.argmin(np.abs(r_nodes - r))]
        sig[r] = float(row.std() / abs(row.mean()))
    print(f"sphere means on {r_nodes.size}x{args.n_dirs} dirs in {time.time()-t0:.0f}s; "
          f"Phi_true(1.09)={phi_prof[0]:.1f} (handoff e1801a91: -198204.6); "
          f"sigma_Omega at r=5/20 kpc = {sig[5.0]:.3e}/{sig[20.0]:.3e} "
          "(handoff: 3.03e-02/2.54e-02)")

    xs = np.linspace(-args.r_outer, args.r_outer, args.n_grid)
    xx, zz = np.meshgrid(xs, xs, indexing="xy")
    q_slice = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])
    t0 = time.time()
    phi_slice = (phi_direct(xyz, m, q_slice).reshape(xx.shape) - phi_70)
    print(f"slice {args.n_grid}^2 in {time.time()-t0:.0f}s; "
          f"range [{phi_slice.min():.1f}, {phi_slice.max():.1f}] (km/s)^2")

    print("=== density grids (plain histograms, no integration/smoothing) ===")
    ext = args.rho3d_extent
    n3 = args.rho3d_n
    t0 = time.time()
    H3, edges3 = np.histogramdd(xyz, bins=n3, range=[(-ext, ext)] * 3, weights=m)
    cell3 = (2 * ext / n3) ** 3
    rho3d = (H3 / cell3).astype(np.float32)
    fill = float(np.mean(H3 > 0))
    print(f"rho3d {n3}^3 over [-{ext},{ext}]^3 (cell {2*ext/n3:.4f} kpc) in "
          f"{time.time()-t0:.0f}s; filled cells {fill*100:.1f}%; "
          f"mass closure {(rho3d.sum(dtype=np.float64)*cell3/m.sum()-1):+.2e}")

    r_p = np.linalg.norm(xyz, axis=1)
    re = np.geomspace(args.rho_r_min, ext, args.rho_r_nbins + 1)
    Hr, _ = np.histogram(r_p, bins=re, weights=m)
    shell_vol = 4.0 / 3.0 * np.pi * (re[1:] ** 3 - re[:-1] ** 3)
    rho_r = Hr / shell_vol
    centers = 0.5 * (re[1:] + re[:-1])
    print(f"rho_r {re.size-1} log bins in [{re[0]:.2f},{re[-1]:.0f}] kpc; "
          f"rho_r(8)={np.interp(8.0, centers, rho_r):.3e}, "
          f"rho_r(70)={rho_r[-1]:.3e} Msun/kpc^3")

    hw = args.slab_half_width
    sel = np.abs(xyz[:, 1]) <= hw
    Hs, esx, esz = np.histogram2d(xyz[sel, 0], xyz[sel, 2], bins=args.slab_n,
                                  range=[(-ext, ext)] * 2, weights=m[sel])
    sigma_slab = (Hs / (esx[1] - esx[0]) / (esz[1] - esz[0])).astype(np.float32)
    print(f"sigma_slab {args.slab_n}^2 over [-{ext},{ext}]^2, |y|<={hw} kpc: "
          f"{int(sel.sum())} particles; nonzero cells "
          f"{np.mean(Hs>0)*100:.1f}%; Sigma range "
          f"[{sigma_slab[sigma_slab>0].min():.2e}, {sigma_slab.max():.2e}] Msun/kpc^2")

    if args.truth is not None and Path(args.truth).is_file():
        truth = load_truth(args.truth)
        r_c = np.asarray(truth["r_center"], dtype=float)
        dm = np.asarray(truth["M_shell_total"], dtype=float)
        r_e = np.asarray(truth["r_edges"], dtype=float)
        rho_c = dm / (4.0 / 3.0 * np.pi * (r_e[1:] ** 3 - r_e[:-1] ** 3))
        ours = np.exp(np.interp(np.log(r_c), np.log(centers), np.log(rho_r)))
        rel = np.abs(ours / rho_c - 1.0)
        dom = (r_c >= 8.0) & (r_c <= 75.0)
        print(f"rho_r vs 60-shell truth in 8-75 kpc: median rel diff "
              f"{np.median(rel[dom])*100:.2f}%, max {rel[dom].max()*100:.2f}% "
              "(asset validation was 3.2%)")

    print("=== write product ===")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.attrs.update({
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_asset": str(args.asset),
            "source_sha256": asset_sha,
            "n_particles": int(m.size),
            "M_total_msun": float(m.sum()),
            "G_kpc_kms2_msun": G_KPC_KMS2_MSUN,
            "frame": "stellar principal axes (starframe asset v1)",
            "phi_zero_convention": "Phi_sphere_mean(r=70 kpc) = 0",
            "r_outer_kpc": args.r_outer,
            "n_grid": args.n_grid,
            "n_dirs": args.n_dirs,
            "sobol_seed": args.sobol_seed,
            "n_sphere_nodes": args.n_sphere_nodes,
            "rho3d_n": n3,
            "rho3d_extent_kpc": ext,
            "slab_n": args.slab_n,
            "slab_half_width_kpc": hw,
            "rho_r_nbins": int(re.size - 1),
        })
        for ptype, (n, mm) in by_type.items():
            f.attrs[f"{ptype}_n"] = n
            f.attrs[f"{ptype}_mass_msun"] = mm
        g = f.create_group("potential")
        g["phi_slice"] = phi_slice
        g["x_kpc"] = xs
        g["r_nodes"] = r_nodes
        g["phi_sphere_mean"] = phi_prof
        g["phi_dirs"] = phi_true_nodes.astype(np.float32)
        g["phi_70_mean"] = phi_70
        g["sigma_omega_rel_5kpc"] = sig[5.0]
        g["sigma_omega_rel_20kpc"] = sig[20.0]
        g = f.create_group("density")
        g["rho3d"] = rho3d
        g["rho3d_edges_kpc"] = edges3
        g["rho_r"] = rho_r
        g["rho_r_edges_kpc"] = re
        g["sigma_slab"] = sigma_slab
        g["sigma_slab_edges_x_kpc"] = esx
        g["sigma_slab_edges_z_kpc"] = esz
    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size/1e6:.2f} MB)")

    print("=== read-back verification ===")
    with h5py.File(args.output, "r") as f:
        assert f.attrs["schema"] == SCHEMA
        assert f["potential/phi_slice"].shape == (args.n_grid, args.n_grid)
        assert f["density/rho3d"].shape == (n3, n3, n3)
        rb = float(f["potential/phi_sphere_mean"][0])
        rb_mass = float(np.asarray(f["density/rho3d"][:], dtype=np.float64).sum()
                        * cell3)
    print(f"read-back phi_sphere_mean[0]={rb:.1f} (in-memory {phi_prof[0]:.1f}); "
          f"rho3d mass closure vs M_total: {(rb_mass/m.sum()-1):+.2e}")
    assert abs(rb - phi_prof[0]) <= 1e-9 * abs(phi_prof[0])
    print("PRODUCT_VERIFIED")
    print("BUILD_DONE")


if __name__ == "__main__":
    sys.exit(main())
