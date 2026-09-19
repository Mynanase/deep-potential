#!/usr/bin/env python
"""Build the particle-level total-matter truth asset in the model frame.

1. Solve the rigid transform (center + rotation) by ID-matched Kabsch between
   the raw snapshot PartType4 and the already principal-axis-aligned
   halo_12_stars.hdf5; gate on max residual.
2. Load PartType0/1/4 from all snapshot chunks (the types in the audited
   60-shell truth), transform to the aligned frame, cut r < 75 kpc.
3. Save the asset (idempotent: verify-and-reuse an existing one).
4. Validation gate: M(<r) on the 61 truth edges vs M_cum_total within 1%,
   stellar-only vs PartType4/M_cum likewise.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

TYPES = ("PartType0", "PartType1", "PartType4")


def load_snapshot_group(snapdir, ptype, fields):
    """Concatenate a group's fields across chunk files."""
    parts = []
    for i in range(8):
        path = Path(snapdir) / f"snapshot_127.{i}.hdf5"
        with h5py.File(path, "r") as f:
            if ptype not in f:
                continue
            g = f[ptype]
            if g["ParticleIDs"].shape[0] == 0:
                continue
            cols = [np.asarray(g[k][:], dtype=np.float64) if g[k].dtype.kind == "f"
                    else np.asarray(g[k][:]) for k in fields]
            parts.append(cols)
    return [np.concatenate([p[j] for p in parts], axis=0) for j in range(len(fields))]


def kabsch(x_sim, x_al):
    """R, c such that x_al ~= R @ (x_sim - c)."""
    mu_s, mu_a = x_sim.mean(axis=0), x_al.mean(axis=0)
    H = (x_sim - mu_s).T @ (x_al - mu_a)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    c = mu_s - np.linalg.solve(R, mu_a)
    return R, c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapdir",
                    default="/localdisk/kosmos/my-deep-potential/data/halo12_raw/snapdir_127")
    ap.add_argument("--stars",
                    default="/localdisk/kosmos/my-deep-potential/data/halo_12_stars.hdf5")
    ap.add_argument("--truth",
                    default="/localdisk/kosmos/my-deep-potential/data/auriga/halo_12_total_density.hdf5")
    ap.add_argument("--output",
                    default="/localdisk/kosmos/my-deep-potential/data/auriga/halo12_total_matter_particles_starframe.h5")
    ap.add_argument("--r-max", type=float, default=75.0)
    ap.add_argument("--n-match", type=int, default=8000)
    ap.add_argument("--residual-gate", type=float, default=1e-4)
    args = ap.parse_args()

    t0 = time.time()
    print("=== STEP 1: solve rigid transform by ID-matched Kabsch ===")
    pid_snap, xyz_snap = load_snapshot_group(
        args.snapdir, "PartType4", ["ParticleIDs", "x", "y", "z"])
    print(f"snapshot PartType4: n={pid_snap.size} ({time.time()-t0:.0f}s)")
    with h5py.File(args.stars, "r") as f:
        g = f["PartType4"]
        pid_al = np.asarray(g["ParticleIDs"][:])
        xyz_al = np.stack([g["x"][:], g["y"][:], g["z"][:]], axis=1).astype(np.float64)
    common, ia, ib = np.intersect1d(pid_al, pid_snap, return_indices=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(common.size, size=min(args.n_match, common.size), replace=False)
    R, c = kabsch(xyz_snap[ib[sel]], xyz_al[ia[sel]])
    resid = np.linalg.norm((xyz_snap[ib] - c) @ R.T - xyz_al[ia], axis=1)
    print(f"matched IDs: {common.size}; Kabsch on {sel.size}; "
          f"max|resid| = {resid.max():.3e} kpc (gate {args.residual_gate:.0e})")
    if resid.max() > args.residual_gate:
        print("TRANSFORM GATE FAILED")
        return 4
    print("R =", np.round(R, 6).tolist(), " center_kpc =", np.round(c, 4).tolist())
    with h5py.File(args.stars, "r") as f:
        tiv = np.asarray(f.attrs.get("header_Tiv_star",
                                     np.full((3, 3), np.nan)))
    if np.isfinite(tiv).all():
        print("max|R - Tiv_star| =", float(np.max(np.abs(R - tiv))),
              " max|R - Tiv_star.T| =", float(np.max(np.abs(R - tiv.T))))

    print("=== STEP 2: load types, transform, cut r<75 kpc ===")
    out = Path(args.output)
    if out.exists():
        print(f"asset exists - verify and reuse: {out}")
    else:
        counts, masses = {}, {}
        with h5py.File(out, "w") as fo:
            for ptype in TYPES:
                pid, x, y, z = load_snapshot_group(
                    args.snapdir, ptype, ["ParticleIDs", "x", "y", "z"])
                with h5py.File(args.snapdir + "/snapshot_127.0.hdf5", "r") as f0:
                    mname = "Masses" if "Masses" in f0[ptype] else None
                if mname is None:
                    raise RuntimeError(f"no Masses field in {ptype}")
                mm, = load_snapshot_group(args.snapdir, ptype, ["Masses"])
                xyz = np.stack([x, y, z], axis=1)
                r = np.linalg.norm(xyz - c, axis=1)
                keep = r <= args.r_max
                xyz_t = (xyz[keep] - c) @ R.T
                m = mm[keep]
                g5 = fo.create_group(ptype)
                g5.create_dataset("ParticleIDs", data=pid[keep], compression="lzf")
                g5.create_dataset("x", data=xyz_t[:, 0].astype(np.float32), compression="lzf")
                g5.create_dataset("y", data=xyz_t[:, 1].astype(np.float32), compression="lzf")
                g5.create_dataset("z", data=xyz_t[:, 2].astype(np.float32), compression="lzf")
                g5.create_dataset("mass", data=m.astype(np.float32), compression="lzf")
                counts[ptype] = int(keep.sum())
                masses[ptype] = float(m.sum())
                print(f"{ptype}: kept {counts[ptype]} within {args.r_max} kpc, "
                      f"M = {masses[ptype]:.4e} Msun ({time.time()-t0:.0f}s)")
            fo.attrs.update(dict(
                schema="dpjax.total-matter-particles.starframe.v1",
                snapdir=args.snapdir, r_max_kpc=args.r_max,
                transform_R=R.tolist(), center_kpc=c.tolist(),
                counts=json.dumps(counts),
                total_mass=sum(masses.values())))
        print(f"asset written: {out}")

    print("=== STEP 4: M(<r) validation vs 60-shell truth ===")
    with h5py.File(args.truth, "r") as f:
        edges = np.asarray(f["r_edges"][:])
        m_cum_true = np.asarray(f["M_cum_total"][:])
        m_cum_star_true = np.asarray(f["PartType4/M_cum"][:], dtype=np.float64)
    with h5py.File(out, "r") as f:
        all_xyz, all_m, star_xyz, star_m = [], [], [], []
        for ptype in TYPES:
            g = f[ptype]
            xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], axis=1).astype(np.float64)
            m = g["mass"][:].astype(np.float64)
            all_xyz.append(xyz)
            all_m.append(m)
            if ptype == "PartType4":
                star_xyz, star_m = xyz, m
    r_all = np.linalg.norm(np.concatenate(all_xyz), axis=1)
    m_all = np.concatenate(all_m)
    order = np.argsort(r_all)
    r_sorted, m_sorted = r_all[order], m_all[order]
    idx = np.searchsorted(r_sorted, edges[1:], side="right")
    cum = np.cumsum(m_sorted)
    m_cum = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    rel = np.abs(m_cum - m_cum_true) / m_cum_true
    r_star = np.linalg.norm(star_xyz, axis=1)
    os_ = np.argsort(r_star)
    idx_s = np.searchsorted(r_star[os_], edges[1:], side="right")
    cum_s = np.cumsum(star_m[os_])
    m_cum_s = np.where(idx_s > 0, cum_s[np.maximum(idx_s - 1, 0)], 0.0)
    rel_s = np.abs(m_cum_s - m_cum_star_true) / m_cum_star_true
    print(f"total matter: max rel err vs truth M_cum = {rel.max():.3e} at "
          f"r={edges[1:][np.argmax(rel)]:.2f} kpc")
    print(f"stellar only: max rel err vs PartType4/M_cum = {rel_s.max():.3e}")
    ok = rel.max() < 0.01 and rel_s.max() < 0.01
    for rr, a, b in zip(edges[1:][::12], m_cum[::12], m_cum_true[::12]):
        print(f"  r={rr:7.2f}: particles {a:.4e} truth {b:.4e}")
    print("VALIDATION:", "PASS" if ok else "FAIL")
    print("BUILD_TOTAL_MATTER_TRUTH_DONE")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
