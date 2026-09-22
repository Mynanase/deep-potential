#!/usr/bin/env python
"""Collection of truth-side preprocessing and cached data products.

Subcommands
  build-grids        particle asset (star frame, h5) -> grid product h5
                     (potential slices/sphere means + density histograms).
                     Absorbs archive/build_particle_truth_grids.py and
                     archive/particle_truth.py verbatim; parameters reproduce
                     the handoff figure (run e1801a91).
  build-shell-mass   grid product h5 -> shell mass table h5: per-shell
                     masses with Poisson confidence intervals and M(<r)
                     at radial nodes. The truth base of the enclosed-mass
                     comparison figure.
  build-radial-hists training/validation eta h5 -> r-bin count/mass tables
                     plus per-bin radial and velocity histograms (truth side
                     of the radial-marginals figure).

Cache contract (same spirit as the frozen training data): every product
records lineage (source sha256 + all relevant parameters).  Re-running
with an existing output verifies lineage and REUSES it (exit 0); a lineage
mismatch is an error, never a silent overwrite.  --force overrides after
an explicit decision.

Discretisation note: shell masses assign each histogram cell to the shell
of its centre.  At the 96^3 grid (1.5625 kpc cells) this is second order
and is stated here rather than hidden.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, sobol_directions, make_radial_nodes, load_truth)

GRIDS_SCHEMA = "dpjax.particle-truth-grids.v1"
SHELL_SCHEMA = "dpjax.shell-mass.v1"
RHIST_SCHEMA = "dpjax.radial-hists.v1"

TYPES = ("PartType0", "PartType1", "PartType4")

DEFAULT_R_ANCHOR = 1.0906   # first radial node of the handoff grid contract
DEFAULT_R_OUTER = 70.0
DEFAULT_BAND_EDGES = (2.0, 10.0, 30.0, 50.0, 70.0)  # adjudication bands


# --------------------------------------------------------------------------
# shared helpers (absorbed from archive/particle_truth.py)
# --------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_particles(asset, r_max=75.0):
    """All particle types from the star-frame asset: (xyz kpc f8, mass Msun f8)."""
    xyzs, ms = [], []
    with h5py.File(asset, "r") as f:
        for ptype in TYPES:
            g = f[ptype]
            xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], axis=1).astype(np.float64)
            xyzs.append(xyz)
            ms.append(g["mass"][:].astype(np.float64))
    xyz = np.concatenate(xyzs)
    m = np.concatenate(ms)
    if r_max is not None:
        keep = np.linalg.norm(xyz, axis=1) <= r_max
        xyz, m = xyz[keep], m[keep]
    return xyz, m


def phi_direct(xyz, m, query, eps_kpc=1e-3, chunk_q=256, chunk_p=250_000):
    """Direct-summation potential -G sum m_i/|x-q_i| [(km/s)^2] via GPU.

    One jit-compiled kernel for fixed chunk shapes; the particle loop runs
    outside jit with fp64 numpy accumulation (avoids the pathological XLA
    input_reduce_fusion compile for large 2-D reductions).
    """
    import jax
    import jax.numpy as jnp
    P = jnp.asarray(xyz, dtype=jnp.float32)
    M = jnp.asarray(m, dtype=jnp.float32)
    out = np.empty(query.shape[0], dtype=np.float64)
    eps2 = float(eps_kpc) ** 2

    def kernel(qb, pb, mb):
        d2 = jnp.sum((qb[:, None, :] - pb[None, :, :]) ** 2, axis=2) + eps2
        return jnp.sum(mb / jnp.sqrt(d2), axis=1)

    kernel_jit = jax.jit(kernel)
    n_q = query.shape[0]
    pad = (-n_q) % chunk_q
    q_all = np.concatenate([query, np.repeat(query[:1], pad, axis=0)]) if pad else query
    parts = [(P[i:i + chunk_p], M[i:i + chunk_p])
             for i in range(0, P.shape[0], chunk_p)]
    for i in range(0, q_all.shape[0], chunk_q):
        qb = jnp.asarray(q_all[i:i + chunk_q], dtype=jnp.float32)
        acc = np.zeros(min(chunk_q, q_all.shape[0] - i), dtype=np.float64)
        for pb, mb in parts:
            acc += np.asarray(kernel_jit(qb, pb, mb).block_until_ready(),
                              dtype=np.float64)
        j0, j1 = i, min(i + chunk_q, n_q)
        out[j0:j1] = acc[:j1 - j0]
    return -G_KPC_KMS2_MSUN * out


# --------------------------------------------------------------------------
# lineage / idempotent cache
# --------------------------------------------------------------------------

def _lineage_match(path, kind, lineage):
    """True if `path` exists, has the right schema and identical lineage."""
    if not Path(path).is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            if str(f.attrs.get("schema", "")) != kind:
                return False
            old = json.loads(f.attrs["lineage_json"])
    except Exception:
        return False
    return old == lineage


def _open_with_lineage(path, kind, lineage):
    """Create the product h5 and stamp schema + lineage attrs.

    Returns an open h5py.File; the caller writes datasets in sequence
    inside a `with` block (research-script style, no writer callbacks).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs.update({
        "schema": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lineage_json": json.dumps(lineage, sort_keys=True),
    })
    return f


def _cache_guard(path, kind, lineage, force):
    if _lineage_match(path, kind, lineage):
        print(f"cache hit: {path} lineage matches -> reuse (pass --force to rebuild)")
        return True
    if Path(path).is_file() and not force:
        with h5py.File(path, "r") as f:
            try:
                old = json.loads(f.attrs.get("lineage_json", '"{}"'))
            except Exception:
                old = None
        raise SystemExit(
            f"ERROR: {path} exists with a DIFFERENT lineage.\n"
            f"  old: {old}\n  new: {lineage}\n"
            f"Use a new output name, or --force after an explicit decision.")
    return False


# --------------------------------------------------------------------------
# build-grids (verbatim port of archive/build_particle_truth_grids.py)
# --------------------------------------------------------------------------

def cmd_build_grids(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"asset: {args.asset}")
    t0 = time.time()
    asset_sha = sha256_file(args.asset)
    print(f"asset sha256: {asset_sha} ({time.time()-t0:.0f}s)")
    by_type = {}
    with h5py.File(args.asset, "r") as f:
        for ptype in TYPES:
            by_type[ptype] = (int(f[ptype]["mass"].shape[0]),
                              float(f[ptype]["mass"][:].sum()))
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
    r_anchor = DEFAULT_R_ANCHOR
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
              f"{np.median(rel[dom])*100:.2f}%, max {np.max(rel[dom])*100:.2f}% "
              "(asset validation was 3.2%)")

    print("=== write product ===")
    with _open_with_lineage(args.output, GRIDS_SCHEMA, lineage) as f:
        f.attrs.update({
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

    lineage = {
        "asset_sha256": asset_sha,
        "r_outer": float(args.r_outer), "n_grid": int(args.n_grid),
        "n_dirs": int(args.n_dirs), "sobol_seed": int(args.sobol_seed),
        "n_sphere_nodes": int(args.n_sphere_nodes),
        "rho3d_n": int(n3), "rho3d_extent": float(ext),
        "slab_n": int(args.slab_n), "slab_half_width": float(hw),
        "rho_r_nbins": int(re.size - 1), "rho_r_min": float(args.rho_r_min),
    }
    if _cache_guard(args.output, GRIDS_SCHEMA, lineage, args.force):
        return 0
    print(f"wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB)")

    print("=== read-back verification ===")
    with h5py.File(args.output, "r") as f:
        assert f.attrs["schema"] == GRIDS_SCHEMA
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
    return 0


# --------------------------------------------------------------------------
# build-shell-mass (new: truth base of the enclosed-mass figure)
# --------------------------------------------------------------------------

def shell_mass_from_grids(grids_path, r_edges_kpc, r_nodes_kpc):
    """Shell masses + M(<r) nodes from the grid product, cell-centre binning.

    Returns dict with
      r_edges, M_shell, M_shell_err (Poisson 1-sigma), N_eff_shell,
      r_nodes, M_cum, M_cum_err (Poisson 1-sigma).
    N_eff uses the mean particle mass from the product attrs; for mixed-mass
    populations this is an effective count, adequate for CI scaling.
    """
    with h5py.File(grids_path, "r") as f:
        if str(f.attrs.get("schema", "")) != GRIDS_SCHEMA:
            raise SystemExit(f"{grids_path}: not a {GRIDS_SCHEMA} product")
        rho3d = np.asarray(f["density/rho3d"][:], dtype=np.float64)
        edges3 = np.asarray(f["density/rho3d_edges_kpc"][:], dtype=np.float64)
        m_total = float(f.attrs["M_total_msun"])
        n_particles = int(f.attrs["n_particles"])

    ex = edges3[0]
    c3 = 0.5 * (ex[:-1] + ex[1:])
    cell = float((ex[1] - ex[0]) ** 3)
    xx, yy, zz = np.meshgrid(c3, c3, c3, indexing="ij")
    r_c = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2).ravel()
    m_c = (rho3d * cell).ravel()

    r_edges = np.asarray(r_edges_kpc, dtype=float)
    idx = np.clip(np.digitize(r_c, r_edges) - 1, 0, r_edges.size - 2)
    inside = (r_c >= r_edges[0]) & (r_c < r_edges[-1])
    m_shell = np.zeros(r_edges.size - 1)
    np.add.at(m_shell, idx[inside], m_c[inside])
    mean_m = m_total / max(n_particles, 1)
    n_eff = m_shell / mean_m
    m_shell_err = m_shell / np.sqrt(np.maximum(n_eff, 1e-12))

    # cumulative at nodes: sum of cell masses with r_c <= node
    order = np.argsort(r_c)
    r_sorted = r_c[order]
    m_sorted = np.cumsum(m_c[order])
    n_sorted = np.cumsum((m_c[order] / mean_m))
    r_nodes = np.asarray(r_nodes_kpc, dtype=float)
    pos = np.searchsorted(r_sorted, r_nodes, side="right")
    m_cum = np.where(pos > 0, m_sorted[np.clip(pos - 1, 0, m_sorted.size - 1)], 0.0)
    n_cum = np.where(pos > 0, n_sorted[np.clip(pos - 1, 0, n_sorted.size - 1)], 0.0)
    m_cum_err = m_cum / np.sqrt(np.maximum(n_cum, 1e-12))
    return {
        "r_edges": r_edges, "M_shell": m_shell, "M_shell_err": m_shell_err,
        "N_eff_shell": n_eff, "r_nodes": r_nodes,
        "M_cum": m_cum, "M_cum_err": m_cum_err,
        "M_total": m_total, "n_particles": n_particles,
    }


def cmd_build_shell_mass(args):
    r_edges = np.asarray([float(v) for v in args.edges.split(",") if v != ""],
                         dtype=float)
    if args.nodes:
        r_nodes = np.asarray([float(v) for v in args.nodes.split(",") if v != ""],
                             dtype=float)
    else:
        r_nodes = np.asarray(make_radial_nodes(
            DEFAULT_R_ANCHOR, DEFAULT_R_OUTER, args.n_sphere_nodes - 1), dtype=float)

    grids_sha = sha256_file(args.grids)
    lineage = {"grids_sha256": grids_sha,
               "r_edges": r_edges.tolist(),
               "r_nodes": r_nodes.tolist()}
    if _cache_guard(args.output, SHELL_SCHEMA, lineage, args.force):
        return 0

    tab = shell_mass_from_grids(args.grids, r_edges, r_nodes)
    print(f"shell masses from {args.grids}:")
    for lo, hi, mm, ee, nn in zip(r_edges[:-1], r_edges[1:],
                                  tab["M_shell"], tab["M_shell_err"],
                                  tab["N_eff_shell"]):
        print(f"  {lo:6.2f}-{hi:6.2f} kpc: M={mm:.4e} +- {ee:.1e} Msun "
              f"(N_eff={nn:.0f})")
    closure = tab["M_shell"].sum() / tab["M_total"]
    print(f"sum(M_shell)/M_total = {closure:.4f} "
          "(<1 expected: cells outside the edge range are excluded)")

    with _open_with_lineage(args.output, SHELL_SCHEMA, lineage) as f:
        f.attrs.update({
            "source_grids": str(args.grids),
            "M_total_msun": tab["M_total"],
            "n_particles": tab["n_particles"],
            "method": "rho3d cell-centre binning; Poisson CI from effective counts",
        })
        f["r_edges"] = tab["r_edges"]
        f["M_shell"] = tab["M_shell"]
        f["M_shell_err"] = tab["M_shell_err"]
        f["N_eff_shell"] = tab["N_eff_shell"]
        f["r_nodes"] = tab["r_nodes"]
        f["M_cum"] = tab["M_cum"]
        f["M_cum_err"] = tab["M_cum_err"]

    print(f"wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB)")
    print("SHELL_MASS_DONE")
    return 0


# --------------------------------------------------------------------------
# build-radial-hists (truth side of the radial-marginals figure)
# --------------------------------------------------------------------------

def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


VEL_BINS = np.linspace(-4.5, 4.5, 73)   # code units, rbin-family convention
R_SUBBINS = 64                          # fine radial sub-bins inside each bin


def radial_bin_table(input_h5, r_edges_kpc, weighting="mass", split="val",
                     val_frac=0.25):
    """Truth-side r-bin tables from a training/eta h5 (positions in code units).

    Returns dict with per-bin count, mass, radial sub-histogram and the three
    spherical velocity-component histograms (code units), plus lineage inputs.
    """
    with h5py.File(input_h5, "r") as f:
        eta = np.asarray(f["eta"][:], dtype=np.float64)
        w = np.asarray(f["weights"][:], dtype=np.float64)
        attrs = dict(f.attrs)
    n_tot = eta.shape[0]
    if split == "val":
        n = int(n_tot * val_frac)
        eta, w = eta[:n], w[:n]
    elif split != "all":
        raise ValueError(f"split must be 'val' or 'all', got {split!r}")
    if "length_scale_kpc" not in attrs:
        raise SystemExit(f"{input_h5}: missing length_scale_kpc attr")
    L = float(attrs["length_scale_kpc"])

    weights = w if weighting == "mass" else np.ones_like(w)
    pos = eta[:, :3] * L
    r = np.linalg.norm(pos, axis=1)
    v = sph_vel(pos, eta[:, 3:])

    r_edges = np.asarray(r_edges_kpc, dtype=float)
    idx = np.clip(np.digitize(r, r_edges) - 1, 0, r_edges.size - 2)
    inside = (r >= r_edges[0]) & (r < r_edges[-1])

    bins = []
    for b in range(r_edges.size - 1):
        m = inside & (idx == b)
        lo, hi = r_edges[b], r_edges[b + 1]
        sub = np.geomspace(max(lo, 1e-3), hi, R_SUBBINS + 1)
        h_r, _ = np.histogram(r[m], bins=sub, weights=weights[m])
        hists_v = []
        for ci in range(3):
            hv, _ = np.histogram(v[m, ci], bins=VEL_BINS, weights=weights[m])
            hists_v.append(hv)
        bins.append({
            "count": int(m.sum()),
            "mass": float(weights[m].sum()),
            "r_hist": h_r, "r_sub_edges": sub,
            "v_hists": np.stack(hists_v),
        })
    return {"bins": bins, "r_edges": r_edges, "L_kpc": L,
            "weighting": weighting, "split": split, "n_used": int(inside.sum()),
            "vel_bin_edges": VEL_BINS}


def cmd_build_radial_hists(args):
    r_edges = np.asarray([float(v) for v in args.edges.split(",") if v != ""],
                         dtype=float)
    src_sha = sha256_file(args.input)
    lineage = {"input_sha256": src_sha, "r_edges": r_edges.tolist(),
               "weighting": args.weighting, "split": args.split}
    if _cache_guard(args.output, RHIST_SCHEMA, lineage, args.force):
        return 0

    tab = radial_bin_table(args.input, r_edges, args.weighting, args.split)
    print(f"radial hists from {args.input} (split={args.split}, "
          f"weighting={args.weighting}): {tab['n_used']} particles in range")
    for b, spec in enumerate(tab["bins"]):
        print(f"  {r_edges[b]:6.2f}-{r_edges[b+1]:6.2f} kpc: "
              f"n={spec['count']:8d}  mass={spec['mass']:.4e}")

    with _open_with_lineage(args.output, RHIST_SCHEMA, lineage) as f:
        f.attrs.update({
            "source_input": str(args.input),
            "length_scale_kpc": tab["L_kpc"],
            "weighting": tab["weighting"],
            "split": tab["split"],
            "n_used": tab["n_used"],
        })
        f["r_edges"] = tab["r_edges"]
        f["vel_bin_edges"] = tab["vel_bin_edges"]
        for b, spec in enumerate(tab["bins"]):
            g = f.create_group(f"bin{b:02d}")
            g["count"] = spec["count"]
            g["mass"] = spec["mass"]
            g["r_hist"] = spec["r_hist"]
            g["r_sub_edges"] = spec["r_sub_edges"]
            g["v_hists"] = spec["v_hists"]

    print(f"wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB)")
    print("RADIAL_HISTS_DONE")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("build-grids", help="particle asset -> grid product h5")
    g.add_argument("--asset", type=Path, required=True)
    g.add_argument("--output", type=Path,
                   default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    g.add_argument("--truth", type=Path, default=None,
                   help="60-shell truth hdf5 for the radial cross-check")
    g.add_argument("--r-outer", type=float, default=DEFAULT_R_OUTER)
    g.add_argument("--n-grid", type=int, default=240)
    g.add_argument("--n-dirs", type=int, default=2048)
    g.add_argument("--sobol-seed", type=int, default=20260917)
    g.add_argument("--n-sphere-nodes", type=int, default=24)
    g.add_argument("--rho3d-n", type=int, default=96)
    g.add_argument("--rho3d-extent", type=float, default=75.0)
    g.add_argument("--slab-n", type=int, default=480)
    g.add_argument("--slab-half-width", type=float, default=0.5)
    g.add_argument("--rho-r-nbins", type=int, default=240)
    g.add_argument("--rho-r-min", type=float, default=0.5)
    g.add_argument("--force", action="store_true")
    g.set_defaults(fn=cmd_build_grids)

    s = sub.add_parser("build-shell-mass", help="grid product -> shell mass table h5")
    s.add_argument("--grids", type=Path, required=True)
    s.add_argument("--output", type=Path,
                   default=Path("data/auriga/halo12_shell_mass.h5"))
    s.add_argument("--edges", type=str,
                   default=",".join(str(v) for v in DEFAULT_BAND_EDGES),
                   help="comma-separated shell outer edges [kpc]")
    s.add_argument("--nodes", type=str, default="",
                   help="comma-separated M(<r) nodes [kpc]; default = handoff nodes")
    s.add_argument("--n-sphere-nodes", type=int, default=24)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_build_shell_mass)

    r = sub.add_parser("build-radial-hists",
                       help="eta h5 -> r-bin count/mass/velocity tables h5")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path,
                   default=Path("data/auriga/halo12_radial_hists.h5"))
    r.add_argument("--edges", type=str, default="0,10,20,30,45,64,75",
                   help="comma-separated r-bin outer edges [kpc]")
    r.add_argument("--weighting", choices=("mass", "number"), default="mass")
    r.add_argument("--split", choices=("val", "all"), default="val")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_build_radial_hists)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
