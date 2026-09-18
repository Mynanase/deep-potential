#!/usr/bin/env python
"""Velocity-only debris removal on the raw halo12 (no spatial clustering).

Explores, on the un-removed population (r 45-75 kpc), how much of the
stream + debris structure can be removed by velocity-space criteria alone,
without any physical-space clustering (no FOF linkage, no registry-based
selection). Schemes:

  S0  frozen template gate: |v - mu| < 2.5 sigma per component, template
      from the in-cell detection members (same center as frozen run
      e94c3979), applied to ALL raw stars in the bin (v2's spatial pool
      constraint dropped);
  S1  iterative self-consistent gate: recompute (mu, sigma) from the
      removed set on raw, re-gate, iterate until the pid set converges;
  S2  per-r-bin 3-component GMM in (vT, vr, vth), features standardized by
      the far-field control-cell vT dispersion; debris = component with the
      most negative vT mean; remove P(debris) > 0.5 (weighted EM in numpy,
      no sklearn dependency);
  S3  1D vT tail cut vT < mu_far(r) - k sigma_far(r) for k in {1.5, 2.5},
      (mu_far, sigma_far) from the per-r-bin far-field control cell (same
      sky cell, >= 8 kpc from the detection members; the raw catalog only
      covers r < 75 kpc, so no 75-105 kpc bins exist)
      (control: expected large collateral, quantifies why 1D fails).

Reference sets: frozen v1 (129) and v2 (638) candidates for coverage; the
cascade stage1+v1+v2 kept population (ba7750a4) for the kept frac / W1
reference. Collateral controls: removed prograde (vT > 0) mass and the hit
rate in the v2 far-field control cell (same sky cell, >= 8 kpc from the
detection members -- used for measurement only, never for selection).

Run from repo root (server):
  python scripts/auriga/df_velocity_removal_explore.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orx_figstyle import (  # noqa: E402
    MUTED,
    PALETTE,
    WIDE,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

SRC = Path("/localdisk/kosmos/my-deep-potential")
RAW_H5 = SRC / "data" / "halo_12_stars.hdf5"
UNION_H5 = SRC / "data" / "auriga" / "halo12_all_mass_clean_outer_clump_smooth.h5"
V1_NPZ = SRC / "data" / "auriga" / "clump_debris_candidates.npz"
V2_NPZ = SRC / "data" / "auriga" / "clump_debris_candidates_v2.npz"
REG_NPZ = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "velocity-removal-explore"

R_BINS = [(45.0, 55.0), (55.0, 65.0), (65.0, 75.0)]
R_HI_EXPLORE = 75.0
VT_CUT = -100.0
N_SIGMA = 2.5
V_EDGES = np.linspace(-450.0, 450.0, 61)
VT_IDX = 2  # component order below: (vr, vth, vT)


def sph_v(pos, vel):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = (x * vel[:, 0] + y * vel[:, 1] + z * vel[:, 2]) / r
        vth = (z * vr - r * vel[:, 2]) / R
        vT = (-vel[:, 0] * y + vel[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def fit_gmm(X, w, k=3, n_init=5, seed0=0, max_iter=300, tol=1e-5, reg=1e-6):
    """Weighted full-covariance GMM via EM; returns best (ll, pi, mu, cov)."""
    n, d = X.shape
    best = None
    eye = np.eye(d)
    base_cov = np.cov(X.T, aweights=w) + reg * eye
    for s in range(n_init):
        rng = np.random.default_rng(seed0 + s)
        sub = rng.choice(n, size=min(n, 20000), replace=False)
        Xs = X[sub]
        means = [Xs[rng.integers(len(Xs))]]
        for _ in range(k - 1):
            d2 = np.min(((Xs[:, None, :] - np.stack(means)[None, :, :]) ** 2).sum(-1), axis=1)
            tot = d2.sum()
            means.append(Xs[rng.choice(len(Xs))] if not tot > 0 else
                         Xs[rng.choice(len(Xs), p=d2 / tot)])
        mu = np.stack(means)
        cov = np.stack([base_cov.copy() for _ in range(k)])
        pi = np.full(k, 1.0 / k)
        ll_prev = -np.inf
        for _ in range(max_iter):
            logp = np.stack([multivariate_normal.logpdf(X, mu[j], cov[j])
                             for j in range(k)], axis=1) + np.log(pi)
            resp = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
            rw = resp * w[:, None]
            nk = rw.sum(0) + 1e-12
            pi = nk / nk.sum()
            mu = (rw.T @ X) / nk[:, None]
            for j in range(k):
                diff = X - mu[j]
                cov[j] = ((rw[:, j][:, None] * diff).T @ diff) / nk[j] + reg * eye
            ll = float((logsumexp(logp, axis=1) * w).sum() / w.sum())
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll
        if best is None or ll > best[0]:
            best = (ll, pi, mu, cov)
    return best


# ---- data ----
with h5py.File(RAW_H5, "r") as f:
    g = f["PartType4"]
    pos = np.column_stack([g[n][:] for n in ("x", "y", "z")]).astype(np.float64)
    vel = np.column_stack([g[n][:] for n in ("vx", "vy", "vz")]).astype(np.float64)
    mass = g["Masses"][:].astype(np.float64)
    pid_raw = g["ParticleIDs"][:].astype(np.int64)

v = sph_v(pos, vel)
r = np.linalg.norm(pos, axis=1)
with np.errstate(invalid="ignore", divide="ignore"):
    cth = np.divide(pos[:, 2], r, out=np.zeros_like(r), where=r > 0)
phi = np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi
print(f"raw n={len(pid_raw)}", flush=True)

# template (in-cell detection members), identical to the frozen v2 run
reg = np.load(REG_NPZ)
det = np.asarray(reg["detection_pids"]).astype(np.int64)
mem_det = np.isin(pid_raw, det)
det_pos = pos[mem_det]
r_det = np.linalg.norm(det_pos, axis=1)
cth_det = det_pos[:, 2] / r_det
phi_det = np.mod(np.arctan2(det_pos[:, 1], det_pos[:, 0]), 2 * np.pi) - np.pi
in_cell = mem_det & (cth < -0.6) & (phi >= np.pi / 2) & (phi < np.pi)
tpl_mu = v[in_cell].mean(axis=0)
tpl_sd = v[in_cell].std(axis=0)
print("template (vr, vth, vT) mean:", np.round(tpl_mu, 1), "std:", np.round(tpl_sd, 1), flush=True)
ref_tpl = np.array([0.32, -1.746, -2.05]) * 100.0
assert np.all(np.abs(tpl_mu - ref_tpl) < 2.0), "template drift vs frozen v2 run"

with h5py.File(UNION_H5, "r") as f:
    pid_union = f["particle_id"][:].astype(np.int64)
v1_pids = np.load(V1_NPZ)["debris_pids"].astype(np.int64)
v2_pids = np.load(V2_NPZ)["debris_pids"].astype(np.int64)
ref_pids = np.union1d(v1_pids, v2_pids)
is_ref = np.isin(pid_raw, ref_pids)
is_union = np.isin(pid_raw, pid_union)
is_v1 = np.isin(pid_raw, v1_pids)
is_v2 = np.isin(pid_raw, v2_pids)
print(f"reference set v1|v2 union: n={len(ref_pids)}, in raw={int(is_ref.sum())}", flush=True)

# far-field control cell (measurement only): same sky cell, >= 8 kpc from
# any detection member, r 45-75
d_det = cKDTree(det_pos).query(pos, k=1)[0]
far_cell = (r >= 45.0) & (r < 75.0) & (cth < -0.6) \
    & (phi >= np.pi / 2) & (phi < np.pi) & (d_det >= 8.0)
print(f"far-field control cell: n={int(far_cell.sum())} "
      f"m={mass[far_cell].sum() / 1e6:.2f}e6 "
      f"frac(vT<-100)={float(mass[far_cell & (v[:, VT_IDX] < VT_CUT)].sum() / mass[far_cell].sum()):.3f}",
      flush=True)

explore = (r >= 45.0) & (r < R_HI_EXPLORE)

# far-field calibration for S3 and GMM standardization: per-r-bin control
# cell (same sky cell, d>=8 kpc from detection members). The raw catalog
# has no particles at r >= 75 kpc, so radial 75-105 kpc bins are empty.
far_pts, far_mu, far_sd = [], [], []
for lo, hi in R_BINS:
    m = far_cell & (r >= lo) & (r < hi)
    assert m.sum() > 0, f"empty far-field control bin {lo:.0f}-{hi:.0f}"
    mu_b = float((mass[m] * v[m, VT_IDX]).sum() / mass[m].sum())
    sd_b = float(np.sqrt((mass[m] * (v[m, VT_IDX] - mu_b) ** 2).sum() / mass[m].sum()))
    far_pts.append(0.5 * (lo + hi))
    far_mu.append(mu_b)
    far_sd.append(sd_b)
print(f"far-field vT calibration: r={far_pts} mu={np.round(far_mu, 1)} sd={np.round(far_sd, 1)}", flush=True)
# vT dispersion of the full far-field control cell, used to standardize
# all 3 GMM components (full covariance learns per-component scales).
mfc = far_cell
sd_far = float(np.sqrt((mass[mfc] * (v[mfc, VT_IDX]
            - (mass[mfc] * v[mfc, VT_IDX]).sum() / mass[mfc].sum()) ** 2).sum() / mass[mfc].sum()))
sd_far3 = np.array([sd_far] * 3)


def box_gate(mask, mu, sd, nsig=N_SIGMA):
    g = np.zeros(len(pid_raw), dtype=bool)
    idx = np.where(mask)[0]
    ok = np.ones(len(idx), dtype=bool)
    for c in range(3):
        ok &= np.abs(v[idx, c] - mu[c]) < nsig * sd[c]
    g[idx[ok]] = True
    return g


# S0: frozen template gate on all raw r45-75 (no spatial pool)
rem_s0 = box_gate(explore, tpl_mu, tpl_sd)

# S1: iterative self-consistent gate
mu_it, sd_it = tpl_mu.copy(), tpl_sd.copy()
rem_s1 = box_gate(explore, mu_it, sd_it)
for it in range(1, 21):
    sel = rem_s1
    mu_new = v[sel].mean(axis=0)
    sd_new = v[sel].std(axis=0)
    rem_new = box_gate(explore, mu_new, sd_new)
    same = np.array_equal(np.sort(pid_raw[rem_new]), np.sort(pid_raw[rem_s1]))
    mu_it, sd_it = mu_new, sd_new
    rem_s1 = rem_new
    if same:
        break
print(f"S1 converged after {it} iterations; mu={np.round(mu_it, 1)} sd={np.round(sd_it, 1)}", flush=True)

# S2: per-r-bin GMM
rem_s2 = np.zeros(len(pid_raw), dtype=bool)
gmm_info = []
for lo, hi in R_BINS:
    m = (r >= lo) & (r < hi)
    X = v[m] / sd_far3
    ll, pi, mu_z, cov = fit_gmm(X, mass[m], k=3, n_init=5, seed0=0)
    dej = int(np.argmin(mu_z[:, VT_IDX]))
    logp = np.stack([multivariate_normal.logpdf(X, mu_z[j], cov[j]) for j in range(3)], axis=1) + np.log(pi)
    pdej = np.exp(logp[:, dej] - logsumexp(logp, axis=1))
    idx = np.where(m)[0]
    rem_s2[idx[pdej > 0.5]] = True
    mu_vt = mu_z[:, VT_IDX] * sd_far3[VT_IDX]
    gmm_info.append(dict(bin=[lo, hi], ll=ll, pi=np.round(pi, 3).tolist(),
                         mu_vT_kms=np.round(mu_vt, 1).tolist(),
                         debris_comp=dej, n_removed=int((pdej > 0.5).sum())))
    print(f"S2 bin {lo:.0f}-{hi:.0f}: pi={np.round(pi, 3)} mu_vT={np.round(mu_vt, 1)} "
          f"debris comp={dej} removed n={int((pdej > 0.5).sum())}", flush=True)

# S3: 1D vT tail cuts
rem_s3 = {}
for k in (1.5, 2.5):
    thr = np.zeros(len(r))
    for (lo, hi), mu_b, sd_b in zip(R_BINS, far_mu, far_sd):
        b = (r >= lo) & (r < hi)
        thr[b] = mu_b - k * sd_b
    rem_s3[k] = explore & (v[:, VT_IDX] < thr)

# cascade reference kept population (stage1 + v1 + v2 removed)
kept_ref = is_union & ~is_v1 & ~is_v2


def metrics(rem):
    rows = []
    for lo, hi in R_BINS:
        in_bin = (r >= lo) & (r < hi)
        kept = in_bin & ~rem
        w_kept = mass[kept]
        ref_bin = is_ref & in_bin
        removed = rem & in_bin
        rows.append(dict(
            scheme="", bin=f"{lo:.0f}-{hi:.0f}",
            removed_n=int(removed.sum()),
            removed_mass_1e6=float(mass[removed].sum() / 1e6),
            kept_n=int(kept.sum()),
            kept_mass_1e6=float(w_kept.sum() / 1e6),
            kept_frac_vT_lt_minus100=float(w_kept[v[kept, VT_IDX] < VT_CUT].sum() / w_kept.sum()),
            kept_mean_vT=float((w_kept * v[kept, VT_IDX]).sum() / w_kept.sum()),
            w1_vT_raw_vs_kept=float(wasserstein_distance(
                v[in_bin, VT_IDX], v[kept, VT_IDX], mass[in_bin], w_kept)),
            removed_prograde_mass_1e6=float(mass[removed & (v[:, VT_IDX] > 0)].sum() / 1e6),
            removed_prograde_frac=float(mass[removed & (v[:, VT_IDX] > 0)].sum()
                                        / max(mass[removed].sum(), 1e-30)),
            ref_coverage=float(np.isin(pid_raw[removed & ref_bin], ref_pids).sum()
                               / max(int(ref_bin.sum()), 1)),
            far_cell_hit_frac=float(mass[rem & far_cell].sum() / mass[far_cell].sum()),
        ))
    return rows


schemes = [("S0_gate2.5", rem_s0), ("S1_iter_gate", rem_s1), ("S2_gmm3", rem_s2),
           ("S3_tail_k1.5", rem_s3[1.5]), ("S3_tail_k2.5", rem_s3[2.5])]
all_rows = []
print("\n=== per-bin metrics (mass in 1e6 Msun) ===", flush=True)
for name, rem in schemes:
    rows = metrics(rem)
    for row in rows:
        row["scheme"] = name
    all_rows.extend(rows)
    for row in rows:
        print(f"{name:14s} bin {row['bin']:>8s}: removed n={row['removed_n']:6d} "
              f"m={row['removed_mass_1e6']:7.2f} | kept frac={row['kept_frac_vT_lt_minus100']:.3f} "
              f"mean vT={row['kept_mean_vT']:+7.1f} W1={row['w1_vT_raw_vs_kept']:6.1f} | "
              f"prograde rem={row['removed_prograde_mass_1e6']:6.2f} "
              f"({100 * row['removed_prograde_frac']:4.1f}%) | ref cov={row['ref_coverage']:.2f} | "
              f"far-cell hit={100 * row['far_cell_hit_frac']:4.1f}%", flush=True)

print("\ncascade reference (stage1+v1+v2 kept):", flush=True)
for lo, hi in R_BINS:
    in_bin = (r >= lo) & (r < hi)
    w = mass[kept_ref & in_bin]
    print(f"  bin {lo:.0f}-{hi:.0f}: kept frac={float(w[v[kept_ref & in_bin, VT_IDX] < VT_CUT].sum() / w.sum()):.3f} "
          f"kept m={w.sum() / 1e6:.2f}", flush=True)

# ---- figure: kept vT distributions per bin ----
use_style()
fig, axes = figure_grid(1, 3, width=WIDE, ratio=0.36, sharey=True)
centers = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])
widths = np.diff(V_EDGES)
ymax = 0.0
for ax, (lo, hi) in zip(axes, R_BINS):
    in_bin = (r >= lo) & (r < hi)
    h_raw, _ = np.histogram(v[in_bin, VT_IDX], bins=V_EDGES, weights=mass[in_bin])
    h_ref, _ = np.histogram(v[kept_ref & in_bin, VT_IDX], bins=V_EDGES, weights=mass[kept_ref & in_bin])
    ymax = max(ymax, h_raw.max() / 1e6)
    ax.fill_between(centers, h_ref / 1e6, step="mid", color=MUTED, alpha=0.35,
                    label="cascade reference kept", zorder=2)
    ax.step(V_EDGES, np.append(h_raw, h_raw[-1]) / 1e6, where="post", color="black",
            lw=1.2, label="no removal (raw)", zorder=5)
    for (name, rem), col, lsty in zip(schemes,
                                      (PALETTE["blue"], PALETTE["green"], PALETTE["purple"],
                                       PALETTE["yellow"], PALETTE["orange"]),
                                      ("-", "-", "-", ":", ":")):
        kept = in_bin & ~rem
        h, _ = np.histogram(v[kept, VT_IDX], bins=V_EDGES, weights=mass[kept])
        ax.step(V_EDGES, np.append(h, h[-1]) / 1e6, where="post", color=col, lw=1.0,
                ls=lsty, label=name, zorder=4)
    ax.axvline(tpl_mu[VT_IDX], color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1,
               label="stream-core template")
    ax.set_xlabel("$v_T$ (km/s)")
    ax.set_xlim(-450, 450)
    ax.set_ylim(0, 1.06 * ymax)
axes[0].set_ylabel(r"mass per bin ($10^6\,M_\odot$)")
panel_labels(axes)
handles, labels = axes[0].get_legend_handles_labels()
order = [1, 0, 2, 3, 4, 5, 6, 7]
try:
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", ncols=3, fontsize=6.0)
except (TypeError, ValueError):
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   fontsize=5.0, loc="upper left")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
save(fig, str(OUT / "figs" / "fig_velocity_removal_explore"), formats=("png", "pdf", "svg"))

import csv  # noqa: E402
with open(OUT / "velocity_removal_metrics.csv", "w", newline="") as fcsv:
    wcsv = csv.DictWriter(fcsv, fieldnames=list(all_rows[0].keys()))
    wcsv.writeheader()
    wcsv.writerows(all_rows)
with open(OUT / "summary.json", "w") as fj:
    json.dump(dict(template_mu_kms=tpl_mu.tolist(), template_sd_kms=tpl_sd.tolist(),
                   s1_iterations=int(it), s1_mu_kms=mu_it.tolist(), s1_sd_kms=sd_it.tolist(),
                   gmm=gmm_info, far_calibration=dict(r_kpc=far_pts, mu_kms=far_mu, sd_kms=far_sd),
                   n_ref=int(len(ref_pids)), metrics=all_rows), fj, indent=1)
print("\noutputs under", OUT, flush=True)
print("DONE", flush=True)
