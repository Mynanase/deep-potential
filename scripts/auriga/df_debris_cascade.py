#!/usr/bin/env python
"""55-65 kpc removal cascade: what each stage removes.

Compares, in one radial bin (55-65 kpc), the spherical velocity components
of three nested populations of the Halo12 stellar sample:

  stage 0: raw halo12 (no removal),
  stage 1: union-removed population (clump detection + smooth84 registry),
  stage 2: stage 1 plus the 638 v2 velocity-gate debris candidates removed
           (run e94c3979, clump_debris_candidates_v2.npz).

All populations are particle-ID subsets of the raw export, so the three
curves share one physical mass scale (Msun). The raw histogram is drawn once
and decomposed into what stage 1 removed, what stage 2 removed, and what
survives both removals, so "no removal" and "after two removals" are directly
comparable bin by bin.

Run from repo root (server):
  python scripts/auriga/df_debris_cascade.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orx_figstyle import (  # noqa: E402
    BASELINE,
    MUTED,
    PALETTE,
    WIDE,
    figure_grid,
    panel_labels,
    save,
    use_style,
)
from scipy.stats import wasserstein_distance  # noqa: E402

SRC = Path("/localdisk/kosmos/my-deep-potential")
RAW_H5 = SRC / "data" / "halo_12_stars.hdf5"
UNION_H5 = SRC / "data" / "auriga" / "halo12_all_mass_clean_outer_clump_smooth.h5"
DEBRIS_NPZ = SRC / "data" / "auriga" / "clump_debris_candidates_v2.npz"
REG_NPZ = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "debris-cascade-55-65"

R_LO, R_HI = 55.0, 65.0          # kpc
CTX_BINS = [(45.0, 55.0), (55.0, 65.0), (65.0, 75.0)]
VT_CUT = -100.0                  # km/s; matches census frac(vT < -1) in 100 km/s units
V_EDGES = np.linspace(-450.0, 450.0, 61)
COMP = [("vT", r"$v_T$"), ("vr", r"$v_r$"), ("vth", r"$v_\theta$")]


def sph_vel(pos, vel):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = (x * vel[:, 0] + y * vel[:, 1] + z * vel[:, 2]) / r
        vth = (z * vr - r * vel[:, 2]) / R
        vT = (-vel[:, 0] * y + vel[:, 1] * x) / R
    return dict(vr=vr, vth=vth, vT=vT)


with h5py.File(RAW_H5, "r") as f:
    g = f["PartType4"]
    pos = np.column_stack([g[n][:] for n in ("x", "y", "z")]).astype(np.float64)
    vel = np.column_stack([g[n][:] for n in ("vx", "vy", "vz")]).astype(np.float64)
    mass = g["Masses"][:].astype(np.float64)          # Msun
    pid_raw = g["ParticleIDs"][:].astype(np.int64)

with h5py.File(UNION_H5, "r") as f:
    pid_union = f["particle_id"][:].astype(np.int64)
debris_pids = np.sort(np.load(DEBRIS_NPZ)["debris_pids"].astype(np.int64))

print("=== DEBRIS CASCADE 55-65 kpc ===", flush=True)
print(f"raw n={len(pid_raw)}, union-removed n={len(pid_union)}, "
      f"v2 debris candidates n={len(debris_pids)}", flush=True)

in_raw = np.isin(debris_pids, pid_raw)
in_union = np.isin(debris_pids, pid_union)
print(f"debris pids found in raw: {int(in_raw.sum())}/{len(debris_pids)}; "
      f"in union export: {int(in_union.sum())}/{len(debris_pids)}", flush=True)
assert in_raw.all(), "debris pids missing from raw export"
assert in_union.all(), "debris pids missing from union-removed export"

is_union = np.isin(pid_raw, pid_union)
is_debris = np.isin(pid_raw, debris_pids)

r = np.linalg.norm(pos, axis=1)
v = sph_vel(pos, vel)

# stream-core template (in-cell detection members), as the gate center
reg = np.load(REG_NPZ)
det = np.asarray(reg["detection_pids"]).astype(np.int64)
mem_det = np.isin(pid_raw, det)
cth = pos[:, 2] / r
phi = np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi
in_cell = mem_det & (cth < -0.6) & (phi >= np.pi / 2) & (phi < np.pi)
template = {k: float(v[k][in_cell].mean()) for k, _, _ in COMP}
print("recomputed stream-core template (km/s):",
      {k: round(x, 1) for k, x in template.items()}, flush=True)

# sanity: template must match the frozen v2 run (100 km/s units)
ref = {"vr": 0.32, "vth": -1.746, "vT": -2.05}
for k in ref:
    assert abs(template[k] / 100.0 - ref[k]) < 0.02, f"template drift in {k}"
print("template matches frozen v2 run (e94c3979) within 2 km/s", flush=True)


def stage_masks(lo, hi):
    in_bin = (r >= lo) & (r < hi)
    m0 = in_bin
    m1 = in_bin & is_union
    m2 = in_bin & is_union & ~is_debris
    return m0, m1, m2


def stats(mask):
    w = mass[mask]
    return dict(n=int(mask.sum()),
                mass_1e6_msun=float(w.sum() / 1e6),
                vT_mean_kms=float((w * v["vT"][mask]).sum() / w.sum()) if w.sum() else None,
                frac_vT_lt_minus100=float(w[v["vT"][mask] < VT_CUT].sum() / w.sum()) if w.sum() else None)


rows = []
print("\n=== per-bin cascade (mass in 1e6 Msun) ===", flush=True)
for lo, hi in CTX_BINS:
    m0, m1, m2 = stage_masks(lo, hi)
    s0, s1, s2 = stats(m0), stats(m1), stats(m2)
    rem1 = s0["mass_1e6_msun"] - s1["mass_1e6_msun"]
    rem2 = s1["mass_1e6_msun"] - s2["mass_1e6_msun"]
    print(f"bin {lo:.0f}-{hi:.0f} kpc: raw n={s0['n']} m={s0['mass_1e6_msun']:.2f} | "
          f"stage1 removed m={rem1:.2f} ({100 * rem1 / s0['mass_1e6_msun']:.1f}%) | "
          f"stage2 removed m={rem2:.2f} ({100 * rem2 / s0['mass_1e6_msun']:.1f}% of raw) | "
          f"kept n={s2['n']} m={s2['mass_1e6_msun']:.2f}", flush=True)
    print(f"   frac(vT<-100 km/s): raw={s0['frac_vT_lt_minus100']:.3f} "
          f"after1={s1['frac_vT_lt_minus100']:.3f} after2={s2['frac_vT_lt_minus100']:.3f} | "
          f"mean vT: raw={s0['vT_mean_kms']:+.1f} after1={s1['vT_mean_kms']:+.1f} "
          f"after2={s2['vT_mean_kms']:+.1f} km/s", flush=True)
    for key, _ in COMP:
        w1_02 = wasserstein_distance(v[key][m0], v[key][m2], mass[m0], mass[m2])
        w1_01 = wasserstein_distance(v[key][m0], v[key][m1], mass[m0], mass[m1])
        w1_12 = wasserstein_distance(v[key][m1], v[key][m2], mass[m1], mass[m2])
        rows.append(dict(bin_lo_kpc=lo, bin_hi_kpc=hi, comp=key,
                         w1_raw_vs_after2_kms=w1_02, w1_raw_vs_after1_kms=w1_01,
                         w1_after1_vs_after2_kms=w1_12))
        if (lo, hi) == (R_LO, R_HI):
            print(f"   W1({key}): raw->after1 {w1_01:.1f}, after1->after2 {w1_12:.1f}, "
                  f"raw->after2 {w1_02:.1f} km/s", flush=True)

# per-bin stage stats for the CSV
stat_rows = []
for lo, hi in CTX_BINS:
    m0, m1, m2 = stage_masks(lo, hi)
    for stage, m in (("raw", m0), ("after1", m1), ("after2", m2)):
        stat_rows.append(dict(bin_lo_kpc=lo, bin_hi_kpc=hi, stage=stage, **stats(m)))

# ---- figure: raw histogram decomposed into stage-1 cut, stage-2 cut, kept ----
use_style()
fig, axes = figure_grid(1, 3, width=WIDE, ratio=0.36, sharey=True)
m0, m1, m2 = stage_masks(R_LO, R_HI)
m_rem1 = m0 & ~m1
m_rem2 = m1 & ~m2
centers = 0.5 * (V_EDGES[:-1] + V_EDGES[1:])

for ax, (key, label) in zip(axes, COMP):
    def hist(mask):
        h, _ = np.histogram(v[key][mask], bins=V_EDGES, weights=mass[mask])
        return h / 1e6

    kept = hist(m2)
    cut2 = hist(m_rem2)
    cut1 = hist(m_rem1)
    total = hist(m0)
    # stacked decomposition of the raw (no-removal) histogram
    ax.bar(centers, kept, width=np.diff(V_EDGES), color=PALETTE["blue"],
           edgecolor="none", label="after both removals", zorder=3)
    ax.bar(centers, cut2, width=np.diff(V_EDGES), bottom=kept, color=PALETTE["orange"],
           edgecolor="none", label="removed by velocity gate (stage 2)", zorder=3)
    ax.bar(centers, cut1, width=np.diff(V_EDGES), bottom=kept + cut2, color=MUTED,
           edgecolor="none", label="removed by detection+smooth (stage 1)", zorder=3)
    ax.step(np.append(V_EDGES, V_EDGES[-1]), np.append(total, total[-1]),
            where="post", color="black", lw=1.0, label="no removal (raw)", zorder=4)
    ax.axvline(template[key], color=BASELINE, lw=0.8, ls=(0, (2, 2)), zorder=2,
               label="stream-core template (gate center)")
    ax.set_xlabel(label + " (km/s)")
    ax.set_xlim(-450, 450)
    ax.set_ylim(0, None)
axes[0].set_ylabel(r"mass per bin ($10^6\,M_\odot$)")
panel_labels(axes)
handles, labels = axes[0].get_legend_handles_labels()
order = [3, 0, 1, 2, 4]
try:
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", ncols=3, fontsize=6.5)
except (TypeError, ValueError):
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   fontsize=5.5, loc="upper left")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
stem = str(OUT / "figs" / "fig_debris_cascade_55_65")
save(fig, stem, formats=("png", "pdf", "svg"))

import csv  # noqa: E402
with open(OUT / "debris_cascade_stats.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(stat_rows[0].keys()))
    w.writeheader()
    w.writerows(stat_rows)
with open(OUT / "debris_cascade_w1.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(OUT / "summary.json", "w") as fj:
    json.dump(dict(bins_kpc=CTX_BINS, focus_bin_kpc=[R_LO, R_HI],
                   template_kms=template, stats=stat_rows, w1=rows,
                   inputs=dict(raw=str(RAW_H5), union=str(UNION_H5),
                               debris=str(DEBRIS_NPZ),
                               n_debris=int(len(debris_pids)))), fj, indent=1)
print("\noutputs under", OUT, flush=True)
print("DONE", flush=True)
