#!/usr/bin/env python
"""Direction A v2: velocity-gate detection of the residual debris.

v1 showed the near-clump residual is a diffuse spray (6D FOF caught only the
129 tightest stars). v2 gates VELOCITY space around the stream-core template
(from the in-cell detection members): pool = clean stars within 20 kpc of any
detection member (r 45-75 kpc); keep stars whose (vT, vr, vth) lie within
N_SIGMA (default 2.5) of the template in units of the template's per-component
dispersion. Purity control: the same gate applied to the far-field population
(same cell, d >= 8 kpc, r 45-75) estimates field contamination; the reported
census validates removal against the field level.

Output: $SRC/data/auriga/clump_debris_candidates_v2.npz
"""

import os
import sys
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
REG = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = Path(os.environ.get("DEBRIS_OUT",
                          "/localdisk/kosmos/my-deep-potential/data/auriga/clump_debris_candidates_v2.npz"))

N_SIGMA = 2.5
POOL_RADIUS_KPC = 20.0


def sph_v(pos, vel):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel[:, 0] + y * vel[:, 1] + z * vel[:, 2]) / r
    vth = (z * vr - r * vel[:, 2]) / R
    vT = (-vel[:, 0] * y + vel[:, 1] * x) / R
    return vr, vth, vT


reg = np.load(REG)
det = np.asarray(reg["detection_pids"]).astype(np.uint64)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_f = f["eta"][:]
    pid_f = f["particle_id"][:]
mem = np.isin(pid_f, det)
det_pos = eta_f[mem][:, :3]
r_det = np.linalg.norm(det_pos, axis=1)
cth_det = det_pos[:, 2] / r_det
phi_det = np.mod(np.arctan2(det_pos[:, 1], det_pos[:, 0]), 2 * np.pi) - np.pi
in_cell = (cth_det < -0.6) & (phi_det >= np.pi / 2) & (phi_det < np.pi)
vr_t, vth_t, vT_t = sph_v(det_pos, eta_f[mem][:, 3:])
tpl = dict(vr=(float(vr_t[in_cell].mean()), float(vr_t[in_cell].std())),
           vth=(float(vth_t[in_cell].mean()), float(vth_t[in_cell].std())),
           vT=(float(vT_t[in_cell].mean()), float(vT_t[in_cell].std())))
print("stream-core template (in-cell members):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in tpl.items()}, flush=True)

with h5py.File(REPO / "data" / "auriga" / "halo12-clean.h5", "r") as f:
    pid_c = f["particle_id"][:]
    eta_c = f["eta"][:]
    w_c = f["weights"][:]
pos = eta_c[:, :3]
vel = eta_c[:, 3:]
r = np.linalg.norm(pos, axis=1)
vr, vth, vT = sph_v(pos, vel)

tree = cKDTree(det_pos)
d_det, _ = tree.query(pos, k=1)
pool = (d_det < POOL_RADIUS_KPC / 10.0) & (r >= 4.5) & (r < 7.5)
far = (d_det >= 0.8) & (r >= 4.5) & (r < 7.5) & (pos[:, 2] / r < -0.6) \
    & (np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi >= np.pi / 2) \
    & (np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi < np.pi)


def gate(mask):
    g = np.ones(int(mask.sum()), dtype=bool)
    for comp, arr in (("vr", vr), ("vth", vth), ("vT", vT)):
        mu, sd = tpl[comp]
        g &= np.abs(arr[mask] - mu) < N_SIGMA * sd
    return g


g_pool = gate(pool)
cand_idx = np.where(pool)[0][g_pool]
is_cand = np.zeros(len(pid_c), dtype=bool)
is_cand[cand_idx] = True
n_far_gate = int(gate(far).sum())
far_n = int(far.sum())
print("pool n=%d, gated candidates n=%d (mass %.1f)" % (pool.sum(), len(cand_idx), float(w_c[cand_idx].sum())), flush=True)
print("purity control: far-field n=%d, same-gate hits n=%d (%.1f%%) -> field contamination scale"
      % (far_n, n_far_gate, 100 * n_far_gate / max(far_n, 1)), flush=True)


def census(name, m):
    w = w_c[m]
    frac = float(w[vT[m] < -1].sum() / w.sum()) if w.sum() else float("nan")
    mu = float((w * vT[m]).sum() / w.sum()) if w.sum() else float("nan")
    print("  %-28s n=%5d  vT mean=%+.3f  frac<-1=%.3f" % (name, int(m.sum()), mu, frac))


print("=== CENSUS (before -> after removing v2 candidates) ===", flush=True)
for lay, lo, hi in [("d<3 kpc", 0.0, 0.3), ("3-8 kpc", 0.3, 0.8)]:
    m = (d_det >= lo) & (d_det < hi) & (r >= 4.5) & (r < 7.5)
    census(lay + " before", m)
    census(lay + " after", m & ~is_cand)
census("far-field reference (same gate scale)", far)

cand_pids = np.sort(pid_c[cand_idx]).astype(np.uint64)
print("overlap with detection/smooth65/smooth84: %d / %d / %d"
      % (int(np.isin(cand_pids, det).sum()),
         int(np.isin(cand_pids, np.asarray(reg["smooth65_pids"]).astype(np.uint64)).sum()),
         int(np.isin(cand_pids, np.asarray(reg["smooth84_pids"]).astype(np.uint64)).sum())), flush=True)
summary = dict(n_pool=int(pool.sum()), n_candidates=int(len(cand_pids)),
               candidate_mass=float(w_c[cand_idx].sum()),
               n_sigma=N_SIGMA, template=tpl,
               far_field=dict(n=far_n, gate_hits=n_far_gate))
with open(REPO / "runs" / "debris_v2_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
if len(cand_pids):
    np.savez_compressed(OUT, schema="dpjax.clump-debris-candidates.v2",
                        debris_pids=cand_pids,
                        params=json.dumps(dict(n_sigma=N_SIGMA, pool_radius_kpc=POOL_RADIUS_KPC)),
                        provenance=json.dumps(dict(source="halo12-clean.h5",
                                                    method="velocity gate on stream-core template")))
    print("saved %s" % OUT, flush=True)
print("DONE", flush=True)
