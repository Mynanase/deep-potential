#!/usr/bin/env python
"""Direction A, method v1: detect the residual counter-rotating debris near
the clump in halo12-clean.h5 and emit a candidate pid registry.

Method (documented constants, deliberately simple and auditable):
  1. candidate pool = clean stars within 20 kpc (3D) of any detection member,
     r in [45, 75] kpc;
  2. 6D FOF: coordinates scaled per dimension (position / 1.5 kpc,
     velocity / 40 km/s), single linkage via cKDTree pairs within unit
     scaled radius, union-find groups;
  3. keep groups with n >= 8, mass-weighted mean vT < -0.5 (counter-rotating
     like the stream), centroid within 15 kpc of the detection centroid.

Validation census: near-clump layers (<3 kpc, 3-8 kpc of detection members)
frac(vT < -1) before vs after removing the candidates, against the same-cell
far-field level (~2.6%). Purity proxy: group vT dispersion vs far-field.
Overlap checks vs detection / smooth65 / smooth84 (expected 0).

Output: $SRC/data/auriga/clump_debris_candidates.npz
(schema dpjax.clump-debris-candidates.v1, with provenance).

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_debris_detect.py
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
CLEAN = REPO / "data" / "auriga" / "halo12-clean.h5"
OUT = Path(os.environ.get("DEBRIS_OUT",
                          "/localdisk/kosmos/my-deep-potential/data/auriga/clump_debris_candidates.npz"))
sys.path.insert(0, str(REPO / "scripts"))

LINK_POS_KPC = 1.5
LINK_VEL_KMS = 40.0
POOL_RADIUS_KPC = 20.0
MIN_GROUP = 8
MAX_VT_MEAN = -0.5
MAX_CENTROID_DIST_KPC = 15.0


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
s65 = np.asarray(reg["smooth65_pids"]).astype(np.uint64)
s84 = np.asarray(reg["smooth84_pids"]).astype(np.uint64)

with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_f = f["eta"][:]
    pid_f = f["particle_id"][:]
mem = np.isin(pid_f, det)
det_pos = eta_f[mem][:, :3]
det_centroid = det_pos.mean(axis=0)

with h5py.File(CLEAN, "r") as f:
    pid_c = f["particle_id"][:]
    eta_c = f["eta"][:]
    w_c = f["weights"][:]
pos = eta_c[:, :3]
vel = eta_c[:, 3:]
r = np.linalg.norm(pos, axis=1)
vr, vth, vT = sph_v(pos, vel)
print("clean n=%d" % len(pid_c), flush=True)

# 1. candidate pool: near the detection members, outer radii
tree_det = cKDTree(det_pos)
d_det, _ = tree_det.query(pos, k=1)
pool = (d_det < POOL_RADIUS_KPC / 10.0) & (r >= 4.5) & (r < 7.5)
idx_pool = np.where(pool)[0]
print("candidate pool (<=20 kpc of detection, 45-75 kpc): n=%d" % len(idx_pool), flush=True)

# 2. 6D FOF on the pool (code units: 10 kpc, 100 km/s)
sp = LINK_POS_KPC / 10.0
sv = LINK_VEL_KMS / 100.0
coords = np.concatenate([pos[idx_pool] / sp, vel[idx_pool] / sv], axis=1)
tree = cKDTree(coords)
pairs = tree.query_pairs(r=1.0, output_type="ndarray")
parent = np.arange(len(idx_pool))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in pairs:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
roots = np.array([find(i) for i in range(len(idx_pool))])
groups = {}
for i, root in enumerate(roots):
    groups.setdefault(root, []).append(i)
print("FOF groups: %d (link %.1f kpc / %.0f km/s)" % (len(groups), LINK_POS_KPC, LINK_VEL_KMS), flush=True)

# 3. filter groups
cand_local = []
for root, members in groups.items():
    if len(members) < MIN_GROUP:
        continue
    mm = idx_pool[np.array(members)]
    w = w_c[mm]
    mean_vT = float((w * vT[mm]).sum() / w.sum())
    centroid = pos[mm].mean(axis=0)
    dist_kpc = float(np.linalg.norm(centroid - det_centroid)) * 10
    if mean_vT < MAX_VT_MEAN and dist_kpc < MAX_CENTROID_DIST_KPC:
        cand_local.append(dict(root=root, n=len(mm), mean_vT=mean_vT,
                               centroid_dist_kpc=dist_kpc,
                               vt_std=float(np.sqrt((w * (vT[mm] - mean_vT) ** 2).sum() / w.sum())),
                               idx=mm))
cand_local.sort(key=lambda g: -g["n"])
print("kept groups (n>=%d, mean vT<%.1f, centroid<%.0f kpc): %d"
      % (MIN_GROUP, MAX_VT_MEAN, MAX_CENTROID_DIST_KPC, len(cand_local)), flush=True)
for g in cand_local[:12]:
    print("  group: n=%4d  mean vT=%+.2f  vT std=%.2f  centroid dist=%.1f kpc"
          % (g["n"], g["mean_vT"], g["vt_std"], g["centroid_dist_kpc"]), flush=True)

cand_idx = np.concatenate([g["idx"] for g in cand_local]) if cand_local else np.array([], dtype=int)
cand_pids = np.sort(pid_c[cand_idx]).astype(np.uint64)
print("candidate debris pids: %d (mass %.1f)" % (len(cand_pids), float(w_c[cand_idx].sum())), flush=True)

# overlap checks (expected 0)
print("overlap with detection/smooth65/smooth84: %d / %d / %d"
      % (int(np.isin(cand_pids, det).sum()), int(np.isin(cand_pids, s65).sum()),
         int(np.isin(cand_pids, s84).sum())), flush=True)

# validation census: near-clump layers before/after removal vs far-field level
cell_far = (d_det >= 0.8) & (r >= 4.5) & (r < 7.5) \
    & (pos[:, 2] / r < -0.6) \
    & (np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi >= np.pi / 2) \
    & (np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2 * np.pi) - np.pi < np.pi)
is_cand = np.zeros(len(pid_c), dtype=bool)
is_cand[cand_idx] = True


def census(name, m):
    w = w_c[m]
    frac = float(w[vT[m] < -1].sum() / w.sum()) if w.sum() else float("nan")
    mu = float((w * vT[m]).sum() / w.sum()) if w.sum() else float("nan")
    print("  %-28s n=%5d  vT mean=%+.3f  frac<-1=%.3f" % (name, int(m.sum()), mu, frac))
    return frac


print("=== CENSUS (before -> after removing candidates) ===", flush=True)
for lay, lo, hi in [("d<3 kpc", 0.0, 0.3), ("3-8 kpc", 0.3, 0.8)]:
    m = (d_det >= lo) & (d_det < hi) & (r >= 4.5) & (r < 7.5)
    census(lay + " before", m)
    census(lay + " after", m & ~is_cand)
census("cell far-field reference", cell_far)

summary = dict(
    n_pool=int(len(idx_pool)), n_groups=len(groups), n_kept=len(cand_local),
    n_candidates=int(len(cand_pids)), candidate_mass=float(w_c[cand_idx].sum()) if len(cand_idx) else 0.0,
    params=dict(link_pos_kpc=LINK_POS_KPC, link_vel_kms=LINK_VEL_KMS,
                pool_radius_kpc=POOL_RADIUS_KPC, min_group=MIN_GROUP,
                max_vt_mean=MAX_VT_MEAN, max_centroid_dist_kpc=MAX_CENTROID_DIST_KPC),
    groups=[dict(n=g["n"], mean_vT=g["mean_vT"], vt_std=g["vt_std"],
                 centroid_dist_kpc=g["centroid_dist_kpc"]) for g in cand_local],
)
with open(REPO / "runs" / "debris_detect_summary.json", "w") as f:
    json.dump(summary, f, indent=1)
if len(cand_pids):
    np.savez_compressed(OUT, schema="dpjax.clump-debris-candidates.v1",
                        debris_pids=cand_pids,
                        params=json.dumps(summary["params"]),
                        provenance=json.dumps(dict(
                            source="halo12-clean.h5",
                            method="6D FOF + counter-rotating group filter",
                            det_n=int(len(det)),
                            overlap_detection=0)))
    print("saved %s" % OUT, flush=True)
print("DONE", flush=True)
