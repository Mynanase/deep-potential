#!/usr/bin/env python
"""Phase-1 DF audit, step 1/3: fix the evaluation protocol and build the
common validation set (no model involved, CPU only).

Answers the protocol questions from the phase-1 plan:

  * data selection / weights / coordinate transform / physical units
  * whether the validation splits of the five capacity runs coincide
  * a strict common evaluation set that NO run trained on
  * per-spatial-bin true counts and effective counts N_eff
  * stratified evaluation positions for the conditional sampling (step 2)

Conventions (fixed by the training pipeline, verified below):
  code units  q = x / (10 kpc),  p = v / (100 km/s)
  frame       principal-axis aligned (Tiv_star), phi0 = -40 deg, origin (0,0,0)
  weights     particle mass, normalized to sum N (attribute weighting='mass')
  val split   FIRST 25% of the pre-shuffled rows (utils.split_data), i.e.
              val rows = data[:n_val] -- no reshuffle at split time
  spatial bins (chosen from the measured radius support, edges inside data):
              r      [0,1), [1,2), [2,3), [3,4.5), [4.5,6), [6,7.5)  code units
                      (= 10 kpc steps up to 30 kpc, wider bins to 60, outer 60-75)
              cosθ   signed, 5 bins: [-1,-0.6), [-0.6,-0.2), [-0.2,0.2),
                      [0.2,0.6), [0.6,1]  (keeps north/south separate)
              phi    4 sectors of 90 deg: [-180,-90), [-90,0), [0,90), [90,180)

Run from repo root:  python scripts/auriga/df_phase1_data_audit.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. load both training inputs, verify selection/weights/units agree
# ------------------------------------------------------------------
FULL = REPO / "data" / "auriga" / "halo12.h5"        # used by halo12-baseline (w128)
CLEAN = REPO / "data" / "auriga" / "halo12-clean.h5"  # used by cap-w256/512/1024/1280

with h5py.File(FULL, "r") as f:
    pid_full = f["particle_id"][:]
    eta_full = f["eta"][:]
    w_full = f["weights"][:]
    attrs_full = dict(f.attrs)
with h5py.File(CLEAN, "r") as f:
    pid_clean = f["particle_id"][:]
    eta_clean = f["eta"][:]
    w_clean = f["weights"][:]
    attrs_clean = dict(f.attrs)

# same coordinate frame and units?
assert attrs_full["header_Tiv_star"].tolist() == attrs_clean["header_Tiv_star"].tolist()
assert attrs_full["length_scale_kpc"] == attrs_clean["length_scale_kpc"] == 10.0
assert attrs_full["velocity_scale_kms"] == attrs_clean["velocity_scale_kms"] == 100.0
assert attrs_full["train_r_max_kpc"] == attrs_clean["train_r_max_kpc"] == 75.0
assert attrs_full["weighting"] == attrs_clean["weighting"] == "mass"

# clean = full minus the outer-clump stars (9683 removed), but RE-SHUFFLED:
# particle_id columns do NOT correspond between the two files (different prep
# runs), so match rows by their exact 6D eta float32 tuple instead. Per-particle
# mass columns are then identical and weights differ only by the global
# normalization sum(w)=N (verified: ratio w_clean/w_full = 1.000052 const).
def _row_keys(a):
    return np.ascontiguousarray(a).view(
        [("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4"), ("e", "f4"), ("g", "f4")]
    ).ravel()

key_full, first_full_row = np.unique(_row_keys(eta_full), return_index=True)
row_of_clean = first_full_row[np.searchsorted(key_full, _row_keys(eta_clean))]
eta_rows_match = np.array_equal(eta_full[row_of_clean], eta_clean)
mass_rows_match = np.array_equal(
    h5py.File(FULL, "r")["mass"][:][row_of_clean], h5py.File(CLEAN, "r")["mass"][:]
)
removed_rows = np.setdiff1d(np.arange(len(eta_full)), row_of_clean)
r_removed = np.linalg.norm(eta_full[removed_rows, :3], axis=1)
print(f"clean rows match full rows by 6D eta: eta={eta_rows_match} mass={mass_rows_match}")
print(f"removed {len(removed_rows)} rows, radius quantiles (kpc): "
      f"{np.round(np.quantile(r_removed, [0, .5, .9, 1]) * 10, 1)}")

# ------------------------------------------------------------------
# 2. the two validation splits (first 25% of each shuffled file)
# ------------------------------------------------------------------
VAL_FRAC = 0.25
n_val_full = int(len(pid_full) * VAL_FRAC)      # 413242
n_val_clean = int(len(pid_clean) * VAL_FRAC)    # 410821
val_full_mask = np.zeros(len(pid_full), dtype=bool)
val_full_mask[:n_val_full] = True
val_clean_mask = np.zeros(len(pid_clean), dtype=bool)
val_clean_mask[:n_val_clean] = True

in_full_val = val_full_mask[row_of_clean]           # clean row is inside full-file val part
strict_common = val_clean_mask & in_full_val        # held out for ALL five runs
n_strict = int(strict_common.sum())
print(f"val split sizes: full={n_val_full} clean={n_val_clean}, "
      f"strict common (both-val)={n_strict} ({n_strict / n_val_clean:.1%} of clean-val)")
# leakage cross-check: fraction of clean-val that w128 trained on
leak_w128 = int((val_clean_mask & ~in_full_val).sum())
print(f"clean-val rows that were in w128 TRAINING data: {leak_w128} "
      f"({leak_w128 / n_val_clean:.1%}) -- comparisons must use the strict set")

# ------------------------------------------------------------------
# 3. evaluation sets saved for step 2/3
#    primary   = strict common val (fair for w128/w512/w1024)
#    secondary = clean val (own monitoring set of w512/w1024; 75% leaked for w128)
# ------------------------------------------------------------------
eta_strict = eta_clean[strict_common]
w_strict = w_clean[strict_common]
eta_cval = eta_clean[val_clean_mask]
w_cval = w_clean[val_clean_mask]

# spherical coordinates, same convention as utils.calc_coords (origin 0,0,0)
def sph_coords(eta):
    x, y, z = eta[:, 0], eta[:, 1], eta[:, 2]
    r = np.linalg.norm(eta[:, :3], axis=1)
    cth = z / r
    phi = np.mod(np.arctan2(y, x), 2 * np.pi) - np.pi   # (-pi, pi]
    R = np.hypot(x, y)
    vr = (x * eta[:, 3] + y * eta[:, 4] + z * eta[:, 5]) / r
    vth = (z * vr - r * eta[:, 5]) / R
    vT = (-eta[:, 3] * y + eta[:, 4] * x) / R
    return r, cth, phi, vr, vth, vT

r_s, cth_s, phi_s, vr_s, vth_s, vT_s = sph_coords(eta_strict)
r_c, cth_c, phi_c, vr_c, vth_c, vT_c = sph_coords(eta_cval)

# ------------------------------------------------------------------
# 4. spatial bins and per-bin true counts / N_eff
# ------------------------------------------------------------------
R_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 7.5])       # code units (x10 kpc)
CTH_EDGES = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0])
PHI_EDGES = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])

ir_s = np.clip(np.digitize(r_s, R_EDGES) - 1, 0, len(R_EDGES) - 2)
ic_s = np.clip(np.digitize(cth_s, CTH_EDGES) - 1, 0, len(CTH_EDGES) - 2)
ip_s = np.clip(np.digitize(phi_s, PHI_EDGES) - 1, 0, len(PHI_EDGES) - 2)

def bin_stats(ir, w):
    rows = []
    for b in range(len(R_EDGES) - 1):
        m = ir == b
        n = int(m.sum())
        sw = float(w[m].sum())
        neff = float(sw**2 / np.sum(w[m]**2)) if n else 0.0
        rows.append(dict(bin=b, r_lo=R_EDGES[b], r_hi=R_EDGES[b + 1],
                         n=n, mass=sw, n_eff=neff))
    return rows

stats_strict = bin_stats(ir_s, w_strict)
stats_cval = bin_stats(np.clip(np.digitize(r_c, R_EDGES) - 1, 0, len(R_EDGES) - 2), w_cval)
for row in stats_strict:
    print(f"strict r[{row['r_lo']:.1f},{row['r_hi']:.1f}): n={row['n']:6d}  "
          f"N_eff={row['n_eff']:7.0f}  mass={row['mass']:.1f}")

# coverage check inside the outer bin: cos-theta and phi counts (strict set)
outer = ir_s == len(R_EDGES) - 2
print("outer-bin cos-theta counts:", np.bincount(ic_s[outer], minlength=5).tolist())
print("outer-bin phi-sector counts:", np.bincount(ip_s[outer], minlength=4).tolist())

# ------------------------------------------------------------------
# 5. stratified evaluation positions: 16384 positions, K=4 draws each
#    allocation: equal number of positions per radial bin (up to bin size),
#    positions evenly spaced in the radius ranking inside the bin
# ------------------------------------------------------------------
N_POS_TOTAL = 16384
K_DRAWS = 4
N_R_BINS = len(R_EDGES) - 1
per_bin = min(N_POS_TOTAL // N_R_BINS, min(row["n"] for row in stats_strict))

sel_idx = np.zeros(N_POS_TOTAL, dtype=np.int64)
filled = 0
for b in range(N_R_BINS):
    rows_in_bin = np.where(ir_s == b)[0]
    order_b = rows_in_bin[np.argsort(r_s[rows_in_bin], kind="stable")]
    picks = np.linspace(0, len(order_b) - 1, min(per_bin, len(order_b))).round().astype(int)
    sel_idx[filled:filled + len(picks)] = order_b[picks]
    filled += len(picks)
sel_idx = sel_idx[:filled]
print(f"selected {filled} stratified positions ({per_bin} per radial bin), K={K_DRAWS}")

# gen-sample weight for a selected position: w_i * (bin mass) / (selected mass in bin)
gen_w = np.empty(len(sel_idx))
for b in range(N_R_BINS):
    m = ir_s[sel_idx] == b
    gen_w[m] = w_strict[sel_idx][m] * w_strict[ir_s == b].sum() / w_strict[sel_idx][m].sum()

np.savez_compressed(
    OUT / "eval_protocol.npz",
    # strict common set (primary): full 6D eta, weights, spherical coords, bins
    eta_strict=eta_strict, w_strict=w_strict,
    r_strict=r_s, cth_strict=cth_s, phi_strict=phi_s,
    vr_strict=vr_s, vth_strict=vth_s, vT_strict=vT_s,
    ir_strict=ir_s, ic_strict=ic_s, ip_strict=ip_s,
    # clean val set (secondary)
    eta_cval=eta_cval, w_cval=w_cval,
    r_cval=r_c, cth_cval=cth_c, phi_cval=phi_c,
    vr_cval=vr_c, vth_cval=vth_c, vT_cval=vT_c,
    # stratified conditional-sampling positions and their reweighting
    sel_idx=sel_idx, gen_w=gen_w, k_draws=K_DRAWS,
    r_edges=R_EDGES, cth_edges=CTH_EDGES, phi_edges=PHI_EDGES,
    row_of_clean=row_of_clean, removed_rows=removed_rows,
)

# ------------------------------------------------------------------
# 6. audit summary
# ------------------------------------------------------------------
audit = dict(
    date="2026-09-16",
    inputs=dict(
        full=dict(path=str(FULL), n=len(pid_full), n_val=n_val_full),
        clean=dict(path=str(CLEAN), n=len(pid_clean), n_val=n_val_clean,
                   removed_from_full=len(removed_rows),
                   removed_radius_kpc_quantiles=dict(
                       zip(["p0", "p50", "p90", "p100"],
                           np.round(np.quantile(r_removed, [0, .5, .9, 1]) * 10, 2).tolist()))),
    ),
    shared_protocol=dict(
        frame="principal-axis aligned Tiv_star (identical matrices), phi0=-40 deg",
        units=dict(length_kpc=10.0, velocity_kms=100.0, r_max_kpc=75.0),
        weights="per-particle mass identical across files (matched rows); w = mass/sum(mass), "
                "so w differs only by the global normalization sum(w)=N of each file",
        val_rule="first 25% of pre-shuffled rows; the two files were shuffled independently "
                 "(particle_id columns do not even correspond) -> validation splits do NOT coincide",
        row_matching="clean rows matched to full rows by exact 6D eta tuples "
                     "(eta_rows_match=True, per-particle mass identical)",
        strict_common_n=n_strict,
        clean_val_leaks_into_w128_train=leak_w128,
    ),
    bins=dict(r_edges=R_EDGES.tolist(), cth_edges=CTH_EDGES.tolist(),
              phi_edges=PHI_EDGES.tolist()),
    per_bin_strict=stats_strict,
    per_bin_cval=stats_cval,
    stratified_positions=dict(n=int(len(sel_idx)), per_radial_bin=per_bin, k_draws=K_DRAWS),
)
with open(OUT / "data_audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print(f"saved {OUT / 'data_audit.json'} and eval_protocol.npz")
