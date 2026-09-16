#!/usr/bin/env python
"""Phase-1 DF audit, step 3/3 (v2, rewritten after the 2026-09-16 independent
review): metrics, bootstrap errors, noise-floor calibration, spatial check
against each model's OWN input population.

v2 changes vs v1 (see REPORT_v1_superseded.md):
  * direction-cell metrics now use the full-coverage single draw at EVERY
    strict position (vel_samples_all), not the 4284 stratified positions --
    v1 direction cells in the outer bins had as few as 47 positions
  * W1 "noise floor" calibration: truth-vs-truth W1 on random half splits of
    each region (a reference scale, NOT a formal error bar)
  * spatial flow is compared against each model's OWN full input population
    (halo12.h5 for w128, halo12-clean.h5 for w512/w1024). v1 wrongly used the
    strict intersection, which is radially biased (86.2% mass within 10 kpc
    vs 73% in the populations) because the two file orderings are strongly
    correlated in the inner region (corr 0.89) and decorrelate outward
  * samples with r > 7.5 are counted in a separate overflow bucket (v1
    clipped them into the last bin)
  * global FM reduction decomposed into per-bin contributions (strict weights)
  * `nll_*` fields are log p(v|x) (higher = better); paired dNLL bootstrap
    means are bootstrap means of the difference, not differences of means

Metric definitions (code units; velocities x100 km/s, radii x10 kpc):
  per radial bin: true = ALL strict-set stars in the bin with mass weights;
  gen (sample A) = K draws at the stratified positions, weighted by gen_w;
  gen (sample B, direction cells) = ONE draw at EVERY strict position in the
  cell, weighted by the same star masses as the truth side (perfectly paired).

Run from repo root:  python scripts/auriga/df_phase1_metrics.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"

B_BOOT = 200
B_BOOT_CELL = 100
RNG = np.random.default_rng(20260916)

prot = np.load(OUT / "eval_protocol.npz")
R_EDGES = prot["r_edges"]
CTH_EDGES = prot["cth_edges"]
PHI_EDGES = prot["phi_edges"]
ir = prot["ir_strict"]
ic = prot["ic_strict"]
ip = prot["ip_strict"]
w_strict = prot["w_strict"]
v_true = np.stack([prot["vr_strict"], prot["vth_strict"], prot["vT_strict"]], axis=1)
sel_idx = prot["sel_idx"]
gen_w_sel = prot["gen_w"]
K = int(prot["k_draws"])
ir_sel = ir[sel_idx]
pos_sel = prot["eta_strict"][sel_idx, :3]
eta_strict = prot["eta_strict"]
COMP = ["vr", "vth", "vT"]
MODELS = ["w128", "w512", "w1024"]
MODEL_INPUT = {"w128": "data/auriga/halo12.h5",
               "w512": "data/auriga/halo12-clean.h5",
               "w1024": "data/auriga/halo12-clean.h5"}
n_rbins = len(R_EDGES) - 1


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))

def wstd(a, w):
    mu = wmean(a, w)
    return float(np.sqrt(np.sum(w * (a - mu) ** 2) / np.sum(w)))

def corr_from(a, w):
    mu = np.sum(w[:, None] * a, axis=0) / np.sum(w)
    d = a - mu
    C = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / np.sum(w)
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)

gen = {}
gen_all = {}
for m in MODELS:
    d = np.load(OUT / f"model_{m}.npz")
    v_sph = sph_vel(np.repeat(pos_sel, K, axis=0), d["vel_samples"].reshape(-1, 3))
    gen[m] = dict(
        v=v_sph.reshape(len(sel_idx), K, 3),
        nll=d["nll_strict"], nll_cval=d["nll_cval"],
        fm=(d["fm_resid_strict"] ** 2).sum(-1),
        fm_cval=(d["fm_resid_cval"] ** 2).sum(-1),
        pos_samples=d["pos_samples"],
    )
    # sample B: one draw at every strict position (v2)
    gen_all[m] = sph_vel(eta_strict[:, :3], d["vel_samples_all"])

# ------------------------------------------------------------------
# 1. per-radial-bin metrics + paired bootstrap (sample A, as v1)
# ------------------------------------------------------------------
rows = []
for b in range(n_rbins):
    m_true = ir == b
    n_b = int(m_true.sum())
    w_b = w_strict[m_true]
    v_b = v_true[m_true]
    m_sel = ir_sel == b
    n_sel_b = int(m_sel.sum())
    gw = np.repeat(gen_w_sel[m_sel], K)
    v_g = {m: gen[m]["v"][m_sel].reshape(-1, 3) for m in MODELS}

    row = dict(bin=b, r_lo=R_EDGES[b], r_hi=R_EDGES[b + 1], n_true=n_b,
               n_eff=float(w_b.sum() ** 2 / np.sum(w_b**2)), n_gen_positions=n_sel_b)
    for c, cname in enumerate(COMP):
        row[f"{cname}_true_mean"] = wmean(v_b[:, c], w_b)
        row[f"{cname}_true_std"] = wstd(v_b[:, c], w_b)
    for m in MODELS:
        row[f"logp_{m}"] = wmean(gen[m]["nll"][m_true], w_b)
        row[f"fm_{m}"] = wmean(gen[m]["fm"][m_true].mean(axis=1), w_b)
        for c, cname in enumerate(COMP):
            row[f"w1_{cname}_{m}"] = wasserstein_distance(
                v_b[:, c], v_g[m][:, c], u_weights=w_b, v_weights=gw)
            row[f"bias_mean_{cname}_{m}"] = wmean(v_g[m][:, c], gw) - row[f"{cname}_true_mean"]
            row[f"bias_std_{cname}_{m}"] = wstd(v_g[m][:, c], gw) - row[f"{cname}_true_std"]
        Cg = corr_from(v_g[m], gw)
        Ct = corr_from(v_b, w_b)
        for i in range(3):
            for j in range(i + 1, 3):
                row[f"corr_{COMP[i]}{COMP[j]}_{m}"] = float(Cg[i, j])
                if m == MODELS[0]:
                    row[f"corr_{COMP[i]}{COMP[j]}_true"] = float(Ct[i, j])

    idx_true = np.where(m_true)[0]
    idx_sel = np.where(m_sel)[0]
    boot_w1 = {m: {c: [] for c in range(3)} for m in MODELS}
    boot_nll = {m: [] for m in MODELS}
    for _ in range(B_BOOT):
        it = RNG.choice(idx_true, size=n_b, replace=True)
        gsel = RNG.choice(idx_sel, size=n_sel_b, replace=True)
        gw_r = np.repeat(gen_w_sel[gsel], K)
        wb_r = w_strict[it]
        vb_r = v_true[it]
        for m in MODELS:
            for c in range(3):
                vg_c = gen[m]["v"][gsel][:, :, c].reshape(-1)
                boot_w1[m][c].append(wasserstein_distance(
                    vb_r[:, c], vg_c, u_weights=wb_r, v_weights=gw_r))
            boot_nll[m].append(wmean(gen[m]["nll"][it], wb_r))
    for m in MODELS:
        for c, cname in enumerate(COMP):
            row[f"w1_{cname}_{m}_err"] = float(np.std(boot_w1[m][c]))
        row[f"logp_{m}_err"] = float(np.std(boot_nll[m]))
        if m != "w128":
            for c, cname in enumerate(COMP):
                darr = np.array(boot_w1[m][c]) - np.array(boot_w1["w128"][c])
                row[f"dw1_{cname}_{m}"] = float(np.mean(darr))   # bootstrap mean of the difference
                row[f"dw1_{cname}_{m}_err"] = float(np.std(darr))
            darr = np.array(boot_nll[m]) - np.array(boot_nll["w128"])
            row[f"dlogp_{m}"] = float(np.mean(darr))
            row[f"dlogp_{m}_err"] = float(np.std(darr))
    rows.append(row)
    print(f"radial bin {b} done")

df_bin = pd.DataFrame(rows)
df_bin.to_csv(OUT / "per_bin_metrics.csv", index=False)

# ------------------------------------------------------------------
# 2. truth-vs-truth W1 noise floor (radial bins and direction cells)
#    reference scale: W1 between two random weighted halves of the truth
# ------------------------------------------------------------------
def floor_w1(vals, w, reps):
    n = len(vals)
    out = []
    for _ in range(reps):
        perm = RNG.permutation(n)
        h = perm[: n // 2]
        g = perm[n // 2:]
        out.append(wasserstein_distance(vals[h], vals[g],
                                        u_weights=w[h], v_weights=w[g]))
    return float(np.mean(out)), float(np.std(out))

floors = []
for b in range(n_rbins):
    m = ir == b
    row = dict(bin=b, kind="radial", n=int(m.sum()))
    for c, cname in enumerate(COMP):
        mu, sd = floor_w1(v_true[m, c], w_strict[m], B_BOOT_CELL)
        row[f"floor_{cname}"] = mu
        row[f"floor_{cname}_sd"] = sd
    floors.append(row)

# ------------------------------------------------------------------
# 3. direction cells from sample B (full coverage, paired weights)
# ------------------------------------------------------------------
rows_dir = []
for b in range(n_rbins):
    for a_kind, a_idx, a_edges in [("cth", ic, CTH_EDGES), ("phi", ip, PHI_EDGES)]:
        for a in range(len(a_edges) - 1):
            mt = (ir == b) & (a_idx == a)
            row = dict(bin=b, r_lo=R_EDGES[b], r_hi=R_EDGES[b + 1],
                       axis=a_kind, cut_lo=a_edges[a], cut_hi=a_edges[a + 1],
                       n_true=int(mt.sum()))
            # noise floor from truth halves
            for c, cname in enumerate(COMP):
                mu, sd = floor_w1(v_true[mt, c], w_strict[mt], 50)
                row[f"floor_{cname}"] = mu
            if mt.sum() >= 100:
                w_b = w_strict[mt]
                v_b = v_true[mt]
                row["n_eff"] = float(w_b.sum() ** 2 / np.sum(w_b**2))
                for c, cname in enumerate(COMP):
                    row[f"{cname}_true_mean"] = wmean(v_b[:, c], w_b)
                    row[f"{cname}_true_std"] = wstd(v_b[:, c], w_b)
                    for m in MODELS:
                        vg = gen_all[m][mt, c]
                        row[f"w1_{cname}_{m}"] = wasserstein_distance(
                            v_b[:, c], vg, u_weights=w_b, v_weights=w_b)
                        row[f"bias_mean_{cname}_{m}"] = wmean(vg, w_b) - row[f"{cname}_true_mean"]
                        row[f"bias_std_{cname}_{m}"] = wstd(vg, w_b) - row[f"{cname}_true_std"]
                for m in MODELS:
                    row[f"logp_{m}"] = wmean(gen[m]["nll"][mt], w_b)
                # paired bootstrap of dW1 and dlogp vs w128 (same resampled stars)
                idx = np.where(mt)[0]
                n = len(idx)
                bw = {m: {c: [] for c in range(3)} for m in MODELS}
                bn = {m: [] for m in MODELS}
                for _ in range(B_BOOT_CELL):
                    it = idx[RNG.integers(0, n, n)]
                    wb_r = w_strict[it]
                    for m in MODELS:
                        for c in range(3):
                            bw[m][c].append(wasserstein_distance(
                                v_true[it, c], gen_all[m][it, c],
                                u_weights=wb_r, v_weights=wb_r))
                        bn[m].append(wmean(gen[m]["nll"][it], wb_r))
                for m in MODELS:
                    if m == "w128":
                        continue
                    for c, cname in enumerate(COMP):
                        darr = np.array(bw[m][c]) - np.array(bw["w128"][c])
                        row[f"dw1_{cname}_{m}"] = float(np.mean(darr))
                        row[f"dw1_{cname}_{m}_err"] = float(np.std(darr))
                    darr = np.array(bn[m]) - np.array(bn["w128"])
                    row[f"dlogp_{m}"] = float(np.mean(darr))
                    row[f"dlogp_{m}_err"] = float(np.std(darr))
            rows_dir.append(row)
df_dir = pd.DataFrame(rows_dir)
df_dir.to_csv(OUT / "direction_metrics.csv", index=False)
df_floor = pd.DataFrame(floors)
df_floor.to_csv(OUT / "w1_noise_floor.csv", index=False)

# ------------------------------------------------------------------
# 4. global metrics: strict-weighted AND clean-val-weighted; FM decomposition
# ------------------------------------------------------------------
glob = {}
gw_all = np.repeat(gen_w_sel, K)
glob["n_strict"] = len(w_strict)
idx_all = np.arange(len(w_strict))
for m in MODELS:
    vg = gen[m]["v"].reshape(-1, 3)
    glob[f"logp_strictw_{m}"] = wmean(gen[m]["nll"], w_strict)
    glob[f"logp_cvalw_{m}"] = wmean(gen[m]["nll_cval"], prot["w_cval"])
    glob[f"fm_strictw_{m}"] = wmean(gen[m]["fm"].mean(axis=1), w_strict)
    glob[f"fm_cvalw_{m}"] = wmean(gen[m]["fm_cval"].mean(axis=1), prot["w_cval"])
    for c, cname in enumerate(COMP):
        glob[f"w1_{cname}_{m}"] = wasserstein_distance(
            v_true[:, c], vg[:, c], u_weights=w_strict, v_weights=gw_all)
for m in ["w512", "w1024"]:
    d = []
    for _ in range(B_BOOT):
        it = RNG.choice(idx_all, size=len(idx_all), replace=True)
        d.append(wmean(gen[m]["nll"][it], w_strict[it]) - wmean(gen["w128"]["nll"][it], w_strict[it]))
    glob[f"dlogp_{m}_vs_w128_strictw"] = float(np.mean(d))
    glob[f"dlogp_{m}_vs_w128_strictw_err"] = float(np.std(d))
# FM reduction contribution per bin (strict-mass weights)
masses = np.array([w_strict[ir == b].sum() for b in range(n_rbins)])
fmass = masses / masses.sum()
dfm = (df_bin["fm_w128"].to_numpy() - df_bin["fm_w1024"].to_numpy())
contrib = dfm * fmass
glob["fm_reduction_contribution_bins"] = (contrib / contrib.sum()).tolist()
glob["note_contrib"] = "share of the strict-mass-weighted global FM reduction (w128->w1024) per radial bin"
with open(OUT / "global_metrics.json", "w") as f:
    json.dump(glob, f, indent=2)

# ------------------------------------------------------------------
# 5. spatial flow vs each model's OWN input population (v2)
# ------------------------------------------------------------------
R_EDGES_EXT = np.append(R_EDGES, np.inf)   # overflow bucket r > 7.5
spat_rows = []
margins = {}
for m in MODELS:
    with h5py.File(REPO / MODEL_INPUT[m], "r") as f:
        eta_ref = f["eta"][:]
        w_ref = f["weights"][:]
    r_ref = np.linalg.norm(eta_ref[:, :3], axis=1)
    ib_ref = np.clip(np.digitize(r_ref, R_EDGES) - 1, 0, n_rbins - 1)
    over_ref = int(np.sum(r_ref >= R_EDGES[-1]))
    f_ref = np.array([w_ref[ib_ref == b].sum() for b in range(n_rbins)]) / w_ref.sum()

    xyz = gen[m]["pos_samples"]
    r_s = np.linalg.norm(xyz, axis=1)
    ib_s = np.clip(np.digitize(r_s, R_EDGES) - 1, 0, n_rbins - 1)
    over_s = int(np.sum(r_s >= R_EDGES[-1]))
    f_s = np.array([np.mean(ib_s == b) for b in range(n_rbins)])

    # train/val consistency of the reference itself
    nv = int(0.25 * len(r_ref))
    f_tr = np.array([w_ref[nv:][ib_ref[nv:] == b].sum() for b in range(n_rbins)]) / w_ref[nv:].sum()
    f_va = np.array([w_ref[:nv][ib_ref[:nv] == b].sum() for b in range(n_rbins)]) / w_ref[:nv].sum()

    for b in range(n_rbins):
        spat_rows.append(dict(model=m, bin=b,
                              f_population=f_ref[b], f_train=f_tr[b], f_val=f_va[b],
                              f_model=f_s[b], reldev=f_s[b] / f_ref[b] - 1))
    spat_rows.append(dict(model=m, bin="overflow_r>7.5", f_population=over_ref / len(r_ref),
                          f_train=np.nan, f_val=np.nan,
                          f_model=over_s / len(r_s), reldev=np.nan))

    # angular margins vs own population (density-normalized, full peak range)
    cth_ref = eta_ref[:, 2] / r_ref
    phi_ref = np.mod(np.arctan2(eta_ref[:, 1], eta_ref[:, 0]), 2 * np.pi) - np.pi
    cth_s = xyz[:, 2] / r_s
    phi_s = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi) - np.pi
    bt_c = np.linspace(-1, 1, 41)
    bt_p = np.linspace(-np.pi, np.pi, 37)
    hc_ref, _ = np.histogram(cth_ref, bins=bt_c, weights=w_ref, density=True)
    hc_s, _ = np.histogram(cth_s, bins=bt_c, density=True)
    hp_ref, _ = np.histogram(phi_ref, bins=bt_p, weights=w_ref, density=True)
    hp_s, _ = np.histogram(phi_s, bins=bt_p, density=True)
    margins[m] = dict(cth_centers=((bt_c[:-1] + bt_c[1:]) / 2).tolist(),
                      cth_ref=hc_ref.tolist(), cth_model=hc_s.tolist(),
                      phi_centers=((bt_p[:-1] + bt_p[1:]) / 2).tolist(),
                      phi_ref=hp_ref.tolist(), phi_model=hp_s.tolist(),
                      cth_peak_ref=float(hc_ref.max()), cth_peak_model=float(hc_s.max()),
                      phi_peak_ref=float(hp_ref.max()), phi_peak_model=float(hp_s.max()))

df_spat = pd.DataFrame(spat_rows)
df_spat.to_csv(OUT / "spatial_population_occupancy.csv", index=False)
with open(OUT / "spatial_margins.json", "w") as f:
    json.dump(margins, f)

print("metrics v2 done")
print(df_bin[["bin", "n_true", "w1_vr_w128", "w1_vr_w512", "w1_vr_w1024",
              "logp_w128", "logp_w512", "logp_w1024"]].round(4).to_string(index=False))
print(json.dumps({k: glob[k] for k in glob if "logp" in k or "fm_reduction" in k}, indent=1))
print(df_spat.pivot(index="bin", columns="model", values="reldev").round(3).to_string())
