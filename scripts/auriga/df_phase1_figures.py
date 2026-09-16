#!/usr/bin/env python
"""Phase-1 DF audit, figures v2 (rewritten after the 2026-09-16 review).

v2 changes:
  * spatial panel compares each model against its OWN input population
    (halo12.h5 / halo12-clean.h5), with an overflow bucket r>7.5 listed
    separately and angular margins kept to their full peak height
  * W1 panels overlay the truth-vs-truth half-split noise floor
    (w1_noise_floor.csv) as a shaded reference band
  * direction heatmaps use the full-coverage sample-B metrics (v1 outer
    direction cells had as few as 47 positions and were noise-dominated)
  * outer panels use the FIXED outer confirm runs (v1 conditioning bug)
  * `logp_*` naming throughout (the arrays are log p, higher = better)

Manual y-limits everywhere (constants chosen from measured data ranges,
noted per figure); no autoscale.

Run from repo root:  python scripts/auriga/df_phase1_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7, "figure.constrained_layout.use": True})

MODELS = ["w128", "w512", "w1024"]
COLORS = {"w128": "#1f77b4", "w512": "#ff7f0e", "w1024": "#d62728"}
COMP = ["vr", "vth", "vT"]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-45", "45-60", "60-75"]  # kpc
BIN_CENTER = [5, 15, 25, 37.5, 52.5, 67.5]

prot = np.load(OUT / "eval_protocol.npz")
ir = prot["ir_strict"]
w_strict = prot["w_strict"]
v_true = np.stack([prot["vr_strict"], prot["vth_strict"], prot["vT_strict"]], axis=1)
sel_idx = prot["sel_idx"]
gen_w_sel = prot["gen_w"]
K = int(prot["k_draws"])
ir_sel = ir[sel_idx]
pos_sel = prot["eta_strict"][sel_idx, :3]


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


gen = {}
gen_all = {}
for m in MODELS:
    d = np.load(OUT / f"model_{m}.npz")
    v_sph = sph_vel(np.repeat(pos_sel, K, axis=0), d["vel_samples"].reshape(-1, 3))
    gen[m] = dict(v=v_sph.reshape(len(sel_idx), K, 3), pos_samples=d["pos_samples"])
    gen_all[m] = sph_vel(prot["eta_strict"][:, :3], d["vel_samples_all"])

df_bin = pd.read_csv(OUT / "per_bin_metrics.csv")
df_dir = pd.read_csv(OUT / "direction_metrics.csv")
df_floor = pd.read_csv(OUT / "w1_noise_floor.csv")
glob = json.load(open(OUT / "global_metrics.json"))

VBINS = np.arange(-4.0, 4.01, 0.1)
VC = (VBINS[:-1] + VBINS[1:]) / 2

# ------------------------------------------------------------------
# fig 1: per-bin true-vs-generated histograms (strict, sample A)
# measured ranges: true peaks <= 1.19 (vT 45-60), gen peaks <= 0.69
# ------------------------------------------------------------------
for c, cname in enumerate(COMP):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey=True)
    for b in range(6):
        ax = axs[b // 3, b % 3]
        mt = ir == b
        h, _ = np.histogram(v_true[mt, c], bins=VBINS, weights=w_strict[mt], density=True)
        ax.plot(VC, h, color="k", lw=1.4, label="truth (mass-wtd)")
        gwr = np.repeat(gen_w_sel[ir_sel == b], K)
        for m in MODELS:
            hg, _ = np.histogram(gen[m]["v"][ir_sel == b][:, :, c].reshape(-1),
                                 bins=VBINS, weights=gwr, density=True)
            ax.plot(VC, hg, color=COLORS[m], lw=0.9, alpha=0.85, label=m)
        ax.set_title(f"r {BIN_LABELS[b]} kpc  (n={int(mt.sum())})")
        ax.set_ylim(0.0, 1.35)      # manual: max peak 1.19 (true vT, 45-60 kpc)
        ax.set_xlim(-4.0, 4.0)      # manual: 0.1-99.9% range +-3.5..3.8
        if b % 3 == 0:
            ax.set_ylabel("p(v)  [(100 km/s)$^{-1}$]")
        if b // 3 == 1:
            ax.set_xlabel(f"${cname}$  [100 km/s]")
    axs[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"strict set: conditional ${cname}$ by radius bin (stratified positions, K=4, shared noise)")
    fig.savefig(FIG / f"fig_hist_bins_{cname}.png")
    plt.close(fig)

# ------------------------------------------------------------------
# fig 2: W1 per bin + paired dW1, with truth-vs-truth noise floor
# measured: W1 <= 0.090, floors 0.005..0.097, dW1 in +-0.022
# ------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(8.6, 7.4), sharex=True)
for c, cname in enumerate(COMP):
    ax = axs[c, 0]
    fl = df_floor[f"floor_{cname}"].to_numpy()
    fl_sd = df_floor[f"floor_{cname}_sd"].to_numpy()
    ax.fill_between(BIN_CENTER, np.maximum(fl - fl_sd, 0), fl + fl_sd,
                    color="0.85", label="truth-vs-truth floor")
    for m in MODELS:
        y = df_bin[f"w1_{cname}_{m}"]
        e = df_bin[f"w1_{cname}_{m}_err"]
        ax.errorbar(BIN_CENTER, y, yerr=e, color=COLORS[m], marker="o", ms=3,
                    lw=1, label=m, capsize=2)
    ax.set_ylabel(f"$W_1(${cname}$)$")
    ax.set_ylim(0.0, 0.14)          # manual: W1 max 0.090, floor+sd max 0.13
    ax = axs[c, 1]
    for m in ["w512", "w1024"]:
        y = df_bin[f"dw1_{cname}_{m}"]
        e = df_bin[f"dw1_{cname}_{m}_err"]
        ax.errorbar(BIN_CENTER, y, yerr=e, color=COLORS[m], marker="o", ms=3,
                    lw=1, label=f"{m}-w128", capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(f"$\\Delta W_1$ vs w128")
    ax.set_ylim(-0.05, 0.05)        # manual: paired dW1 range +-0.022
for ax in axs[-1]:
    ax.set_xlabel("r [kpc]")
axs[0, 0].legend(frameon=False, loc="upper left")
axs[0, 1].legend(frameon=False)
fig.suptitle("strict set: W1 (with truth-half-split noise floor) and paired dW1\n"
             "(dW1 = bootstrap mean of the paired difference; negative = wider better)")
fig.savefig(FIG / "fig_w1_bins.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 3: conditional log p, FM, contribution decomposition
# measured: logp in [-3.71, -0.92], dlogp [0.02, 1.13], FM [2.89, 5.28],
#           contribution shares <= 0.62
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(16.5, 3.2))
ax = axs[0]
for m in MODELS:
    ax.errorbar(BIN_CENTER, df_bin[f"logp_{m}"], yerr=df_bin[f"logp_{m}_err"],
                color=COLORS[m], marker="o", ms=3, lw=1, label=m, capsize=2)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("cond. log p(v|x) (higher = better)")
ax.set_ylim(-4.0, -0.5)            # manual: bin logp range -3.71..-0.92
ax.legend(frameon=False)
ax = axs[1]
for m in ["w512", "w1024"]:
    ax.errorbar(BIN_CENTER, df_bin[f"dlogp_{m}"], yerr=df_bin[f"dlogp_{m}_err"],
                color=COLORS[m], marker="o", ms=3, lw=1, label=f"{m}-w128", capsize=2)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("paired dlog p vs w128")
ax.set_ylim(0.0, 1.4)              # manual: paired dlogp range 0.02..1.13
ax.legend(frameon=False)
ax = axs[2]
for m in MODELS:
    ax.plot(BIN_CENTER, df_bin[f"fm_{m}"], color=COLORS[m], marker="o", ms=3, lw=1, label=m)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("FM val residual [(100 km/s)$^2$]")
ax.set_ylim(2.7, 5.5)              # manual: per-bin FM range 2.89..5.28
ax.legend(frameon=False)
ax = axs[3]
for m in ["w512", "w1024"]:
    ax.plot(BIN_CENTER, 1 - df_bin[f"fm_{m}"] / df_bin["fm_w128"],
            color=COLORS[m], marker="o", ms=3, lw=1, label=f"{m} vs w128")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("FM relative reduction")
ax.set_ylim(-0.02, 0.16)           # manual: reduction range 0.0..0.13
ax.legend(frameon=False)
ax = axs[4]
contrib = glob["fm_reduction_contribution_bins"]
ax.bar(range(6), contrib, color="#4c72b0")
ax.set_xticks(range(6))
ax.set_xticklabels(BIN_LABELS, rotation=45)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("share of global FM reduction")
ax.set_ylim(0.0, 0.7)              # manual: max share 0.62 (0-10 kpc)
ax.set_title("w128->w1024, strict-mass weights")
fig.suptitle("strict set: per-bin conditional log p, paired dlog p, unregularized FM val loss\n"
             "(FM noise keys split for t and x0 in v2; shared across models)")
fig.savefig(FIG / "fig_nll_fm_bins.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 4: direction heatmaps (sample B, full coverage)
# measured: W1(vT) <= 0.26 in outer cells where floors are ~0.16-0.20
# ------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(9.2, 10.2))
for col, axis in enumerate(["cth", "phi"]):
    sub = df_dir[df_dir.axis == axis]
    cuts = sorted(sub.cut_lo.unique())
    def mat(colname):
        M = np.full((6, len(cuts)), np.nan)
        for _, row in sub.iterrows():
            if pd.notna(row.get(colname, np.nan)):
                M[int(row["bin"]), cuts.index(row["cut_lo"])] = row[colname]
        return M
    M1 = mat("w1_vT_w128")
    M2 = mat("w1_vT_w1024")
    M4 = mat("dlogp_w1024")
    for r, (M, title, cmap, vlim) in enumerate([
            (M1, "W1(vT) w128", "viridis", (0.0, 0.28)),
            (M2, "W1(vT) w1024", "viridis", (0.0, 0.28)),
            (M4, "dlog p (w1024-w128)", "magma", (0.0, 1.3))]):
        ax = axs[r, col]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="w" if M[i, j] > vlim[1] * 0.6 else "k")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="0.5")
        ax.set_xticks(range(len(cuts)))
        ax.set_xticklabels([f"{c:.1f}" for c in cuts])
        ax.set_yticks(range(6))
        ax.set_yticklabels(BIN_LABELS)
        if r == 2:
            ax.set_xlabel("cosθ lower edge" if axis == "cth" else "φ sector lower edge [rad]")
        ax.set_ylabel("r [kpc]")
        ax.set_title(title + f"  ({'cosθ' if axis == 'cth' else 'φ'} split, sample B)")
        fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("strict set, direction cells (full-coverage draws at every strict position)\n"
             "note: outer-row W1 values sit at the truth-half-split noise floor (~0.16-0.20)")
fig.savefig(FIG / "fig_direction_heatmap.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 5: outer bins on clean-val (FIXED v2 runs)
# measured: W1 <= 0.070, std-bias in [-0.036, +0.012], peaks <= 1.10
# ------------------------------------------------------------------
oc = np.load(OUT / "outer_cleanval_samples.npz")
pos_oc = oc["pos"]
w_oc = oc["w_out"]
ir_oc = oc["ir_out"]
r_cval = prot["r_cval"]
eta_oc = prot["eta_cval"][r_cval >= 4.5]
v_oc_true = sph_vel(pos_oc, eta_oc[:, 3:])
gen_oc = {m: sph_vel(np.repeat(pos_oc, K, axis=0), oc[f"vel_{m}"].reshape(-1, 3)
                     ).reshape(len(pos_oc), K, 3) for m in MODELS}
df_oc = pd.read_csv(OUT / "outer_cleanval_metrics.csv")
fig, axs = plt.subplots(2, 4, figsize=(13, 6))
for i, b in enumerate([4, 5]):
    mt = ir_oc == (b - 4)
    for c, cname in enumerate(COMP):
        ax = axs[i, c]
        h, _ = np.histogram(v_oc_true[mt, c], bins=VBINS, weights=w_oc[mt], density=True)
        ax.plot(VC, h, color="k", lw=1.4, label="truth")
        for m in MODELS:
            hg, _ = np.histogram(gen_oc[m][mt][:, :, c].reshape(-1), bins=VBINS,
                                 weights=np.repeat(w_oc[mt], K), density=True)
            ax.plot(VC, hg, color=COLORS[m], lw=0.9, alpha=0.85, label=m)
        ax.set_title(f"r {BIN_LABELS[b]} kpc, ${cname}$ (n={int(mt.sum())})")
        ax.set_xlim(-4, 4)
        ax.set_ylim(0.0, 1.25)      # manual: clean-val outer true peaks <= 1.10 (vT bin4)
        if c == 0:
            ax.set_ylabel("p(v)")
    ax = axs[i, 3]
    x = np.arange(3)
    row = df_oc[df_oc.bin == (4 + i)].iloc[0]
    for m in MODELS:
        ax.bar(x + 0.2 * MODELS.index(m) - 0.2,
               [row[f"bias_std_{cn}_{m}"] for cn in COMP], width=0.18,
               color=COLORS[m], label=m)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(COMP)
    ax.set_title(f"r {BIN_LABELS[4 + i]} kpc: sigma(gen)-sigma(truth)")
    ax.set_ylim(-0.06, 0.03)       # manual: fixed-run std-bias range -0.036..+0.012
axs[0, 3].legend(frameon=False)
fig.suptitle("clean-val outer bins, v2 FIXED conditioning (v1 figure used wrong conditions)\n"
             "leak-free for w512/w1024; w128 has 41% training leakage here")
fig.savefig(FIG / "fig_outer_cleanval.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 6: outer 2D velocity maps: strict bin5 (sample B) and clean-val bin5 (fixed)
# measured: cell densities peak ~0.3
# ------------------------------------------------------------------
def plot_2d_cell(ax, vxyz, title, weights=None):
    h, _, _ = np.histogram2d(vxyz[:, 0], vxyz[:, 2], bins=(np.arange(-4, 4.01, 0.15),
                                                           np.arange(-4, 4.01, 0.15)),
                             weights=weights)
    h = h / (h.sum() * 0.15 * 0.15)
    im = ax.imshow(h.T, origin="lower", extent=[-4, 4, -4, 4], cmap="magma",
                   vmin=0, vmax=0.35)   # manual: cell density peaks <= 0.32
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("$v_r$")
    ax.set_ylabel("$v_\\phi$")
    return im

fig, axs = plt.subplots(4, 4, figsize=(12, 11))
mt = ir == 5
plot_2d_cell(axs[0, 0], v_true[mt], f"truth: strict 60-75 kpc n={int(mt.sum())}",
             weights=w_strict[mt])
for j, m in enumerate(MODELS):
    plot_2d_cell(axs[j + 1, 0], gen_all[m][mt], f"{m} (1 draw per position)",
                 weights=w_strict[mt])
mc = ir_oc == 1
plot_2d_cell(axs[0, 2], v_oc_true[mc], f"truth: clean-val 60-75 kpc n={int(mc.sum())}",
             weights=w_oc[mc])
for j, m in enumerate(MODELS):
    plot_2d_cell(axs[j + 1, 2], gen_oc[m][mc].reshape(-1, 3), f"{m} (K=4, fixed cond)",
                 weights=np.repeat(w_oc[mc], K))
for ax in axs[:, 1].flatten():
    ax.axis("off")
for ax in axs[:, 3].flatten():
    ax.axis("off")
fig.suptitle("outer 2D velocity maps (vr-vphi): left = strict set (sample B), right = clean-val (v2 fixed)")
fig.savefig(FIG / "fig_anomalous_2d.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 7: spatial flow vs EACH MODEL'S OWN input population (v2)
# measured: occupancy 0.0029..0.73; reldev within +-3.3%; cth peaks <= 1.77;
#           phi densities <= 0.20
# ------------------------------------------------------------------
df_spat = pd.read_csv(OUT / "spatial_population_occupancy.csv")
margins = json.load(open(OUT / "spatial_margins.json"))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axs[0]
x = np.arange(6)
occ = df_spat[df_spat["bin"] != "overflow_r>7.5"]
for i, m in enumerate(MODELS):
    sub = occ[occ.model == m].sort_values("bin")
    f_ref = sub["f_population"].to_numpy()
    f_mod = sub["f_model"].to_numpy()
    if i == 0:
        ax.bar(x - 0.27, f_ref, width=0.16, color="k", alpha=0.7, label="own input population")
    ax.bar(x + 0.11 * (i - 1), f_mod, width=0.16, color=COLORS[m], alpha=0.85, label=m)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(BIN_LABELS)
ax.set_xlabel("r [kpc]")
ax.set_ylabel("mass fraction")
ax.set_ylim(1e-3, 1.5)             # manual: occupancy spans 2.9e-3 (60-75) to 0.73 (0-10)
ax.legend(frameon=False, fontsize=6)
ax2 = ax.twinx()
for i, m in enumerate(MODELS):
    sub = occ[occ.model == m].sort_values("bin")
    ax2.plot(x, sub["reldev"].to_numpy() * 100, color=COLORS[m], marker="s", ms=3, lw=0)
ax2.set_ylim(-6, 6)                # manual: reldev range -3.3%..+0.6%
ax2.set_ylabel("model/population - 1  [%] (squares)")
ax = axs[1]
for i, m in enumerate(MODELS):
    d = margins[m]
    c = np.array(d["cth_centers"])
    ax.plot(c, d["cth_ref"], color="k", lw=1.4)
    ax.plot(c, d["cth_model"], color=COLORS[m], lw=0.9)
ax.set_xlabel("cosθ")
ax.set_ylabel("p(cosθ)")
ax.set_ylim(0, 2.1)                # manual: cth peaks 1.72-1.77 (v1 cut this at 1.6)
ax = axs[2]
for i, m in enumerate(MODELS):
    d = margins[m]
    c = np.array(d["phi_centers"])
    ax.plot(c, d["phi_ref"], color="k", lw=1.4)
    ax.plot(c, d["phi_model"], color=COLORS[m], lw=0.9)
ax.set_xlabel("φ [rad]")
ax.set_ylabel("p(φ)")
ax.set_ylim(0, 0.30)               # manual: phi densities <= 0.20
fig.suptitle("spatial flow n(x) vs each model's OWN input population (v2; v1 wrongly used the "
             "radially biased strict intersection)\nblack = population; overflow r>7.5 samples: "
             "w128 67, w512 48, w1024 47 of 262144")
fig.savefig(FIG / "fig_spatial.png")
plt.close(fig)

# ------------------------------------------------------------------
# fig 8: correlation structure per bin
# measured: all correlations within [-0.20, +0.15], max |model-truth| = 0.048
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
for k, pair in enumerate(["vrvT", "vrvth", "vthvT"]):
    ax = axs[k]
    ax.plot(BIN_CENTER, df_bin[f"corr_{pair}_true"], "k", marker="o", ms=4, label="truth")
    for m in MODELS:
        ax.plot(BIN_CENTER, df_bin[f"corr_{pair}_{m}"], color=COLORS[m], marker="o",
                ms=3, lw=1, label=m)
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_ylim(-0.25, 0.25)       # manual: correlations span -0.19..+0.15
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(f"corr(${pair[:2]}$,${pair[2:]}$)")
axs[0].legend(frameon=False)
fig.suptitle("strict set: velocity component correlations vs radius (max |model-truth| = 0.048)")
fig.savefig(FIG / "fig_corr_bins.png")
plt.close(fig)

# ------------------------------------------------------------------
# summary tables
# ------------------------------------------------------------------
with open(OUT / "summary_tables.md", "w") as f:
    f.write("# Phase-1 summary tables (v2; strict set unless noted)\n\n")
    f.write("## per radial bin: conditional log p (paired dlog p) and FM residual\n\n")
    cols = ["bin", "n_true", "n_eff", "n_gen_positions",
            "logp_w128", "logp_w512", "logp_w1024", "dlogp_w512", "dlogp_w1024",
            "fm_w128", "fm_w512", "fm_w1024"]
    f.write(df_bin[cols].round(3).to_markdown(index=False))
    f.write("\n\n## per radial bin: W1 per component + truth-vs-truth floor\n\n")
    fl = df_floor.set_index("bin")
    parts = ["bin"]
    for c in COMP:
        parts += [f"w1_{c}_w128", f"w1_{c}_w512", f"w1_{c}_w1024"]
    tab = df_bin[parts].copy()
    for c in COMP:
        tab[f"floor_{c}"] = fl[f"floor_{c}"].to_numpy()
    f.write(tab.round(4).to_markdown(index=False))
    f.write("\n\n## per radial bin: std bias (gen - truth)\n\n")
    cols = ["bin"] + [f"bias_std_{c}_{m}" for c in COMP for m in MODELS]
    f.write(df_bin[cols].round(4).to_markdown(index=False))
    f.write("\n\n## clean-val outer bins, v2 FIXED conditioning (w512/w1024 leak-free)\n\n")
    sel = ["bin", "n_true"] + [f"bias_std_{c}_{m}" for c in COMP for m in MODELS] \
        + [f"w1_{c}_{m}" for c in COMP for m in MODELS]
    f.write(df_oc[sel].round(3).to_markdown(index=False))
    f.write("\n\n## spatial occupancy vs own population (reldev = model/population - 1)\n\n")
    f.write(df_spat.round(4).to_markdown(index=False))
    f.write("\n\n## global\n\n```json\n")
    f.write(json.dumps(glob, indent=1))
    f.write("\n```\n")
print("figures + summary tables v2 saved")
