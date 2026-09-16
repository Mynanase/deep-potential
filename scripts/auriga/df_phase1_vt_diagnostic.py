#!/usr/bin/env python
"""Phase-1 follow-up: where does the 45-75 kpc vT-direction W1 disadvantage
come from -- model capacity or training-data population?

Three cuts on top of the phase-1 audit artifacts
(runs/halo12-phase1-df-audit-20260916, audit run 5c4bf67d):

  A. strict common set (fair to w128/w512/w1024): per-radial-bin W1(vT) with
     paired bootstrap differences, against the audit truth-vs-truth noise
     floor; plus a counterfactual per-bin decomposition of the GLOBAL
     strict W1(vT) gap (which bin generates the global ordering).
  B. moment decomposition of the vT gap in bins 0, 4, 5: location (mean) /
     scale (std) / residual-shape shares of W1(vT), using the audit's
     full-coverage paired single draws (vel_samples_all).
  C. training-population cut: fresh conditional draws (K=4, shared noise,
     same guard as the audit) at ALL halo12-val outer positions; W1(vT)
     with vs without the 9,683 clump stars (eval_protocol.removed_rows,
     radius quantiles ~54-63 kpc) that exist in halo12.h5 but were removed
     from halo12-clean.h5, the training population of w512/w1024.

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=2 python scripts/auriga/df_phase1_vt_diagnostic.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
OUT = REPO / "runs" / "halo12-vt-diagnostic-20260917"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = ["w128", "w512", "w1024"]
RUN_DIR = {"w128": "runs/halo12-baseline",
           "w512": "runs/halo12-cap-w512",
           "w1024": "runs/halo12-cap-w1024"}
B_BOOT = 100
RNG = np.random.default_rng(20260917)
K = 4


def sph_vel(pos, vel_cart):
    """Same convention as the audit suite (origin 0, signed vT = v_phi)."""
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


def w1(a, b, wa, wb):
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


# =====================================================================
# shared inputs: strict protocol + audit per-model full-coverage draws
# =====================================================================
prot = np.load(AUD / "eval_protocol.npz")
eta_strict = prot["eta_strict"]
w_strict = prot["w_strict"]
ir = prot["ir_strict"]
vT_true = prot["vT_strict"]
vth_true = prot["vth_strict"]
R_EDGES = prot["r_edges"]
n_rbins = len(R_EDGES) - 1

gen_all = {}
for m in MODELS:
    d = np.load(AUD / ("model_" + m + ".npz"))
    gen_all[m] = sph_vel(eta_strict[:, :3], d["vel_samples_all"])  # paired draws

df_floor = pd.read_csv(AUD / "w1_noise_floor.csv")
floor_rad = df_floor[df_floor["kind"] == "radial"].set_index("bin")
df_sampa = pd.read_csv(AUD / "per_bin_metrics.csv").set_index("bin")

# =====================================================================
# A. strict per-bin W1(vT) + paired bootstrap + global decomposition
# =====================================================================
print("=== A. STRICT PER-BIN W1(vT) (sample-B paired draws, audit protocol) ===", flush=True)
rows = []
boot_store = {}
for b in range(n_rbins):
    m_bin = ir == b
    t = vT_true[m_bin]
    w = w_strict[m_bin]
    row = dict(bin=b, r_lo=R_EDGES[b] * 10, r_hi=R_EDGES[b + 1] * 10,
               n=int(m_bin.sum()), mass_frac=float(w.sum() / w_strict.sum()),
               floor_vT=float(floor_rad.loc[b, "floor_vT"]),
               floor_vT_sd=float(floor_rad.loc[b, "floor_vT_sd"]))
    for m in MODELS:
        row["w1_vT_" + m] = w1(t, gen_all[m][m_bin, 2], w, w)
        row["w1_vT_" + m + "_sampleA"] = float(df_sampa.loc[b, "w1_vT_" + m])
    idx = np.where(m_bin)[0]
    n_b = len(idx)
    diffs = {"dW1_w1024_w128": [], "dW1_w1024_w512": []}
    for _ in range(B_BOOT):
        s = idx[RNG.integers(0, n_b, n_b)]
        ws = w_strict[s]
        ts = vT_true[s]
        w1s = {m: w1(ts, gen_all[m][s, 2], ws, ws) for m in MODELS}
        diffs["dW1_w1024_w128"].append(w1s["w1024"] - w1s["w128"])
        diffs["dW1_w1024_w512"].append(w1s["w1024"] - w1s["w512"])
    boot_store[b] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in diffs.items()}
    for k, (mu, sd) in boot_store[b].items():
        row[k] = mu
        row[k + "_err"] = sd
    rows.append(row)
df_bins = pd.DataFrame(rows)
df_bins.to_csv(OUT / "vt_strict_bins.csv", index=False)
cols = ["bin", "r_lo", "n", "mass_frac", "floor_vT",
        "w1_vT_w128", "w1_vT_w512", "w1_vT_w1024",
        "dW1_w1024_w128", "dW1_w1024_w128_err", "dW1_w1024_w512", "dW1_w1024_w512_err"]
print(df_bins[[c for c in cols if c in df_bins.columns]].round(4).to_string(index=False), flush=True)

# global strict W1(vT) and counterfactual per-bin contributions
glob = {m: w1(vT_true, gen_all[m][:, 2], w_strict, w_strict) for m in MODELS}
contrib_rows = []
for hi, lo in [("w1024", "w128"), ("w1024", "w512")]:
    base = gen_all[hi][:, 2]
    d_glob = glob[hi] - glob[lo]
    row = dict(pair=hi + "-" + lo, dW1_global=d_glob)
    for b in range(n_rbins):
        hyb = base.copy()
        m_bin = ir == b
        hyb[m_bin] = gen_all[lo][m_bin, 2]
        row["contrib_bin" + str(b)] = glob[hi] - w1(vT_true, hyb, w_strict, w_strict)
    contrib_rows.append(row)
df_glob = pd.DataFrame(contrib_rows)
df_glob.to_csv(OUT / "vt_global_decomp.csv", index=False)
print("global strict W1(vT):", {m: round(glob[m], 4) for m in MODELS}, flush=True)
print(df_glob.round(4).to_string(index=False), flush=True)

# =====================================================================
# B. moment decomposition (location / scale / shape) of W1(vT)
# =====================================================================
print("=== B. MOMENT DECOMPOSITION OF W1(vT) (strict, paired draws) ===", flush=True)
rows = []
for b in (0, 4, 5):
    m_bin = ir == b
    t = vT_true[m_bin]
    w = w_strict[m_bin]
    mu_t, sd_t = wmean(t, w), wstd(t, w)
    for m in MODELS:
        y = gen_all[m][m_bin, 2]
        mu_m, sd_m = wmean(y, w), wstd(y, w)
        w1_tot = w1(t, y, w, w)
        y_meanmatch = y - (mu_m - mu_t)
        w1_mean = w1(t, y_meanmatch, w, w)
        y_scalematch = mu_t + (y - mu_m) * (sd_t / sd_m)
        w1_scale = w1(t, y_scalematch, w, w)
        rows.append(dict(bin=b, r_lo=R_EDGES[b] * 10, model=m,
                         true_mean=mu_t, true_std=sd_t,
                         bias_mean=mu_m - mu_t, bias_std=sd_m - sd_t,
                         w1_total=w1_tot, w1_meanmatched=w1_mean,
                         w1_scalematched=w1_scale,
                         loc_share=1 - w1_mean / w1_tot,
                         scale_share=(w1_mean - w1_scale) / w1_tot,
                         shape_share=w1_scale / w1_tot))
df_mom = pd.DataFrame(rows)
df_mom.to_csv(OUT / "vt_moments_bins.csv", index=False)
print(df_mom.round(4).to_string(index=False), flush=True)

# =====================================================================
# C. halo12-val outer: clump vs field decomposition (fresh draws)
# =====================================================================
print("=== C. HALO12-VAL OUTER (45-75 kpc): CLUMP VS FIELD ===", flush=True)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_full = f["eta"][:]
    w_full = f["weights"][:]
n_val = int(len(eta_full) * 0.25)
eta_val = eta_full[:n_val]
w_val = w_full[:n_val]
clump_val = np.zeros(n_val, dtype=bool)
clump_val[prot["removed_rows"][prot["removed_rows"] < n_val]] = True
r_val = np.linalg.norm(eta_val[:, :3], axis=1)
outer = r_val >= 4.5
pos = eta_val[outer, :3]
w_out = w_val[outer]
clump_out = clump_val[outer]
ir_out = (r_val[outer] >= 6.0).astype(int)
v_true_sph = sph_vel(pos, eta_val[outer, 3:])
print("outer halo12-val stars: %d (bin4=%d, bin5=%d, clump=%d)"
      % (outer.sum(), int((ir_out == 0).sum()), int((ir_out == 1).sum()), int(clump_out.sum())), flush=True)

key = jax.random.key(917)
z0 = jax.random.normal(key, (len(pos), K, 3))


@eqx.filter_jit
def sample_stage(cvf, z0_, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0_.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


draws = {}
for m in MODELS:
    flow = fit_all.load_flow(REPO / RUN_DIR[m] / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    explicit = np.asarray(sample_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), jnp.asarray(pos[:256])))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=jnp.asarray(pos[:256])))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (m, dmax), flush=True)
    assert dmax < 2e-2, "sample_stage mismatch for " + m
    vs = np.asarray(sample_stage(cvf, z0, jnp.asarray(pos)))
    draws[m] = sph_vel(np.repeat(pos, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos), K, 3)
    del flow, cvf
    jax.clear_caches()

rows = []
for b, lo_kpc, hi_kpc in [(0, 45.0, 60.0), (1, 60.0, 75.0)]:
    m_bin = ir_out == b
    m_f = m_bin & ~clump_out
    m_c = m_bin & clump_out
    row = dict(bin=4 + b, r_lo=lo_kpc, r_hi=hi_kpc,
               n_field=int(m_f.sum()), n_clump=int(m_c.sum()),
               mass_clump_frac=float(w_out[m_c].sum() / w_out[m_bin].sum()),
               field_vT_mean=wmean(v_true_sph[m_f, 2], w_out[m_f]),
               field_vT_std=wstd(v_true_sph[m_f, 2], w_out[m_f]),
               clump_vT_mean=wmean(v_true_sph[m_c, 2], w_out[m_c]) if m_c.any() else float("nan"),
               clump_vT_std=wstd(v_true_sph[m_c, 2], w_out[m_c]) if m_c.any() else float("nan"),
               clump_vth_std=wstd(v_true_sph[m_c, 1], w_out[m_c]) if m_c.any() else float("nan"),
               field_vth_std=wstd(v_true_sph[m_f, 1], w_out[m_f]))
    for m in MODELS:
        for cname, ci in [("vT", 2), ("vth", 1)]:
            t_all = v_true_sph[m_bin, ci]
            g_all = draws[m][m_bin][:, :, ci].reshape(-1)
            w_all = np.repeat(w_out[m_bin], K)
            row["w1_" + cname + "_" + m + "_all"] = w1(t_all, g_all, w_out[m_bin], w_all)
            t_f = v_true_sph[m_f, ci]
            g_f = draws[m][m_f][:, :, ci].reshape(-1)
            w_f = np.repeat(w_out[m_f], K)
            row["w1_" + cname + "_" + m + "_exclclump"] = w1(t_f, g_f, w_out[m_f], w_f)
        gT_f = draws[m][m_f][:, :, 2].reshape(-1)
        row["bias_mean_vT_" + m + "_field"] = wmean(gT_f, np.repeat(w_out[m_f], K)) - row["field_vT_mean"]
        if m_c.any():
            gT_c = draws[m][m_c][:, :, 2].reshape(-1)
            row["bias_mean_vT_" + m + "_clump"] = wmean(gT_c, np.repeat(w_out[m_c], K)) - row["clump_vT_mean"]
    rows.append(row)
df_clump = pd.DataFrame(rows)
df_clump.to_csv(OUT / "vt_h12val_clump.csv", index=False)
sel = ["bin", "n_field", "n_clump", "mass_clump_frac",
       "field_vT_mean", "clump_vT_mean", "field_vT_std", "clump_vT_std"]
for m in MODELS:
    sel += ["w1_vT_" + m + "_all", "w1_vT_" + m + "_exclclump"]
print(df_clump[sel].round(4).to_string(index=False), flush=True)

# clump localization: (radial bin, cos-theta, phi-sector) cells
CTH_EDGES = prot["cth_edges"]
PHI_EDGES = prot["phi_edges"]
x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
r_o = np.linalg.norm(pos, axis=1)
cth_o = z / r_o
phi_o = np.mod(np.arctan2(y, x), 2 * np.pi) - np.pi
ic_o = np.clip(np.digitize(cth_o, CTH_EDGES) - 1, 0, len(CTH_EDGES) - 2)
ip_o = np.clip(np.digitize(phi_o, PHI_EDGES) - 1, 0, len(PHI_EDGES) - 2)
cells = pd.DataFrame(dict(bin=4 + ir_out, cth=ic_o, phi=ip_o, clump=clump_out))
cell_counts = cells[cells["clump"]].groupby(["bin", "cth", "phi"]).size().sort_values(ascending=False)
print("top clump direction cells (bin, cth idx, phi idx): count / total clump", flush=True)
tot_clump = int(clump_out.sum())
for (b_, c_, p_), n_ in cell_counts.head(4).items():
    print("  bin=%d cth=%d phi=%d: %d (%.1f%%)" % (b_, c_, p_, n_, 100.0 * n_ / tot_clump), flush=True)

# direction cells on the strict side (audit CSV), bins 4-5, largest paired
# vT differences -- do localized cells drive the strict outer vT ordering?
df_dir = pd.read_csv(AUD / "direction_metrics.csv")
dir_sel = []
for axis in ["cth", "phi"]:
    dd = df_dir[(df_dir["bin"] >= 4) & (df_dir["axis"] == axis)].copy()
    dd = dd.sort_values("dw1_vT_w1024", ascending=False)
    dir_sel.append(dd.head(3))
df_dirsel = pd.concat(dir_sel)
keep = ["bin", "axis", "cut_lo", "cut_hi", "n_true", "floor_vT",
        "w1_vT_w128", "w1_vT_w512", "w1_vT_w1024",
        "dw1_vT_w1024", "dw1_vT_w1024_err"]
df_dirsel[keep].to_csv(OUT / "vt_direction_cells.csv", index=False)
print("strict direction cells with the largest dw1_vT(w1024-w128), bins 4-5:", flush=True)
print(df_dirsel[keep].round(4).to_string(index=False), flush=True)

# =====================================================================
# figures
# =====================================================================
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

centers = df_bins["r_lo"] + 0.5 * (df_bins["r_hi"] - df_bins["r_lo"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.fill_between(centers, df_bins["floor_vT"] - df_bins["floor_vT_sd"],
                df_bins["floor_vT"] + df_bins["floor_vT_sd"], color="0.85", label="noise floor")
for m, mk in zip(MODELS, "osd"):
    ax.plot(centers, df_bins["w1_vT_" + m], mk + "-", label=m)
ax.set_xlabel("r (kpc, bin center)")
ax.set_ylabel("W1(vT)")
ax.set_title("strict common set: W1(vT) per radial bin")
ax.legend()
ax = axes[1]
for k, mk in [("dW1_w1024_w128", "o-"), ("dW1_w1024_w512", "s-")]:
    ax.errorbar(centers, df_bins[k], yerr=df_bins[k + "_err"], fmt=mk, capsize=3, label=k)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("r (kpc, bin center)")
ax.set_ylabel("paired dW1(vT)")
ax.set_title("paired bootstrap differences (sample B)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_vt_strict_bins.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
width = 0.35
xs = np.arange(2)
for i, m in enumerate(MODELS):
    ax.bar(xs - width / 2 + i * width / 3, df_clump["w1_vT_" + m + "_all"], width / 3, label=m + " all")
    ax.bar(xs + 1.2 + i * width / 3, df_clump["w1_vT_" + m + "_exclclump"], width / 3, label=m + " excl-clump")
ax.set_xticks([0.5, 1.7])
ax.set_xticklabels(["45-60 kpc", "60-75 kpc"])
ax.set_ylabel("W1(vT) on halo12-val")
ax.set_title("clump-star cut: halo12-val outer W1(vT)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_vt_clump_cut.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
m_b4 = ir_out == 0
bins_h = np.linspace(-4, 4, 61)
ax.hist(v_true_sph[m_b4 & ~clump_out, 2], bins=bins_h, weights=w_out[m_b4 & ~clump_out],
        density=True, alpha=0.5, label="field stars (truth)")
ax.hist(v_true_sph[m_b4 & clump_out, 2], bins=bins_h, weights=w_out[m_b4 & clump_out],
        density=True, alpha=0.5, label="clump stars (truth)")
gT = draws["w1024"][m_b4][:, :, 2].reshape(-1)
ax.hist(gT, bins=bins_h, weights=np.repeat(w_out[m_b4], K), density=True,
        histtype="step", lw=1.5, label="w1024 draws")
ax.set_xlabel("vT (100 km/s)")
ax.set_title("halo12-val 45-60 kpc: vT of field vs clump vs w1024")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_vt_clump_hist.png", dpi=150)
plt.close(fig)

# =====================================================================
# summary
# =====================================================================
summary = dict(
    strict_global_w1_vT={m: glob[m] for m in MODELS},
    strict_global_dW1={r["pair"]: r["dW1_global"] for r in contrib_rows},
    strict_bin_contribution={
        r["pair"]: {("bin%d" % b): r["contrib_bin%d" % b] for b in range(n_rbins)}
        for r in contrib_rows},
    strict_bins=df_bins.to_dict(orient="records"),
    moment_shares=df_mom.to_dict(orient="records"),
    h12val_clump=df_clump.to_dict(orient="records"),
    clump_total=int(clump_out.sum()),
    direction_top=df_dirsel[keep].to_dict(orient="records"),
)
with open(OUT / "vt_diagnostic_summary.json", "w") as f:
    json.dump(summary, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

print("=== VT DIAGNOSTIC FINAL SUMMARY ===", flush=True)
print("strict global W1(vT): w128 %.4f  w512 %.4f  w1024 %.4f"
      % (glob["w128"], glob["w512"], glob["w1024"]), flush=True)
for r in contrib_rows:
    top = max(range(n_rbins), key=lambda b: r["contrib_bin%d" % b])
    print("%s: dW1_global=%+.4f, largest contribution bin%d (%+.4f)"
          % (r["pair"], r["dW1_global"], top, r["contrib_bin%d" % top]), flush=True)
for _, r in df_clump.iterrows():
    gap_all = r["w1_vT_w1024_all"] - r["w1_vT_w128_all"]
    gap_ex = r["w1_vT_w1024_exclclump"] - r["w1_vT_w128_exclclump"]
    print("h12val bin%d: dW1(vT, w1024-w128) all=%+.4f excl-clump=%+.4f (clump mass %.1f%%)"
          % (r["bin"], gap_all, gap_ex, 100 * r["mass_clump_frac"]), flush=True)
print("outputs under", OUT, flush=True)
