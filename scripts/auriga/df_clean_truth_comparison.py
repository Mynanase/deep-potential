#!/usr/bin/env python
"""w1024clean vs the CLEAN population as truth (its own zero-leakage val set).

Three figures (same names as the full-population versions):
  fig_clump_conditional_vT.png   truth = clean-val residual population within
                                 8 kpc of any clump star (the debris + local
                                 field that survives in halo12-clean.h5);
                                 models = conditional draws at the 1,977
                                 clump-star positions. Gray step = clean-val
                                 cell field farther than 8 kpc.
  fig_radial_marginals_h12val_outer.png
                                 outer 45-60 / 60-75 kpc marginals, truth =
                                 clean-val (no clump), draws at the SAME
                                 positions (K=4).
  fig_radial_marginals_strict.png
                                 6 radial bins over ALL clean-val (n=410821),
                                 paired single draws at every position.

w1024clean is emphasized; w1024full / w128 are context lines (both trained on
the full population; ~41% of clean-val rows were in their training data).

Noise keys (documented): 20260918 -> clean-val single draws; 918 -> outer
K=4 draws; 919 -> clump-position K=4 draws. Guards as in the audit suite.

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_clean_truth_comparison.py
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
OUT = REPO / "runs" / "clean-truth-comparison-20260917"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = [("w1024clean", REPO / "runs" / "halo12-cap-w1024"),
          ("w1024full", REPO / "runs" / "w1024full"),
          ("w128", REPO / "runs" / "halo12-baseline")]
EMPH = "w1024clean"
NAMES = [m[0] for m in MODELS]
SHORT = {"w1024full": "full", "w128": "128", "w1024clean": "clean"}
K = 4
CHUNK = 8192
COMP = ["vr", "vth", "vT"]


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def w1(a, b, wa, wb):
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


prot = np.load(AUD / "eval_protocol.npz")
removed = prot["removed_rows"]
R_EDGES = prot["r_edges"]
n_rbins = len(R_EDGES) - 1

# ---- clean population + its val split (w1024clean's monitoring val) ----
with h5py.File(REPO / "data" / "auriga" / "halo12-clean.h5", "r") as f:
    eta_c = f["eta"][:]
    w_c = f["weights"][:]
n_cv = int(len(eta_c) * 0.25)
eta_cv = eta_c[:n_cv]
w_cv = w_c[:n_cv]
r_cv = np.linalg.norm(eta_cv[:, :3], axis=1)
ir_cv = np.clip(np.digitize(r_cv, R_EDGES) - 1, 0, n_rbins - 1)
v_true_cv = sph_vel(eta_cv[:, :3], eta_cv[:, 3:])
outer_c = r_cv >= 4.5
ir_out_c = (r_cv[outer_c] >= 6.0).astype(int)
print("clean-val stars: %d (outer=%d, bin4=%d, bin5=%d)"
      % (n_cv, int(outer_c.sum()), int((ir_out_c == 0).sum()),
         int((ir_out_c == 1).sum())), flush=True)

# ---- clump region (geometric): positions of the 9683 removed stars ----
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_f = f["eta"][:]
clump_pos_all = eta_f[removed][:, :3]
nv_f = int(len(eta_f) * 0.25)
eta_fv = eta_f[:nv_f]
cl_v = np.zeros(nv_f, dtype=bool)
cl_v[removed[removed < nv_f]] = True
r_fv = np.linalg.norm(eta_fv[:, :3], axis=1)
m_cl = cl_v & (r_fv >= 4.5) & (r_fv < 6.0)
pos_cl = eta_fv[m_cl, :3]
print("clump-star conditioning positions (bin4 val): %d" % len(pos_cl), flush=True)

# clean-val residual membership around the clump (8 kpc = 0.8 code units)
tree = cKDTree(clump_pos_all)
d_cv, _ = tree.query(eta_cv[:, :3], k=1)
near = outer_c & (d_cv < 0.8) & (ir_out_c == 0)
x, y, z = eta_cv[:, 0], eta_cv[:, 1], eta_cv[:, 2]
r_c_ = np.linalg.norm(eta_cv[:, :3], axis=1)
cth_c = z / r_c_
phi_c = np.mod(np.arctan2(y, x), 2 * np.pi) - np.pi
far_cell = outer_c & (ir_out_c == 0) & (cth_c < -0.6) \
    & (phi_c >= np.pi / 2) & (phi_c < np.pi) & (d_cv >= 0.8)
print("clean-val residual near clump (d<8kpc, bin4): %d ; cell field (d>=8kpc): %d"
      % (int(near.sum()), int(far_cell.sum())), flush=True)

key1 = jax.random.key(20260918)
z0_cv = jax.random.normal(key1, (n_cv, 3))
key2 = jax.random.key(918)
z0_out = jax.random.normal(key2, (int(outer_c.sum()), K, 3))
key3 = jax.random.key(919)
z0_cl = jax.random.normal(key3, (len(pos_cl), K, 3))


@eqx.filter_jit
def sample_one_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(z0, cond)


@eqx.filter_jit
def sample_k_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


cv_draws = {}
out_draws = {}
cl_draws = {}
for nm, run_dir in MODELS:
    print("===== %s: %s =====" % (nm, run_dir), flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(eta_cv[:256, :3])
    explicit = np.asarray(sample_k_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2, "sample path mismatch for " + nm

    v1 = np.empty((n_cv, 3), dtype=np.float32)
    for i in range(0, n_cv, CHUNK):
        sl = slice(i, min(i + CHUNK, n_cv))
        v1[sl] = np.asarray(sample_one_stage(cvf, z0_cv[sl], jnp.asarray(eta_cv[sl, :3])))
    cv_draws[nm] = sph_vel(eta_cv[:, :3], v1)

    vs = np.asarray(sample_k_stage(cvf, z0_out, jnp.asarray(eta_cv[outer_c, :3])))
    out_draws[nm] = sph_vel(np.repeat(eta_cv[outer_c, :3], K, axis=0),
                            vs.reshape(-1, 3)).reshape(int(outer_c.sum()), K, 3)

    vs2 = np.asarray(sample_k_stage(cvf, z0_cl, jnp.asarray(pos_cl)))
    cl_draws[nm] = sph_vel(np.repeat(pos_cl, K, axis=0), vs2.reshape(-1, 3)).reshape(len(pos_cl), K, 3)
    del flow, cvf
    jax.clear_caches()

rows = []


def panel(ax, truth, w_t, draws, w_d, w1s, title, emph=EMPH):
    bins = np.linspace(-4.5, 4.5, 73)
    ax.hist(truth, bins=bins, weights=w_t, density=True, alpha=0.45,
            color="C0", label="truth (clean-val)")
    styles = {"w1024full": ("C1", "-", 1.3), "w128": ("C2", "--", 1.3),
              "w1024clean": ("C3", ":", 2.2)}
    for nm in NAMES:
        c, ls, lw = styles[nm]
        ax.hist(draws[nm], bins=bins, weights=w_d[nm], density=True,
                histtype="step", lw=lw, color=c, ls=ls, label=nm)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-4.5, 4.5)
    ax.tick_params(labelsize=7)
    txt = "W1 " + " ".join("%s %.3f" % (SHORT[nm], w1s[nm]) for nm in NAMES)
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.2, va="top")


# ---- figure 1: clump-region conditional vT, clean residual truth ----
fig, ax = plt.subplots(figsize=(8, 4.6))
bins_h = np.linspace(-4.5, 4.5, 73)
ax.hist(v_true_cv[near, 2], bins=bins_h, weights=w_cv[near], density=True,
        alpha=0.5, color="C0", label="clean-val residual d<8 kpc (n=%d)" % near.sum())
ax.hist(v_true_cv[far_cell, 2], bins=bins_h, weights=w_cv[far_cell], density=True,
        histtype="step", lw=1.4, color="0.45",
        label="clean-val cell field d>=8 kpc (n=%d)" % far_cell.sum())
styles = {"w1024full": ("C1", "-", 1.2), "w128": ("C2", "--", 1.2),
          "w1024clean": ("C3", ":", 2.2)}
for nm in NAMES:
    c, ls, lw = styles[nm]
    g = cl_draws[nm][:, :, 2].reshape(-1)
    ax.hist(g, bins=bins_h, weights=np.ones_like(g), density=True,
            histtype="step", lw=lw, color=c, ls=ls,
            label=nm + " draws @ clump pos")
ax.set_xlabel("vT (100 km/s)")
ax.set_ylabel("density")
ax.set_title("clump region (45-60 kpc south-polar): clean residual truth vs conditional draws", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_clump_conditional_vT.png", dpi=150)
plt.close(fig)

# ---- figure 2: outer bins, clean-val truth, draws at same positions ----
fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for ci, cn in enumerate(COMP):
    for b, (lo, hi) in enumerate([(4.5, 6.0), (6.0, 7.5)]):
        m = ir_out_c == b
        w_b = w_cv[outer_c][m]
        truth = v_true_cv[outer_c][m, ci]
        draws = {nm: out_draws[nm][m][:, :, ci].reshape(-1) for nm in NAMES}
        w_d = {nm: np.repeat(w_b, K) for nm in NAMES}
        w1s = {nm: w1(truth, draws[nm], w_b, w_d[nm]) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="cleanval_outer", bin=4 + b, r_lo=lo * 10, r_hi=hi * 10,
                             comp=cn, model=nm, w1=w1s[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, w_d, w1s,
              "clean-val %g-%g kpc (n=%d)" % (lo * 10, hi * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("outer bins, CLEAN population truth (no clump); w1024full/w128 context (41% leakage)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_h12val_outer.png", dpi=150)
plt.close(fig)

# ---- figure 3: all radial bins over clean-val ----
fig, axes = plt.subplots(3, n_rbins, figsize=(19, 7.8), sharex=True)
for ci, cn in enumerate(COMP):
    for b in range(n_rbins):
        m = ir_cv == b
        w_b = w_cv[m]
        truth = v_true_cv[m, ci]
        draws = {nm: cv_draws[nm][m, ci] for nm in NAMES}
        w1s = {nm: w1(truth, draws[nm], w_b, w_b) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="cleanval_bins", bin=b, r_lo=R_EDGES[b] * 10,
                             r_hi=R_EDGES[b + 1] * 10, comp=cn, model=nm,
                             w1=w1s[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, {nm: w_b for nm in NAMES}, w1s,
              "%g-%g kpc (n=%d)" % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, -1].legend(fontsize=7, loc="upper right")
fig.suptitle("CLEAN-val population: radial-bin velocity marginals (truth vs three DFs)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_strict.png", dpi=150)
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / "clean_truth_w1.csv", index=False)
print("=== W1(vT) per radial bin (clean-val) ===", flush=True)
piv = df[(df["set"] == "cleanval_bins") & (df["comp"] == "vT")].pivot(index="bin", columns="model", values="w1")
print(piv.round(4).to_string(), flush=True)
print("=== W1(vT) outer (clean-val) ===", flush=True)
piv2 = df[(df["set"] == "cleanval_outer") & (df["comp"] == "vT")].pivot(index="bin", columns="model", values="w1")
print(piv2.round(4).to_string(), flush=True)
print("outputs under", OUT, flush=True)
with open(OUT / "summary.json", "w") as f:
    json.dump({"w1": df.to_dict(orient="records"),
               "n_cleanval": int(n_cv), "n_near": int(near.sum()),
               "n_far_cell": int(far_cell.sum())}, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
