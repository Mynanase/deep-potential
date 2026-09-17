#!/usr/bin/env python
"""w1024clean vs clean truth, v2: clump sample defined by PID identity.

The clump truth is extracted directly from data/auriga/clump_pid_registry.npz
(detection_pids = the 9,683 removed outer-clump members; none remain in
halo12-clean.h5). The field reference is the clean-val population in the same
angular cell and a radius band matched to the clump extent. Models are
conditioned at the member positions (K=4, key 920).

Figures 2-3 (radial marginals vs clean-val) regenerate the v1 protocol
unchanged (keys 20260918/918) so this node is a self-contained artifact set.

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_clean_truth_pid.py
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
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
REG = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "clean-truth-pid-20260917"
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


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))


def wstd(a, w):
    mu = wmean(a, w)
    return float(np.sqrt(np.sum(w * (a - mu) ** 2) / np.sum(w)))


# ------------------------------------------------------------------
# pid registry + populations
# ------------------------------------------------------------------
reg = np.load(REG)
print("registry schema:", str(reg["schema"]), flush=True)
detection = np.asarray(reg["detection_pids"]).astype(np.uint64)
print("detection_pids: n=%d (unique %d)" % (len(detection), np.unique(detection).size), flush=True)

with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    pid_f = f["particle_id"][:]
    eta_f = f["eta"][:]
    w_f = f["weights"][:]
mem_f = np.isin(pid_f, detection)
assert int(mem_f.sum()) == 9683, "detection members in halo12.h5 != 9683"
pos_cl = eta_f[mem_f, :3]
w_cl = w_f[mem_f]
v_cl = sph_vel(pos_cl, eta_f[mem_f, 3:])
r_cl = np.linalg.norm(pos_cl, axis=1)
print("clump members: n=%d  r[kpc]=%s  vT=%+.3f+-%.3f  frac<-1=%.3f"
      % (mem_f.sum(), np.round(np.quantile(r_cl, [0, .5, 1]) * 10, 1),
         wmean(v_cl[:, 2], w_cl), wstd(v_cl[:, 2], w_cl),
         float(w_cl[v_cl[:, 2] < -1].sum() / w_cl.sum())), flush=True)

with h5py.File(REPO / "data" / "auriga" / "halo12-clean.h5", "r") as f:
    pid_c = f["particle_id"][:]
    eta_c = f["eta"][:]
    w_c = f["weights"][:]
print("detection members remaining in halo12-clean.h5: %d"
      % int(np.isin(pid_c, detection).sum()), flush=True)

n_cv = int(len(eta_c) * 0.25)
eta_cv = eta_c[:n_cv]
w_cv = w_c[:n_cv]
pos_cv = eta_cv[:, :3]
r_cv = np.linalg.norm(pos_cv, axis=1)
x, y, z = pos_cv[:, 0], pos_cv[:, 1], pos_cv[:, 2]
cth_cv = z / r_cv
phi_cv = np.mod(np.arctan2(y, x), 2 * np.pi) - np.pi
cell = (cth_cv < -0.6) & (phi_cv >= np.pi / 2) & (phi_cv < np.pi)
field = cell & (r_cv >= 5.4) & (r_cv < 6.4) & ~np.isin(pid_c[:n_cv], detection)
v_true_cv = sph_vel(pos_cv, eta_cv[:, 3:])
print("clean-val field reference (cell, 54-64 kpc): n=%d  vT=%+.3f+-%.3f"
      % (field.sum(), wmean(v_true_cv[field, 2], w_cv[field]),
         wstd(v_true_cv[field, 2], w_cv[field])), flush=True)

key_cl = jax.random.key(920)
z0_cl = jax.random.normal(key_cl, (len(pos_cl), K, 3))


@eqx.filter_jit
def sample_one_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(z0, cond)


@eqx.filter_jit
def sample_k_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


cl_draws = {}
for nm, run_dir in MODELS:
    print("===== %s: %s =====" % (nm, run_dir), flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(pos_cl[:256])
    explicit = np.asarray(sample_k_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2, "sample path mismatch for " + nm
    vs = np.asarray(sample_k_stage(cvf, z0_cl, jnp.asarray(pos_cl)))
    cl_draws[nm] = sph_vel(np.repeat(pos_cl, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos_cl), K, 3)
    del flow, cvf
    jax.clear_caches()

# ------------------------------------------------------------------
# figure 1: clump truth by pid vs conditional draws at member positions
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6))
bins_h = np.linspace(-4.5, 4.5, 73)
ax.hist(v_cl[:, 2], bins=bins_h, weights=w_cl, density=True, alpha=0.5,
        color="C0", label="clump truth (detection_pids, n=%d)" % mem_f.sum())
ax.hist(v_true_cv[field, 2], bins=bins_h, weights=w_cv[field], density=True,
        histtype="step", lw=1.4, color="0.45",
        label="clean-val field (cell, 54-64 kpc, n=%d)" % field.sum())
styles = {"w1024full": ("C1", "-", 1.2), "w128": ("C2", "--", 1.2),
          "w1024clean": ("C3", ":", 2.2)}
w1s = {}
for nm in NAMES:
    c, ls, lw = styles[nm]
    g = cl_draws[nm][:, :, 2].reshape(-1)
    gw = np.repeat(w_cl, K)
    w1s[nm] = w1(v_cl[:, 2], g, w_cl, gw)
    ax.hist(g, bins=bins_h, weights=gw, density=True, histtype="step",
            lw=lw, color=c, ls=ls, label="%s (W1 %.3f)" % (nm, w1s[nm]))
ax.set_xlabel("vT (100 km/s)")
ax.set_ylabel("density")
ax.set_title("clump by pid vs conditional draws at member positions", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_clump_conditional_vT.png", dpi=150)
plt.close(fig)

print("=== CLUMP (pid) COMPARISON ===", flush=True)
for nm in NAMES:
    g = cl_draws[nm][:, :, 2].reshape(-1)
    gw = np.repeat(w_cl, K)
    print("%s: vT mean bias=%+.4f  std bias=%+.4f  W1=%.4f"
          % (nm, wmean(g, gw) - wmean(v_cl[:, 2], w_cl),
             wstd(g, gw) - wstd(v_cl[:, 2], w_cl), w1s[nm]), flush=True)

# ------------------------------------------------------------------
# figures 2-3: radial marginals vs clean-val (v1 protocol, unchanged)
# ------------------------------------------------------------------
prot = np.load(AUD / "eval_protocol.npz")
R_EDGES = prot["r_edges"]
n_rbins = len(R_EDGES) - 1
ir_cv = np.clip(np.digitize(r_cv, R_EDGES) - 1, 0, n_rbins - 1)
outer_c = r_cv >= 4.5
ir_out_c = (r_cv[outer_c] >= 6.0).astype(int)

key1 = jax.random.key(20260918)
z0_cv = jax.random.normal(key1, (n_cv, 3))
key2 = jax.random.key(918)
z0_out = jax.random.normal(key2, (int(outer_c.sum()), K, 3))

cv_draws = {}
out_draws = {}
for nm, run_dir in MODELS:
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    v1 = np.empty((n_cv, 3), dtype=np.float32)
    for i in range(0, n_cv, CHUNK):
        sl = slice(i, min(i + CHUNK, n_cv))
        v1[sl] = np.asarray(sample_one_stage(cvf, z0_cv[sl], jnp.asarray(eta_cv[sl, :3])))
    cv_draws[nm] = sph_vel(pos_cv, v1)
    vs = np.asarray(sample_k_stage(cvf, z0_out, jnp.asarray(eta_cv[outer_c, :3])))
    out_draws[nm] = sph_vel(np.repeat(eta_cv[outer_c, :3], K, axis=0),
                            vs.reshape(-1, 3)).reshape(int(outer_c.sum()), K, 3)
    del flow, cvf
    jax.clear_caches()

rows = []


def panel(ax, truth, w_t, draws, w_d, w1s2, title):
    bins = np.linspace(-4.5, 4.5, 73)
    ax.hist(truth, bins=bins, weights=w_t, density=True, alpha=0.45,
            color="C0", label="truth (clean-val)")
    styles = {"w1024full": ("C1", "-", 1.3), "w128": ("C2", "--", 1.3),
              "w1024clean": ("C3", ":", 2.0)}
    for nm in NAMES:
        c, ls, lw = styles[nm]
        ax.hist(draws[nm], bins=bins, weights=w_d[nm], density=True,
                histtype="step", lw=lw, color=c, ls=ls, label=nm)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-4.5, 4.5)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.97, "W1 " + " ".join("%s %.3f" % (SHORT[n], w1s2[n]) for n in NAMES),
            transform=ax.transAxes, fontsize=6.2, va="top")


fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for ci, cn in enumerate(COMP):
    for b, (lo, hi) in enumerate([(4.5, 6.0), (6.0, 7.5)]):
        m = ir_out_c == b
        w_b = w_cv[outer_c][m]
        truth = v_true_cv[outer_c][m, ci]
        draws = {nm: out_draws[nm][m][:, :, ci].reshape(-1) for nm in NAMES}
        w_d = {nm: np.repeat(w_b, K) for nm in NAMES}
        w1p = {nm: w1(truth, draws[nm], w_b, w_d[nm]) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="cleanval_outer", bin=4 + b, r_lo=lo * 10, r_hi=hi * 10,
                             comp=cn, model=nm, w1=w1p[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, w_d, w1p,
              "clean-val %g-%g kpc (n=%d)" % (lo * 10, hi * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("outer bins, CLEAN population truth (no clump); w1024full/w128 context (41% leakage)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_h12val_outer.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, n_rbins, figsize=(19, 7.8), sharex=True)
for ci, cn in enumerate(COMP):
    for b in range(n_rbins):
        m = ir_cv == b
        w_b = w_cv[m]
        truth = v_true_cv[m, ci]
        draws = {nm: cv_draws[nm][m, ci] for nm in NAMES}
        w1p = {nm: w1(truth, draws[nm], w_b, w_b) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="cleanval_bins", bin=b, r_lo=R_EDGES[b] * 10,
                             r_hi=R_EDGES[b + 1] * 10, comp=cn, model=nm,
                             w1=w1p[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, {nm: w_b for nm in NAMES}, w1p,
              "%g-%g kpc (n=%d)" % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, -1].legend(fontsize=7, loc="upper right")
fig.suptitle("CLEAN-val population: radial-bin velocity marginals (truth vs three DFs)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_strict.png", dpi=150)
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / "clean_truth_pid_w1.csv", index=False)
print("=== W1(vT) per radial bin (clean-val) ===", flush=True)
print(df[(df["set"] == "cleanval_bins") & (df["comp"] == "vT")]
      .pivot(index="bin", columns="model", values="w1").round(4).to_string(), flush=True)
with open(OUT / "summary.json", "w") as f:
    json.dump({"clump_w1": w1s, "w1": df.to_dict(orient="records"),
               "n_members": int(mem_f.sum()), "n_field": int(field.sum()),
               "registry_schema": str(reg["schema"])}, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("outputs under", OUT, flush=True)
