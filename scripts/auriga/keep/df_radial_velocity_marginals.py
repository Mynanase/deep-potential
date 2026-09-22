#!/usr/bin/env python
"""Radial-bin velocity marginal distributions for the three DF models:
w1024full (full population), w128 (full population), w1024clean (clean).

Figure A (strict set, leakage-free for all three): truth = strict stars per
radial bin; models = one shared-noise draw at every strict position (audit
key-20260916 protocol). Figure B (halo12-val outer 45-75 kpc): truth
includes the clump stream; models = K=4 guarded draws (key-917 protocol).

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_radial_velocity_marginals.py
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
OUT = REPO / "runs" / "radial-velocity-marginals-20260917"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = [("w1024full", REPO / "runs" / "w1024full"),
          ("w128", REPO / "runs" / "halo12-baseline"),
          ("w1024clean", REPO / "runs" / "halo12-cap-w1024")]
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
eta_strict = prot["eta_strict"]
w_strict = prot["w_strict"]
ir = prot["ir_strict"]
v_true_strict = np.stack([prot["vr_strict"], prot["vth_strict"], prot["vT_strict"]], axis=1)
R_EDGES = prot["r_edges"]
n_rbins = len(R_EDGES) - 1

key0 = jax.random.key(20260916)
_, key_z0_all, _, _, _ = jax.random.split(key0, 5)
z0_all = jax.random.normal(key_z0_all, (len(eta_strict), 3))

with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_full = f["eta"][:]
    w_full = f["weights"][:]
n_val = int(len(eta_full) * 0.25)
eta_val = eta_full[:n_val]
w_val = w_full[:n_val]
r_val = np.linalg.norm(eta_val[:, :3], axis=1)
outer = r_val >= 4.5
pos_out = eta_val[outer, :3]
w_out = w_val[outer]
ir_out = (r_val[outer] >= 6.0).astype(int)
v_true_out = sph_vel(pos_out, eta_val[outer, 3:])
print("outer halo12-val stars: %d (bin4=%d, bin5=%d)"
      % (outer.sum(), int((ir_out == 0).sum()), int((ir_out == 1).sum())), flush=True)

key917 = jax.random.key(917)
z0_out = jax.random.normal(key917, (len(pos_out), K, 3))


@eqx.filter_jit
def sample_one_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(z0, cond)


@eqx.filter_jit
def sample_k_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


strict_draws = {}
outer_draws = {}
for nm, run_dir in MODELS:
    print("===== %s: %s =====" % (nm, run_dir), flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(eta_strict[:256, :3])
    explicit = np.asarray(sample_k_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2, "sample path mismatch for " + nm

    v1 = np.empty((len(eta_strict), 3), dtype=np.float32)
    for i in range(0, len(eta_strict), CHUNK):
        sl = slice(i, min(i + CHUNK, len(eta_strict)))
        v1[sl] = np.asarray(sample_one_stage(cvf, z0_all[sl], jnp.asarray(eta_strict[sl, :3])))
    strict_draws[nm] = sph_vel(eta_strict[:, :3], v1)

    vs = np.asarray(sample_k_stage(cvf, z0_out, jnp.asarray(pos_out)))
    outer_draws[nm] = sph_vel(np.repeat(pos_out, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos_out), K, 3)
    del flow, cvf
    jax.clear_caches()


def panel(ax, truth, w_t, draws, w_d, w1s, title, show_w1=True):
    bins = np.linspace(-4.5, 4.5, 73)
    ax.hist(truth, bins=bins, weights=w_t, density=True, alpha=0.45,
            color="C0", label="truth")
    styles = {"w1024full": ("C1", "-"), "w128": ("C2", "--"), "w1024clean": ("C3", ":")}
    for nm in NAMES:
        c, ls = styles[nm]
        ax.hist(draws[nm], bins=bins, weights=w_d[nm], density=True,
                histtype="step", lw=1.4, color=c, ls=ls, label=nm)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-4.5, 4.5)
    ax.tick_params(labelsize=7)
    if show_w1:
        txt = "W1 " + " ".join("%s %.3f" % (SHORT[nm], w1s[nm]) for nm in NAMES)
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.2, va="top")


rows = []
fig, axes = plt.subplots(3, n_rbins, figsize=(19, 7.8), sharex=True)
for ci, cn in enumerate(COMP):
    for b in range(n_rbins):
        m = ir == b
        w_b = w_strict[m]
        truth = v_true_strict[m, ci]
        draws = {nm: strict_draws[nm][m, ci] for nm in NAMES}
        w1s = {nm: w1(truth, draws[nm], w_b, w_b) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="strict", bin=b, r_lo=R_EDGES[b] * 10,
                             r_hi=R_EDGES[b + 1] * 10, comp=cn, model=nm,
                             w1=w1s[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, {nm: w_b for nm in NAMES}, w1s,
              "%g-%g kpc (n=%d)" % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, -1].legend(fontsize=7, loc="upper right")
fig.suptitle("Strict common set: radial-bin velocity marginals (truth vs three DFs)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_strict.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for ci, cn in enumerate(COMP):
    for b, (lo, hi) in enumerate([(4.5, 6.0), (6.0, 7.5)]):
        m = ir_out == b
        w_b = w_out[m]
        truth = v_true_out[m, ci]
        draws = {nm: outer_draws[nm][m][:, :, ci].reshape(-1) for nm in NAMES}
        w_d = {nm: np.repeat(w_b, K) for nm in NAMES}
        w1s = {nm: w1(truth, draws[nm], w_b, w_d[nm]) for nm in NAMES}
        for nm in NAMES:
            rows.append(dict(set="h12val_outer", bin=4 + b, r_lo=lo * 10, r_hi=hi * 10,
                             comp=cn, model=nm, w1=w1s[nm], n_true=int(m.sum())))
        ax = axes[ci, b]
        panel(ax, truth, w_b, draws, w_d, w1s,
              "halo12-val %g-%g kpc (n=%d)" % (lo * 10, hi * 10, int(m.sum())))
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("halo12-val outer bins (truth incl. clump stream): velocity marginals", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_marginals_h12val_outer.png", dpi=150)
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / "radial_marginal_w1.csv", index=False)
print("=== W1(vT) per radial bin (strict) ===", flush=True)
piv = df[(df["set"] == "strict") & (df["comp"] == "vT")].pivot(index="bin", columns="model", values="w1")
print(piv.round(4).to_string(), flush=True)
print("outputs under", OUT, flush=True)
with open(OUT / "summary.json", "w") as f:
    json.dump({"w1": df.to_dict(orient="records")}, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
