#!/usr/bin/env python
"""Radial velocity marginals with r bins chosen to COVER the clump.

Bin edges (kpc): [0, 10, 20, 30, 45, 64, 75] -- the boundary 60 -> 64 kpc
makes the bin [45, 64) contain 100% of the 9,683 clump members
(r 54.1-63.5 kpc). Truth = halo12-val (full population, clump present).
All three models draw K=4 samples at the SAME star positions per bin from
one shared noise stream (key 921), so the sampling bins are identical across
models by construction.

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_radial_rbin.py
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

REPO = Path(__file__).resolve().parents[3]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
REG = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "radial-rbin-20260917"
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
R_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 6.4, 7.5])  # code units x10 kpc


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


# truth: halo12-val (full population, clump included)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_f = f["eta"][:]
    w_f = f["weights"][:]
nv = int(len(eta_f) * 0.25)
eta = eta_f[:nv]
w = w_f[:nv]
pos = eta[:, :3]
r = np.linalg.norm(pos, axis=1)
ir = np.clip(np.digitize(r, R_EDGES) - 1, 0, len(R_EDGES) - 2)
v_true = sph_vel(pos, eta[:, 3:])

# clump membership for per-bin annotation (val rows only)
det = np.asarray(np.load(REG)["detection_pids"]).astype(np.uint64)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    pid_f = f["particle_id"][:]
mem = np.isin(pid_f[:nv], det)
r_mem = r[mem]
print("clump members in halo12-val: %d ; in bin [4.5,6.4): %d (%.1f%%)"
      % (mem.sum(), ((r_mem >= 4.5) & (r_mem < 6.4)).sum(),
         100 * ((r_mem >= 4.5) & (r_mem < 6.4)).mean()), flush=True)

key = jax.random.key(921)
z0 = jax.random.normal(key, (nv, K, 3))


@eqx.filter_jit
def sample_k_stage(cvf, z0_, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0_.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


draws = {}
for nm, run_dir in MODELS:
    print("===== %s =====" % nm, flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(pos[:256])
    explicit = np.asarray(sample_k_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2
    out = np.empty((nv, K, 3), dtype=np.float32)
    flat = nv * K
    for i in range(0, flat, 4 * CHUNK):
        rows = np.arange(i, min(i + 4 * CHUNK, flat))
        star = rows // K
        draw = rows % K
        zz = z0.reshape(flat, 3)[rows]
        xx = jnp.asarray(pos[star])
        cond = (xx - cvf.cond_mean) / cvf.cond_std
        vs = jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(zz, cond)
        out.reshape(flat, 3)[rows] = np.asarray(vs)
    draws[nm] = sph_vel(np.repeat(pos, K, axis=0), out.reshape(flat, 3)).reshape(nv, K, 3)
    del flow, cvf
    jax.clear_caches()

rows = []


def panel(ax, truth, w_t, dr, w_d, w1p, title, clump_frac=None):
    bins = np.linspace(-4.5, 4.5, 73)
    ax.hist(truth, bins=bins, weights=w_t, density=True, alpha=0.45,
            color="C0", label="truth (halo12-val)")
    styles = {"w1024full": ("C1", "-", 1.3), "w128": ("C2", "--", 1.3),
              "w1024clean": ("C3", ":", 2.0)}
    for nm in NAMES:
        c, ls, lw = styles[nm]
        ax.hist(dr[nm], bins=bins, weights=w_d[nm], density=True,
                histtype="step", lw=lw, color=c, ls=ls, label=nm)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-4.5, 4.5)
    ax.tick_params(labelsize=7)
    txt = "W1 " + " ".join("%s %.3f" % (SHORT[n], w1p[n]) for n in NAMES)
    if clump_frac is not None:
        txt = "clump %.0f%% mass\n" % (100 * clump_frac) + txt
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.2, va="top")


n_rbins = len(R_EDGES) - 1
fig, axes = plt.subplots(3, n_rbins, figsize=(19, 7.8), sharex=True)
for ci, cn in enumerate(COMP):
    for b in range(n_rbins):
        m = ir == b
        w_b = w[m]
        truth = v_true[m, ci]
        dr = {nm: draws[nm][m][:, :, ci].reshape(-1) for nm in NAMES}
        w_d = {nm: np.repeat(w_b, K) for nm in NAMES}
        w1p = {nm: w1(truth, dr[nm], w_b, w_d[nm]) for nm in NAMES}
        cfrac = float(w[m & mem].sum() / w_b.sum()) if b == 4 else None
        for nm in NAMES:
            rows.append(dict(bin=b, r_lo=R_EDGES[b] * 10, r_hi=R_EDGES[b + 1] * 10,
                             comp=cn, model=nm, w1=w1p[nm], n_true=int(m.sum()),
                             clump_mass_frac=cfrac if cfrac is not None else ""))
        ax = axes[ci, b]
        panel(ax, truth, w_b, dr, w_d, w1p,
              "%g-%g kpc (n=%d)" % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10, int(m.sum())),
              clump_frac=cfrac)
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, -1].legend(fontsize=7, loc="upper right")
fig.suptitle("halo12-val radial bins (edges cover the clump: 45-64 kpc holds 100% of members)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_rbins_full.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for ci, cn in enumerate(COMP):
    for b, (lo, hi) in enumerate([(4.5, 6.4), (6.4, 7.5)]):
        m = (r >= lo) & (r < hi)
        w_b = w[m]
        truth = v_true[m, ci]
        dr = {nm: draws[nm][m][:, :, ci].reshape(-1) for nm in NAMES}
        w_d = {nm: np.repeat(w_b, K) for nm in NAMES}
        w1p = {nm: w1(truth, dr[nm], w_b, w_d[nm]) for nm in NAMES}
        cfrac = float(w[m & mem].sum() / w_b.sum()) if lo == 4.5 else None
        ax = axes[ci, b]
        panel(ax, truth, w_b, dr, w_d, w1p,
              "halo12-val %g-%g kpc (n=%d)" % (lo * 10, hi * 10, int(m.sum())),
              clump_frac=cfrac)
        if b == 0:
            ax.set_ylabel(cn, fontsize=10)
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle("outer bins covering the clump (45-64 kpc = 100% of members); same positions for all models", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_rbins_outer.png", dpi=150)
plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / "radial_rbin_w1.csv", index=False)
print("=== W1(vT) per bin (45-64 covers clump) ===", flush=True)
print(df[df["comp"] == "vT"].pivot(index="bin", columns="model", values="w1").round(4).to_string(), flush=True)
print("clump mass fraction in bin4 [45,64): see CSV / annotations", flush=True)
with open(OUT / "summary.json", "w") as f:
    json.dump({"w1": df.to_dict(orient="records"),
               "edges_kpc": (R_EDGES * 10).tolist(),
               "clump_members_val": int(mem.sum()),
               "clump_in_bin4": int(((r_mem >= 4.5) & (r_mem < 6.4)).sum())},
              f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("outputs under", OUT, flush=True)
