#!/usr/bin/env python
"""fig_clump_conditional_vT with the clump-REMOVED truth as reference.

Curves:
  filled blue  clump truth = detection_pids members (n=9683, from halo12.h5)
  gray step    clean truth = halo12-clean.h5 ALL rows (truth-only reference;
               leakage does not apply) in the clump cell and radius band
               54-64 kpc -- the post-removal population actually present
               around the clump (includes unregistered diffuse debris)
  3 model lines  conditional draws at the member positions (K=4, key 920,
               identical protocol to the v2 node -> identical draws)

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_clump_fig_reference.py
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
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REG = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "clump-fig-reference-20260917"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = [("w1024clean", REPO / "runs" / "halo12-cap-w1024"),
          ("w1024full", REPO / "runs" / "w1024full"),
          ("w128", REPO / "runs" / "halo12-baseline")]
NAMES = [m[0] for m in MODELS]
K = 4


def sph_vT(pos, vel):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.hypot(x, y)
    return (-vel[:, 0] * y + vel[:, 1] * x) / R


def w1(a, b, wa, wb):
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))


def wstd(a, w):
    mu = wmean(a, w)
    return float(np.sqrt(np.sum(w * (a - mu) ** 2) / np.sum(w)))


det = np.asarray(np.load(REG)["detection_pids"]).astype(np.uint64)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    pid_f = f["particle_id"][:]
    eta_f = f["eta"][:]
    w_f = f["weights"][:]
mem = np.isin(pid_f, det)
assert int(mem.sum()) == 9683
pos_cl = eta_f[mem, :3]
w_cl = w_f[mem]
v_cl = sph_vT(pos_cl, eta_f[mem, 3:])

# clump-removed truth: ALL clean rows in the clump cell + radius band
with h5py.File(REPO / "data" / "auriga" / "halo12-clean.h5", "r") as f:
    eta_c = f["eta"][:]
    w_c = f["weights"][:]
pos_c = eta_c[:, :3]
r_c = np.linalg.norm(pos_c, axis=1)
cth_c = pos_c[:, 2] / r_c
phi_c = np.mod(np.arctan2(pos_c[:, 1], pos_c[:, 0]), 2 * np.pi) - np.pi
clean_ref = (cth_c < -0.6) & (phi_c >= np.pi / 2) & (phi_c < np.pi) \
    & (r_c >= 5.4) & (r_c < 6.4)
v_clean = sph_vT(pos_c[clean_ref], eta_c[clean_ref, 3:])
w_clean = w_c[clean_ref]
print("clump truth        : n=%d  vT=%+.3f+-%.3f  frac<-1=%.3f"
      % (mem.sum(), wmean(v_cl, w_cl), wstd(v_cl, w_cl),
         float(w_cl[v_cl < -1].sum() / w_cl.sum())), flush=True)
print("clean truth (band) : n=%d  vT=%+.3f+-%.3f  frac<-1=%.3f"
      % (clean_ref.sum(), wmean(v_clean, w_clean), wstd(v_clean, w_clean),
         float(w_clean[v_clean < -1].sum() / w_clean.sum())), flush=True)

key = jax.random.key(920)
z0 = jax.random.normal(key, (len(pos_cl), K, 3))


@eqx.filter_jit
def sample_k_stage(cvf, z0_, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0_.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


draws = {}
gw = np.repeat(w_cl, K)
w1s = {}
for nm, run_dir in MODELS:
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(pos_cl[:256])
    explicit = np.asarray(sample_k_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2
    vs = np.asarray(sample_k_stage(cvf, z0, jnp.asarray(pos_cl)))
    draws[nm] = sph_vT(np.repeat(pos_cl, K, axis=0), vs.reshape(-1, 3))
    w1s[nm] = w1(v_cl, draws[nm], w_cl, gw)
    del flow, cvf
    jax.clear_caches()

fig, ax = plt.subplots(figsize=(8.4, 4.8))
bins_h = np.linspace(-4.5, 4.5, 73)
ax.hist(v_cl, bins=bins_h, weights=w_cl, density=True, alpha=0.5, color="C0",
        label="clump truth (detection_pids, n=%d)" % mem.sum())
ax.hist(v_clean, bins=bins_h, weights=w_clean, density=True, histtype="step",
        lw=1.6, color="0.35",
        label="clean truth, clump removed (cell+54-64 kpc, all rows, n=%d)" % clean_ref.sum())
styles = {"w1024full": ("C1", "-", 1.2), "w128": ("C2", "--", 1.2),
          "w1024clean": ("C3", ":", 2.2)}
for nm in NAMES:
    c, ls, lw = styles[nm]
    ax.hist(draws[nm], bins=bins_h, weights=gw, density=True, histtype="step",
            lw=lw, color=c, ls=ls, label="%s (W1 %.3f)" % (nm, w1s[nm]))
ax.set_xlabel("vT (100 km/s)")
ax.set_ylabel("density")
ax.set_title("clump region: clump truth vs clump-removed truth vs conditional draws", fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_clump_conditional_vT.png", dpi=150)
plt.close(fig)

summary = dict(
    clump=dict(n=int(mem.sum()), vT_mean=wmean(v_cl, w_cl), vT_std=wstd(v_cl, w_cl)),
    clean_ref=dict(n=int(clean_ref.sum()), vT_mean=wmean(v_clean, w_clean),
                   vT_std=wstd(v_clean, w_clean)),
    model_w1_vs_clump=w1s,
)
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("=== SUMMARY ===", flush=True)
print(json.dumps(summary, indent=1), flush=True)
print("outputs under", OUT, flush=True)
