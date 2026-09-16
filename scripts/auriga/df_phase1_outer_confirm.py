#!/usr/bin/env python
"""Phase-1 add-on: confirm the outer-bin story on clean-val (secondary set).

clean-val is the monitoring validation set of w512/w1024 (zero leakage for
them, 41% leakage into w128's training set).  We generate conditional draws
at ALL clean-val positions with r >= 4.5 (code units) with the same shared
noise protocol and recompute the radial-bin W1 / moment-bias table for
r in [4.5, 6) and [6, 7.5).

Run from repo root:
  CUDA_VISIBLE_DEVICES=0 python scripts/auriga/df_phase1_outer_confirm.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402
from scipy.stats import wasserstein_distance


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

prot = np.load(OUT / "eval_protocol.npz")
R_EDGES = prot["r_edges"]
K = 4
eta_cval = prot["eta_cval"]
w_cval = prot["w_cval"]
r_cval = np.linalg.norm(eta_cval[:, :3], axis=1)
outer_mask = r_cval >= 4.5
pos = eta_cval[outer_mask, :3]
v_true_cart = eta_cval[outer_mask, 3:]
w_out = w_cval[outer_mask]
v_true_sph = sph_vel(pos, v_true_cart)
ir_out = np.where(r_cval[outer_mask] >= 6.0, 1, 0)   # 0: [4.5,6), 1: [6,7.5)
print(f"outer clean-val stars: {outer_mask.sum()} (bin4={np.sum(ir_out==0)}, bin5={np.sum(ir_out==1)})")

key = jax.random.key(916)
z0 = jax.random.normal(key, (len(pos), K, 3))

@eqx.filter_jit
def sample_stage(cvf, z0, x):
    """v2 fix: pass the NORMALIZED condition into the bijection. v1 computed
    cond but passed raw x, so draws were conditioned on cond_mean+cond_std*x
    instead of x (caught in the 2026-09-16 independent review)."""
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)

results = {}
sample_store = {}
for name, run_dir in [("w128", "runs/halo12-baseline"), ("w512", "runs/halo12-cap-w512"),
                      ("w1024", "runs/halo12-cap-w1024")]:
    flow = fit_all.load_flow(REPO / run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    # regression guard: the explicit path must reproduce NormalizingFlow.sample
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    explicit = np.asarray(sample_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), jnp.asarray(pos[:256])))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=jnp.asarray(pos[:256])))
    # tolerance 2e-2: w128/w512 agree bitwise; w1024 shows up to ~6e-3 solver-level
    # differences at stiff OUTER positions (Tsit5 rtol 1e-4), i.e. ODE numerical
    # noise, not a conditioning bug (v1's cond bug produced O(1) differences)
    dmax = np.abs(explicit[:, 0] - builtin).max()
    print(f"guard max|explicit-builtin| = {dmax:.2e} "
          f"(rows >1e-3: {int((np.abs(explicit[:, 0] - builtin) > 1e-3).sum())}/256)", flush=True)
    assert dmax < 2e-2, "sample_stage mismatch"
    vs = np.asarray(sample_stage(cvf, z0, jnp.asarray(pos)))     # (n, K, 3) cartesian
    results[name] = sph_vel(np.repeat(pos, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos), K, 3)
    sample_store[name] = vs.astype(np.float32)
    del flow, cvf
    jax.clear_caches()

np.savez_compressed(OUT / "outer_cleanval_samples.npz", pos=pos.astype(np.float32),
                    w_out=w_out, ir_out=ir_out,
                    **{f"vel_{m}": sample_store[m] for m in sample_store})

rows = []
for b, (lo, hi) in enumerate([(4.5, 6.0), (6.0, 7.5)]):
    m = ir_out == b
    v_t = v_true_sph[m]
    w_t = w_out[m]
    row = dict(bin=4 + b, r_lo=lo, r_hi=hi, n_true=int(m.sum()))
    for c, cname in enumerate(["vr", "vth", "vT"]):
        row[f"{cname}_true_mean"] = wmean(v_t[:, c], w_t)
        row[f"{cname}_true_std"] = wstd(v_t[:, c], w_t)
        for name in results:
            vg = results[name][m][:, :, c].reshape(-1)
            gw = np.repeat(w_t, K)   # v2: each draw carries its position's mass
            row[f"w1_{cname}_{name}"] = wasserstein_distance(v_t[:, c], vg, u_weights=w_t, v_weights=gw)
            row[f"bias_mean_{cname}_{name}"] = wmean(vg, gw) - row[f"{cname}_true_mean"]
            row[f"bias_std_{cname}_{name}"] = wstd(vg, gw) - row[f"{cname}_true_std"]
    rows.append(row)

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
df.to_csv(OUT / "outer_cleanval_metrics.csv", index=False)
print(df.round(4).to_string(index=False))
