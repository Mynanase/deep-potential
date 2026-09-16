#!/usr/bin/env python
"""Phase-1 add-on 2: outer-bin check on halo12-val (w128's OWN validation set).

Companion to df_phase1_outer_confirm.py (which used clean-val, the own val
set of w512/w1024).  Here the truth is the first 25% of halo12.h5 -- the set
w128 was monitored on, so w128 has zero leakage (w512/w1024 have ~41%
leakage here, their columns are shown for context only).

Run from repo root:
  CUDA_VISIBLE_DEVICES=0 python scripts/auriga/df_phase1_outer_confirm_w128val.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_val = f["eta"][:413242]
    w_val = f["weights"][:413242]


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


K = 4
r_val = np.linalg.norm(eta_val[:, :3], axis=1)
outer_mask = r_val >= 4.5
pos = eta_val[outer_mask, :3]
v_true_sph = sph_vel(pos, eta_val[outer_mask, 3:])
w_out = w_val[outer_mask]
ir_out = np.where(r_val[outer_mask] >= 6.0, 1, 0)
print(f"outer halo12-val stars: {outer_mask.sum()} "
      f"(bin4={np.sum(ir_out == 0)}, bin5={np.sum(ir_out == 1)})")

key = jax.random.key(917)
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
for name, run_dir in [("w128", "runs/halo12-baseline"), ("w512", "runs/halo12-cap-w512"),
                      ("w1024", "runs/halo12-cap-w1024")]:
    flow = fit_all.load_flow(REPO / run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
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
    vs = np.asarray(sample_stage(cvf, z0, jnp.asarray(pos)))
    results[name] = sph_vel(np.repeat(pos, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos), K, 3)
    del flow, cvf
    jax.clear_caches()

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

df = pd.DataFrame(rows)
df.to_csv(OUT / "outer_h12val_metrics.csv", index=False)
cols = ["bin", "n_true"] + [f"{k}_{c}_{m}" for c in ["vr", "vth", "vT"]
                            for k in ["true_std"] for m in [""]]
sel = ["bin", "n_true"]
for c in ["vr", "vth", "vT"]:
    sel += [f"{c}_true_std"] + [f"bias_std_{c}_{m}" for m in ["w128", "w512", "w1024"]] \
           + [f"w1_{c}_{m}" for m in ["w128", "w512", "w1024"]]
print(df[sel].round(4).to_string(index=False))
