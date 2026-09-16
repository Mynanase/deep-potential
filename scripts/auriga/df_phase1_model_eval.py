#!/usr/bin/env python
"""Phase-1 DF audit, step 2/3: conditional-velocity diagnostics for the three
final checkpoints (w128 = halo12-baseline, w512, w1024), one shared protocol.

For every model, at the SAME validation positions with the SAME base noise:

  1. conditional velocity samples  v_ik ~ p_theta(v | x_i)   (K=4 per position)
  2. conditional NLL  -log p_theta(v_i | x_i)  on the strict common set
     and on clean-val (secondary; leaked for w128)
  3. unregularized FM validation residual (t, x0 drawn once, shared across
     models), stored in PHYSICAL velocity units (100 km/s)
  4. spatial-flow position samples (262144, shared base noise across models)

All models use their own baked-in normalization (cond_mean/cond_std and the
velocity denormalizer), so inputs/outputs are in code units (10 kpc, 100 km/s).
ODE solver: training-time defaults (Tsit5, rtol 1e-4, atol 1e-5).

Run from repo root:
  CUDA_VISIBLE_DEVICES=0 python scripts/auriga/df_phase1_model_eval.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "halo12-phase1-df-audit-20260916"
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

# ------------------------------------------------------------------
# 1. shared protocol: evaluation data, shared noise draws
# ------------------------------------------------------------------
prot = np.load(OUT / "eval_protocol.npz")
eta_strict = prot["eta_strict"]
w_strict = prot["w_strict"]
eta_cval = prot["eta_cval"]
sel_idx = prot["sel_idx"]
K = int(prot["k_draws"])

pos_sel = eta_strict[sel_idx, :3]          # (n_sel, 3) fixed generation positions
n_sel = len(pos_sel)

N_FM_DRAWS = 4                              # (t, x0) draws per star for the FM residual
N_POS_SAMPLES = 262144                      # spatial-flow samples per model

key0 = jax.random.key(20260916)
key_z0_vel, key_z0_all, key_fm_t, key_fm_x0, key_z0_pos = jax.random.split(key0, 5)
# base noise for conditional sampling: (n_sel, K, 3), IDENTICAL for all models
z0_vel = jax.random.normal(key_z0_vel, (n_sel, K, 3))
# base noise for the full-coverage single draw at EVERY strict position (v2:
# direction-cell diagnostics need all positions, not only the stratified 4284)
z0_all = jax.random.normal(key_z0_all, (len(eta_strict), 3))
# shared FM draws on the strict set and clean-val. v2: t and x0 come from
# SEPARATE keys -- v1 drew them from the same key, so t was a deterministic
# function of the same bit stream as x0 (correlated (t, x0) pairs).
t_fm_strict = jax.random.uniform(key_fm_t, (len(eta_strict), N_FM_DRAWS))
x0_fm_strict = jax.random.normal(key_fm_x0, (len(eta_strict), N_FM_DRAWS, 3))
t_fm_cval = jax.random.uniform(jax.random.fold_in(key_fm_t, 1), (len(eta_cval), N_FM_DRAWS))
x0_fm_cval = jax.random.normal(jax.random.fold_in(key_fm_x0, 2), (len(eta_cval), N_FM_DRAWS, 3))
# base noise for spatial sampling: (N_POS_SAMPLES, 3), IDENTICAL for all models
z0_pos_all = jax.random.normal(key_z0_pos, (N_POS_SAMPLES, 3))

# ------------------------------------------------------------------
# 2. shared compiled stages (no model captured in the closure)
# ------------------------------------------------------------------
@eqx.filter_jit
def sample_vel_stage(cvf, z0, x):
    """v = bijection.transform(z0 | cond-normalized x), replicating
    NormalizingFlow.sample but with explicit base noise."""
    cond = (x - cvf.cond_mean) / cvf.cond_std
    fn = jax.vmap(lambda z, c: cvf.flow.bijection.transform(z, condition=c))
    return fn(z0, cond)

@eqx.filter_jit
def nll_stage(cvf, v, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return cvf.flow.log_prob(v, condition=cond)

@eqx.filter_jit
def fm_stage(net, t, xt, cond):
    """predicted vector field vt for given (t, xt) in normalized space."""
    return jax.vmap(net)(t, xt, condition=cond)

@eqx.filter_jit
def sample_pos_stage(sf, z0):
    fn = jax.vmap(lambda z: sf.flow.bijection.transform(z))
    return fn(z0)

CHUNK_SAMPLE = 8192
CHUNK_NLL = 8192
CHUNK_FM = 65536

MODELS = [
    ("w128", REPO / "runs" / "halo12-baseline"),
    ("w512", REPO / "runs" / "halo12-cap-w512"),
    ("w1024", REPO / "runs" / "halo12-cap-w1024"),
]

for model_name, run_dir in MODELS:
    t_start = time.time()
    print(f"===== {model_name}: {run_dir} =====", flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    sf = flow.spatial_flow
    net = cvf.flow.bijection[0].dynamics_net
    # the velocity denormalizer is Affine(scale=Parameterize(softplus), loc):
    # recover the physical scale vector baked in at training time
    vel_std = np.asarray(jax.nn.softplus(np.asarray(cvf.flow.bijection[1].scale.args[0])))
    vel_mean = np.asarray(cvf.flow.bijection[1].loc)
    print(f"vel_mean={vel_mean} vel_std={vel_std}", flush=True)

    # --- sanity: explicit-noise sampling reproduces NormalizingFlow.sample ---
    key_chk = jax.random.key(7)
    z_chk = cvf.flow.base_dist.sample(key_chk, (512,)).block_until_ready()
    x_chk = jnp.asarray(pos_sel[:512])
    explicit = np.asarray(sample_vel_stage(cvf, z_chk, x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 512, condition=x_chk))
    assert np.allclose(explicit, builtin, atol=1e-4), "explicit sampling path mismatch"
    print("explicit-noise sampling matches NormalizingFlow.sample", flush=True)

    # --- 1. conditional velocity samples at the stratified positions ---
    vel_samples = np.empty((n_sel, K, 3), dtype=np.float32)
    for a, b in [(i, min(i + CHUNK_SAMPLE, n_sel * K)) for i in range(0, n_sel * K, CHUNK_SAMPLE)]:
        rows = np.arange(a, b)
        z_chunk = z0_vel.reshape(n_sel * K, 3)[rows]
        x_chunk = jnp.repeat(jnp.asarray(pos_sel), K, axis=0)[rows]
        v_chunk = sample_vel_stage(cvf, z_chunk, x_chunk)
        vel_samples.reshape(n_sel * K, 3)[rows] = np.asarray(v_chunk)
    print(f"conditional samples done ({time.time() - t_start:.0f}s)", flush=True)

    # --- 1b. full-coverage single draw at EVERY strict position (v2) ---
    # one draw per position with identical base noise across models; used for
    # direction-cell diagnostics where the stratified 4284 positions are too few
    vel_samples_all = np.empty((len(eta_strict), 3), dtype=np.float32)
    for i in range(0, len(eta_strict), CHUNK_NLL):
        sl = slice(i, min(i + CHUNK_NLL, len(eta_strict)))
        v_chunk = sample_vel_stage(cvf, z0_all[sl], jnp.asarray(eta_strict[sl, :3]))
        vel_samples_all[sl] = np.asarray(v_chunk)
    print(f"full-coverage strict draws done ({time.time() - t_start:.0f}s)", flush=True)

    # --- 2. conditional NLL on strict set and clean-val ---
    def nll_all(eta):
        n = len(eta)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, CHUNK_NLL):
            sl = slice(i, min(i + CHUNK_NLL, n))
            lp = nll_stage(cvf, jnp.asarray(eta[sl, 3:]), jnp.asarray(eta[sl, :3]))
            out[sl] = np.asarray(lp)
        return out
    nll_strict = nll_all(eta_strict)
    print(f"NLL strict done ({time.time() - t_start:.0f}s): "
          f"weighted mean = {float(np.sum(w_strict * nll_strict) / np.sum(w_strict)):.4f}", flush=True)
    nll_cval = nll_all(eta_cval)
    print(f"NLL clean-val done ({time.time() - t_start:.0f}s)", flush=True)

    # --- 3. FM validation residual (physical units), shared (t, x0) ---
    def fm_resid(eta, t_fm, x0_fm):
        n = len(eta)
        x1 = jnp.asarray((eta[:, 3:] - vel_mean) / vel_std)
        cond = jnp.asarray((eta[:, :3] - np.asarray(cvf.cond_mean)) / np.asarray(cvf.cond_std))
        resid = np.empty((n, N_FM_DRAWS, 3), dtype=np.float32)
        flat_n = n * N_FM_DRAWS
        for i in range(0, flat_n, CHUNK_FM):
            rows = np.arange(i, min(i + CHUNK_FM, flat_n))
            star = rows // N_FM_DRAWS
            draw = rows % N_FM_DRAWS
            t = t_fm[star, draw]
            x0 = x0_fm[star, draw]
            x1b = x1[star]
            xt = t[:, None] * x1b + (1 - t[:, None]) * x0
            vt = fm_stage(net, t, xt, cond[star])
            resid.reshape(flat_n, 3)[rows] = np.asarray(vt - (x1b - x0)) * vel_std
        return resid
    fm_resid_strict = fm_resid(eta_strict, t_fm_strict, x0_fm_strict)
    print(f"FM residual strict done ({time.time() - t_start:.0f}s): "
          f"mean|res|^2 = {float(np.mean(np.sum(fm_resid_strict**2, axis=-1))):.4f} (100 km/s)^2", flush=True)
    fm_resid_cval = fm_resid(eta_cval, t_fm_cval, x0_fm_cval)
    print(f"FM residual clean-val done ({time.time() - t_start:.0f}s)", flush=True)

    # --- 4. spatial-flow position samples ---
    pos_samples = np.empty((N_POS_SAMPLES, 3), dtype=np.float32)
    for i in range(0, N_POS_SAMPLES, CHUNK_FM):
        sl = slice(i, min(i + CHUNK_FM, N_POS_SAMPLES))
        pos_samples[sl] = np.asarray(sample_pos_stage(sf, z0_pos_all[sl]))
    print(f"spatial samples done ({time.time() - t_start:.0f}s)", flush=True)

    np.savez_compressed(
        OUT / f"model_{model_name}.npz",
        vel_samples=vel_samples, vel_samples_all=vel_samples_all,
        nll_strict=nll_strict, nll_cval=nll_cval,
        fm_resid_strict=fm_resid_strict, fm_resid_cval=fm_resid_cval,
        pos_samples=pos_samples, vel_mean=vel_mean, vel_std=vel_std,
    )
    print(f"saved {OUT / f'model_{model_name}.npz'} ({time.time() - t_start:.0f}s total)", flush=True)

    del flow, cvf, sf, net
    jax.clear_caches()

print("all models done")
