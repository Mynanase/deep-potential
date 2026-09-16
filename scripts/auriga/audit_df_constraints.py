#!/usr/bin/env python
"""Halo12 step-6 audit: DF score chain, CBE residual, local force constraints.

(plan: vault 30-Experiments/2026-09-14-halo12-mass-audit-plan.md, step 6.
Execution log: runs/halo12-mass-audit-20260914T0453/progress.md, R6.)

Conventions fixed by the plan (code units throughout):

  q = x / L,  p = v / V          (eta = [q, p], L = 10 kpc, V = 100 km/s)
  Phi_phys = V^2 * phi           (phi = phi_model(q), dimensionless)

  Steady-state CBE in code units (frameshift of the audited runs is zero
  and frozen, so it drops out):

      R = term1 - term2,   term1 = p . d_q lnF,   term2 = (d_q phi) . d_p lnF

  Everything below is dimensionless in these units.

Pre-declared scales and criteria (fixed BEFORE the audit runs, from
runs/halo12-baseline/data/df_gradients.h5 statistics measured 2026-09-15:
median |p| = 2.13, median |d_q lnF| = 9.43, median |d_p lnF| = 2.49,
median |term1| = 8.17 with p16-p84 = 1.49-24.86):

  CBE_EPS     = 1.0    epsilon in |R|/(|t1|+|t2|+eps): round scale ~12% of
                      the median |term1|, well below the typical term scale,
                      only there to kill spikes where both terms -> 0.
  TH_SCORE_MEDIAN / TH_SCORE_P99 : stored-vs-autodiff floored relative error
                      (x64 recompute vs stored f32 GPU values).
  TH_ODE_MEDIAN     : default-vs-strict ODE tolerance floored rel. error.
  TH_FD_PLATEAU     : central-FD-vs-autodiff plateau (best step size).
  RCOND_SVD = 1e-6   : singular-value cutoff for effective rank / lstsq.
  COND_MAX = 1e4     : "well-conditioned" position (for g_hat vs grad phi).

All thresholds are ENGINEERING CRITERIA, not significance levels.

Products (in <audit-dir>/step6/, never overwritten without --force):
  score_audit.json / .npz
  cbe_diagnostics.json / .npz
  local_force_constraints.json / .npz

Run from the repo root, e.g.:
  JAX_PLATFORMS=cpu python scripts/auriga/audit_df_constraints.py score-audit --smoke
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

from validate_enclosed_mass import (  # noqa: E402
    RUNS, TRUTH_PATH, DF_CKPT_INDEX, PHI_CKPT_INDEX,
    _dump_json, finite_or_fail, sha256_file, sobol_directions,
)

DEFAULT_AUDIT_DIR = REPO / "runs/halo12-mass-audit-20260914T0453"

L_KPC, V_KMS = 10.0, 100.0

# ---- pre-declared scales / engineering criteria (see module docstring) ----
CBE_EPS = 1.0
TH_SCORE_MEDIAN = 1e-3
TH_SCORE_P99 = 1e-2
TH_ODE_MEDIAN = 1e-3
TH_FD_PLATEAU = 5e-2
RCOND_SVD = 1e-6
COND_MAX = 1e4

FD_STEPS = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-5, 3e-5, 1e-6, 3e-6])
ODE_STRICT_RTOL, ODE_STRICT_ATOL = 1e-7, 1e-8   # vs production 1e-4 / 1e-5

# audit-region anchors (programmatic; asserted against the truth file)
A2_KPC_TARGET = 2.0        # first truth outer edge >= this (step-4 anchor 2.128006)
ROUT_KPC = 65.0            # outer edge of the three-run common region

# sampling seeds (fixed, independent of previous steps)
SEED_DIRS_CBE = 20260915
SEED_DIRS_FORCE = 20260916
SEED_VEL = 20260917

SCORE_N_ROWS = 4096        # stratified stored rows per run
SCORE_N_STRICT = 1024      # subset for the strict-tolerance recompute
SCORE_N_FD = 32            # subset for the finite-difference scan
CBE_N_RADII, CBE_N_DIRS, CBE_N_VEL = 32, 16, 8   # 32*16*8 = 4096 points
FORCE_N_RADII, FORCE_N_DIRS = 16, 4             # 64 positions
FORCE_M_TIERS = (32, 128)                       # velocities per position
CBE_N_STORED_ROWS = 1024    # stored-training-point R comparison

BATCH = 64                  # matches production grad_batch_size


# --------------------------------------------------------------------------
# generic, model-agnostic helpers (unit-tested on analytic cases)
# --------------------------------------------------------------------------

def rel_with_floor(diff, ref, floor):
    """|diff| / max(|ref|, floor), elementwise; floor is a per-component
    scale (broadcastable) declared alongside the result."""
    diff, ref, floor = np.broadcast_arrays(diff, ref, floor)
    return np.abs(diff) / np.maximum(np.abs(ref), np.maximum(floor, 0.0))


def component_floors(stored):
    """Per-component scale floor = median |stored| over the audited rows."""
    return np.median(np.abs(stored), axis=0)


def stratified_rows_by_radius(eta, n):
    """Row indices spanning the full radius range: evenly spaced in the
    radius ranking (covers inner/outer edges, exactly n rows).  The returned
    indices are in ASCENDING radius order."""
    r = np.linalg.norm(eta[:, :3], axis=1)
    order = np.argsort(r, kind="stable")
    picks = np.linspace(0, len(order) - 1, n).round().astype(int)
    return order[picks]


def stratified_subset_indices(n_total, n_pick):
    """Evenly spaced indices spanning [0, n_total-1]: a radius-stratified
    subset of a radius-ordered row list.  NOT a prefix — the first version
    of this audit sliced [:n] off the radius-sorted rows, so the strict-ODE
    and finite-difference checks only certified the inner ~1-3 kpc
    (corrected 2026-09-15 after independent review)."""
    return np.linspace(0, n_total - 1, n_pick).round().astype(int)


def cbe_residual(p, dlnf_dq, dlnf_dp, dphi_dq, eps=CBE_EPS):
    """R = p.d_q lnF - (d_q phi).(d_p lnF) plus normalized form.
    All inputs (..., 3); returns (term1, term2, R, relR)."""
    term1 = np.sum(p * dlnf_dq, axis=-1)
    term2 = np.sum(dphi_dq * dlnf_dp, axis=-1)
    r = term1 - term2
    rel = np.abs(r) / (np.abs(term1) + np.abs(term2) + eps)
    return term1, term2, r, rel


def local_force_svd(A, b, rcond=RCOND_SVD):
    """SVD diagnostics of the local force constraint A g ~ b, with
    A[j] = d_p lnF(q, p_j) and b[j] = p_j . d_q lnF(q, p_j).

    Returns dict with singular values, effective rank, condition number,
    relative least-squares residual ||Ag-b||/||b|| and g_hat (or None when
    rank-deficient: no stable force estimate is emitted)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    rank = int(np.sum(s > rcond * s[0])) if s[0] > 0 else 0
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    g, _res, _rank, _s = np.linalg.lstsq(A, b, rcond=rcond)
    resid_rel = float(np.linalg.norm(A @ g - b)
                      / max(np.linalg.norm(b), 1e-300))
    return {
        "sigma": s,
        "rank": rank,
        "cond": cond,
        "resid_rel": resid_rel,
        "g_hat": g if rank == A.shape[1] else None,
        "b_norm": float(np.linalg.norm(b)),
    }


def central_fd_grad(fn_scalar, x, h):
    """Central-difference gradient of a scalar function at vector x."""
    x = np.asarray(x)
    g = np.zeros_like(x)
    for d in range(x.size):
        e = np.zeros_like(x)
        e[d] = h
        g[d] = (fn_scalar(x + e) - fn_scalar(x - e)) / (2.0 * h)
    return g


def fd_scan(fn_scalar, rows, ad_grads, floors):
    """FD-vs-autodiff error per step size; returns (err_per_h_per_comp, best).

    err(h)[c] = median_i |fd_i,c - ad_i,c| / max(|ad_i,c|, floor_c)."""
    n_h = len(FD_STEPS)
    n, n_c = ad_grads.shape
    err = np.full((n_h, n_c), np.nan)
    for hi, h in enumerate(FD_STEPS):
        abs_err = np.empty((n, n_c))
        for i in range(n):
            abs_err[i] = np.abs(
                central_fd_grad(fn_scalar, rows[i], h) - ad_grads[i])
        rel = abs_err / np.maximum(np.abs(ad_grads), floors)
        err[hi] = np.median(rel, axis=0)
    best_h = FD_STEPS[int(np.argmin(err.max(axis=1)))]
    return err, float(best_h)


def group_summary(per_group):
    """per_group: {'spatial': arr, 'velocity': arr} -> flat summary dict."""
    out = {}
    for g, a in per_group.items():
        a = np.asarray(a)
        out[f"{g}_median"] = float(np.median(a))
        out[f"{g}_p99"] = float(np.percentile(a, 99))
        out[f"{g}_max"] = float(np.max(a))
    return out


# --------------------------------------------------------------------------
# model loading (production code path, checkpoint indices frozen at audit)
# --------------------------------------------------------------------------

def _load_models(run_name):
    """Load DF (flow-21) and Phi (potential-10) with x64 OFF (f32
    checkpoints refuse to deserialise into f64 trees), then return them for
    the caller to evaluate in either regime."""
    import jax
    import fit_all
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        flow = fit_all.load_flow(RUNS[run_name] / "models" / "df" / "flow",
                                 checkpoint_index=DF_CKPT_INDEX)
        phi = fit_all.load_potential(RUNS[run_name] / "models" / "Phi",
                                     checkpoint_index=PHI_CKPT_INDEX)
    finally:
        jax.config.update("jax_enable_x64", prev)
    return flow, phi


def _with_ode_tolerance(flow, rtol, atol):
    """Same DF weights, stricter diffrax PIDController on both sub-flows.

    The controller is a static field of the eqx.VectorField bijection, so
    eqx.tree_at refuses to graft a replacement (its float attributes are
    not pytree leaves).  A deepcopy of the flow is mutated in place via
    object.__setattr__ instead; the caller's original flow is untouched."""
    import copy
    import diffrax

    strict = diffrax.PIDController(rtol=rtol, atol=atol)
    flow2 = copy.deepcopy(flow)
    for sub in (flow2.spatial_flow, flow2.conditional_velocity_flow):
        vf = sub.flow.bijection[0]
        object.__setattr__(vf, "stepsize_controller", strict)
    return flow2


def _x64(on):
    import jax
    import contextlib

    prev = bool(jax.config.jax_enable_x64)

    class _Ctx:
        def __enter__(self):
            jax.config.update("jax_enable_x64", on)
            return self

        def __exit__(self, *exc):
            jax.config.update("jax_enable_x64", prev)

    return _Ctx()


def _grad_lnf_fn(flow):
    """vmapped (lnF, d lnF / d eta) exactly like production
    (flow_sampling.value_and_grad_lnf_fn), plus jit for CPU speed."""
    import jax
    import equinox as eqx
    return jax.jit(jax.vmap(eqx.filter_value_and_grad(flow.log_prob)))


def _lnf_fn_scalar(flow):
    import jax

    def fn(x):
        return flow.log_prob(x)
    # plain closure (id-hashable): jax.jit on the bound method itself fails
    # because jit's cache hashes the eqx module, which contains arrays
    return jax.jit(fn)


def _phi_grad_fn(phi_model):
    """vmapped (d_q phi, lap phi) via the production implementation."""
    import jax
    from potential import calc_phi_derivatives
    vf = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))

    def fn(q):
        return vf(phi_model, q)
    return jax.jit(fn)


def _sample_velocities(flow, q_row, m, key):
    """m velocities from p(v | q) at one fixed position, f32 regime
    (matches production sampling)."""
    import jax
    import jax.numpy as jnp
    cond = jnp.tile(jnp.asarray(q_row, dtype=jnp.float32), (m, 1))
    return np.asarray(flow.conditional_velocity_flow.sample(key, m, condition=cond))


def eval_batched(fn, arr, batch=BATCH):
    """Apply a vmapped fn in batches; fn must return a tuple of arrays with
    leading batch axis."""
    outs = []
    for i in range(0, len(arr), batch):
        outs.append(fn(arr[i:i + batch]))
    return tuple(
        np.concatenate([np.asarray(o[k]) for o in outs], axis=0)
        for k in range(len(outs[0])))


def _provenance(run_names):
    """Checkpoint paths + sha256 re-hashed now and cross-checked against the
    frozen step-0 manifest (fails loudly if a file changed)."""
    mpath = DEFAULT_AUDIT_DIR / "step0" / "manifest.json"
    frozen = json.loads(mpath.read_text())
    prov = {}
    for name in run_names:
        for kind in ("df_checkpoint", "phi_checkpoint"):
            path = Path(frozen["runs"][name][kind]["path"])
            sha = sha256_file(path)
            if sha != frozen["runs"][name][kind]["sha256"]:
                raise RuntimeError(
                    f"{path} changed since step 0 ({sha[:12]} != "
                    f"{frozen['runs'][name][kind]['sha256'][:12]})")
            prov[f"{name}.{kind}"] = {"path": str(path), "sha256": sha}
    return prov


def _audit_region():
    """CBE/force radii anchor: first truth outer edge >= 2 kpc."""
    import h5py
    with h5py.File(TRUTH_PATH, "r") as f:
        edges = f["r_edges"][:]
    a2 = float(edges[edges >= A2_KPC_TARGET][0])
    if abs(a2 - 2.128006) > 1e-3:
        raise RuntimeError(f"unexpected truth anchor edge {a2}")
    return a2


# --------------------------------------------------------------------------
# step 6.1-6.3: score audit (stored vs autodiff, ODE tolerance, FD)
# --------------------------------------------------------------------------

def score_audit(audit_dir, smoke=False):
    import h5py
    import jax
    import jax.numpy as jnp

    out_dir = Path(audit_dir) / "step6"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_rows = 256 if smoke else SCORE_N_ROWS
    n_strict = 128 if smoke else SCORE_N_STRICT
    n_fd = 8 if smoke else SCORE_N_FD

    prov = _provenance(RUNS.keys())

    per_run, npz = {}, {}
    for run_name in RUNS:
        print(f"[step6.score] {run_name}: loading stored gradients ...")
        with h5py.File(RUNS[run_name] / "data" / "df_gradients.h5", "r") as f:
            eta_all = f["eta"][:]
            lnf_stored_all = f["lnf"][:]
            dlnf_stored_all = f["dlnf_deta"][:]
        rows = stratified_rows_by_radius(eta_all, n_rows)
        eta = eta_all[rows]
        lnf_st = lnf_stored_all[rows]
        dlnf_st = dlnf_stored_all[rows]

        # radius-stratified subsets spanning the FULL support (independent
        # review 2026-09-15: the previous [:n] prefix slices certified only
        # the inner 1-3 kpc)
        strict_idx = stratified_subset_indices(n_rows, n_strict)
        fd_idx = stratified_subset_indices(n_rows, n_fd)

        flow, _phi = _load_models(run_name)

        # regime f32 (production-matching arithmetic; production device was
        # GPU, this is CPU, so device-level f32 nondeterminism remains inside
        # this comparison by construction)
        with _x64(False):
            fn32 = _grad_lnf_fn(flow)
            lnf_32, dlnf_32 = eval_batched(fn32, jnp.asarray(eta, dtype=jnp.float32))

        # regime x64 (f64 inputs x f32 weights)
        with _x64(True):
            fn64 = _grad_lnf_fn(flow)
            lnf_64, dlnf_64 = eval_batched(fn64, jnp.asarray(eta, dtype=jnp.float64))

        # strict ODE tolerance on the stratified subset (x64)
        flow_strict = _with_ode_tolerance(flow, ODE_STRICT_RTOL, ODE_STRICT_ATOL)
        with _x64(True):
            fns = _grad_lnf_fn(flow_strict)
            eta_s = jnp.asarray(eta[strict_idx], dtype=jnp.float64)
            _lnf_s, dlnf_strict = eval_batched(fns, eta_s)

        # FD scan on the stratified subset (x64); autodiff reference on the
        # same subset
        with _x64(True):
            scalar = _lnf_fn_scalar(flow)
            fd_err, fd_best_h = fd_scan(
                scalar, np.asarray(eta[fd_idx], dtype=np.float64),
                np.asarray(dlnf_64[fd_idx]), component_floors(dlnf_st))

        floors = component_floors(dlnf_st)
        cmp32 = rel_with_floor(dlnf_32 - dlnf_st, dlnf_st, floors)
        cmp64 = rel_with_floor(dlnf_64 - dlnf_st, dlnf_st, floors)
        # decomposition: CPU arithmetic precision alone (f32 vs x64, same
        # device, same inputs) vs the stored-GPU value
        cmp_arith = rel_with_floor(dlnf_32 - dlnf_64, dlnf_64,
                                   component_floors(dlnf_64))
        cmp_ode = rel_with_floor(
            dlnf_strict - dlnf_64[strict_idx], dlnf_64[strict_idx],
            component_floors(dlnf_64[strict_idx]))
        abs64 = {"spatial": np.abs(dlnf_64[:, :3] - dlnf_st[:, :3]),
                 "velocity": np.abs(dlnf_64[:, 3:] - dlnf_st[:, 3:])}
        lnf_abs64 = np.abs(lnf_64 - lnf_st)
        lnf_rel64 = lnf_abs64 / max(np.median(np.abs(lnf_st)), 1e-30)

        s32 = group_summary({"spatial": cmp32[:, :3], "velocity": cmp32[:, 3:]})
        s64 = group_summary({"spatial": cmp64[:, :3], "velocity": cmp64[:, 3:]})
        sar = group_summary({"spatial": cmp_arith[:, :3],
                             "velocity": cmp_arith[:, 3:]})
        s64_abs = {f"{g}_median": float(np.median(a))
                   for g, a in abs64.items()}
        r_kpc_rows = np.linalg.norm(eta[:, :3], axis=1) * L_KPC
        sode = group_summary({"spatial": cmp_ode[:, :3], "velocity": cmp_ode[:, 3:]})
        gate_ok = (max(s64["spatial_median"], s64["velocity_median"]) < TH_SCORE_MEDIAN
                   and max(s64["spatial_p99"], s64["velocity_p99"]) < TH_SCORE_P99)
        ode_ok = (max(sode["spatial_median"], sode["velocity_median"])
                  < TH_ODE_MEDIAN)
        fd_ok = bool(np.nanmax(np.nanmin(fd_err, axis=0)) < TH_FD_PLATEAU)

        per_run[run_name] = {
            "n_rows": int(n_rows),
            "rows_radius_range_norm": [
                float(np.min(np.linalg.norm(eta[:, :3], axis=1))),
                float(np.max(np.linalg.norm(eta[:, :3], axis=1)))],
            "strict_subset": {
                "n_rows": int(n_strict),
                "radius_range_kpc": [float(r_kpc_rows[strict_idx].min()),
                                     float(r_kpc_rows[strict_idx].max())]},
            "fd_subset": {
                "n_points": int(n_fd),
                "radius_range_kpc": [float(r_kpc_rows[fd_idx].min()),
                                     float(r_kpc_rows[fd_idx].max())]},
            "component_floors_median_abs_stored": finite_or_fail(
                floors, "floors").tolist(),
            "stored_vs_autodiff_f32_cpu": s32,
            "stored_vs_autodiff_x64": s64,
            "stored_vs_autodiff_x64_absolute": s64_abs,
            "f32cpu_vs_x64cpu_arithmetic": sar,
            "ode_strict_vs_default_x64": sode,
            "ode_strict_n_rows": int(n_strict),
            "lnf_x64_abs_err_median": float(np.median(lnf_abs64)),
            "lnf_x64_abs_err_p99": float(np.percentile(lnf_abs64, 99)),
            "lnf_x64_rel_err_median_floored": float(np.median(lnf_rel64)),
            "fd_scan": {
                "n_points": int(n_fd),
                "steps": FD_STEPS.tolist(),
                "median_rel_err_per_step_per_comp": finite_or_fail(
                    fd_err, "fd_err").tolist(),
                "best_step": fd_best_h,
                "plateau_max_median_rel": float(np.nanmax(
                    np.nanmin(fd_err, axis=0))),
            },
            "gates": {
                "score_chain_ok": bool(gate_ok),
                "ode_tolerance_ok": bool(ode_ok),
                "fd_plateau_ok": fd_ok,
            },
        }
        npz[f"{run_name}__eta"] = eta
        npz[f"{run_name}__lnf_stored"] = lnf_st
        npz[f"{run_name}__dlnf_stored"] = dlnf_st
        npz[f"{run_name}__dlnf_ad_f32"] = dlnf_32
        npz[f"{run_name}__dlnf_ad_x64"] = dlnf_64
        dlnf_strict_padded = np.full((n_rows, 6), np.nan)
        dlnf_strict_padded[strict_idx] = dlnf_strict
        npz[f"{run_name}__dlnf_ad_strict"] = dlnf_strict_padded
        npz[f"{run_name}__strict_idx"] = strict_idx
        npz[f"{run_name}__fd_idx"] = fd_idx
        npz[f"{run_name}__stored_row_indices"] = rows
        npz[f"{run_name}__fd_err"] = fd_err
        print(f"[step6.score] {run_name}: x64 vs stored "
              f"median/p99 spatial {s64['spatial_median']:.2e}/"
              f"{s64['spatial_p99']:.2e}, velocity {s64['velocity_median']:.2e}/"
              f"{s64['velocity_p99']:.2e}; gates {per_run[run_name]['gates']}")

    result = {
        "step": "6.1-6.3 score audit",
        "definitions": {
            "stored": "runs/<run>/data/df_gradients.h5 (f32, GPU, Tsit5, "
                      "PIDController rtol=1e-4 atol=1e-5)",
            "autodiff": "eqx.filter_value_and_grad(flow.log_prob) on the "
                        "frozen flow-21 checkpoint",
            "rel_err": "|diff| / max(|stored|, floor_c), floor_c = per-comp "
                       "median |stored| over the audited rows",
            "f32_pass": "CPU f32, same arithmetic class as production "
                        "(device difference GPU-vs-CPU included)",
            "x64_pass": "CPU f64 inputs x f32 weights (higher-accuracy "
                        "reference)",
            "ode_strict": f"PIDController rtol={ODE_STRICT_RTOL} "
                          f"atol={ODE_STRICT_ATOL} on both sub-flows",
            "fd": "central differences of lnF in x64, steps "
                  f"{FD_STEPS.tolist()}",
        },
        "thresholds": {
            "score_chain_ok": f"x64-vs-stored floored rel: median < "
                              f"{TH_SCORE_MEDIAN} and p99 < {TH_SCORE_P99} "
                              "in both groups",
            "ode_tolerance_ok": f"strict-vs-default median < {TH_ODE_MEDIAN}",
            "fd_plateau_ok": f"per-comp plateau (best h) median rel < "
                             f"{TH_FD_PLATEAU}",
        },
        "provenance": prov,
        "environment": {"jax_devices": [str(d) for d in jax.devices()]},
        "smoke": bool(smoke),
        "elapsed_s": time.time() - t0,
        "per_run": per_run,
    }
    _dump_json(out_dir / "score_audit.json", result)
    np.savez_compressed(out_dir / "score_audit.npz", **npz)
    print(f"[step6.score] done -> {out_dir / 'score_audit.json'} "
          f"({result['elapsed_s']:.1f} s)")
    return result


# --------------------------------------------------------------------------
# step 6.4: CBE residual on stratified fresh points + stored-point comparison
# --------------------------------------------------------------------------

def cbe_residual_audit(audit_dir, run_name="baseline", smoke=False):
    import h5py
    import jax
    import jax.numpy as jnp

    out_dir = Path(audit_dir) / "step6"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prov = _provenance([run_name])
    a2 = _audit_region()
    n_radii = 8 if smoke else CBE_N_RADII
    n_vel = 1 if smoke else CBE_N_VEL

    radii = np.geomspace(a2, ROUT_KPC, n_radii)
    dirs = sobol_directions(CBE_N_DIRS, scramble_seed=SEED_DIRS_CBE)

    flow, phi = _load_models(run_name)

    # positions: (r, dir) grid; velocities from p(v|q) at each position
    qs = [(r_kpc / L_KPC) * d for r_kpc in radii for d in dirs]
    key = jax.random.key(SEED_VEL)
    eta_rows, r_rows = [], []
    with _x64(False):  # f32 sampling, production-matching
        for q in qs:
            key, sub = jax.random.split(key)
            vs = _sample_velocities(flow, q, n_vel, sub)
            for v in vs:
                eta_rows.append(np.concatenate([q, v]))
                r_rows.append(np.linalg.norm(q) * L_KPC)
    eta = np.array(eta_rows, dtype=np.float64)
    r_rows = np.array(r_rows)

    # x64 evaluation: full-DF gradients + spatial-density log-nu + Phi grads
    def _log_nu(x):
        return flow.log_prob_position(x)

    with _x64(True):
        fn = _grad_lnf_fn(flow)
        lnf, dlnf = eval_batched(fn, jnp.asarray(eta))
        nu_fn = jax.jit(jax.vmap(_log_nu))
        ln_nu = np.concatenate(
            [np.asarray(nu_fn(jnp.asarray(eta[i:i + 1024, :3])))
             for i in range(0, len(eta), 1024)], axis=0)
        pg = _phi_grad_fn(phi.phi_model)
        dphi, _lap = eval_batched(pg, jnp.asarray(eta[:, :3]))

    term1, term2, resid, rel = cbe_residual(
        eta[:, 3:], dlnf[:, :3], dlnf[:, 3:], dphi)
    finite_or_fail(term1, "term1"); finite_or_fail(term2, "term2")
    finite_or_fail(resid, "R"); finite_or_fail(rel, "relR")

    # DF-expectation importance weights for this design.  The sampling
    # measure is uniform in log r x uniform on the sphere x p(v|q), i.e.
    # density per unit volume h(q) ~ r^-3; so the weight toward the DF
    # probability measure is w ~ F/h ~ r^3 nu(q) — the p(v|q) factor is
    # already accounted for by the conditional velocity sampling.  (The
    # first version of this audit weighted by exp(lnf) directly, which
    # double-counted high-probability velocities and dropped the r^3
    # volume factor; corrected 2026-09-15 after independent review.)
    r_norm = np.linalg.norm(eta[:, :3], axis=1)
    ln_w = 3.0 * np.log(r_norm) + ln_nu
    w = np.exp(ln_w - np.max(ln_w))
    w = w / np.sum(w)
    weight_ess = float(np.sum(w) ** 2 / np.sum(w ** 2))
    order = np.argsort(np.abs(resid))      # weighted median of |R|
    r_cw = np.cumsum(w[order])
    abs_r_wmedian = float(np.abs(resid)[order][np.searchsorted(r_cw, 0.5)])
    binned = []
    for i, r_kpc in enumerate(radii):
        sl = slice(i * CBE_N_DIRS * n_vel, (i + 1) * CBE_N_DIRS * n_vel)
        binned.append({
            "r_kpc": float(r_kpc),
            "abs_R_median": float(np.median(np.abs(resid[sl]))),
            "abs_R_p84": float(np.percentile(np.abs(resid[sl]), 84)),
            "relR_median": float(np.median(rel[sl])),
            "relR_p84": float(np.percentile(rel[sl], 84)),
            "abs_term1_median": float(np.median(np.abs(term1[sl]))),
            "abs_term2_median": float(np.median(np.abs(term2[sl]))),
            "lnf_median": float(np.median(lnf[sl])),
            "df_weight_fraction": float(np.sum(w[sl])),
        })

    # stored-training-point comparison: R seen by the trainer (stored scores)
    # vs R with fresh x64 scores at the same points
    n_stored = 128 if smoke else CBE_N_STORED_ROWS
    with h5py.File(RUNS[run_name] / "data" / "df_gradients.h5", "r") as f:
        eta_all = f["eta"][:]; dlnf_all = f["dlnf_deta"][:]
    srows = stratified_rows_by_radius(eta_all, n_stored)
    eta_s = eta_all[srows]; dlnf_s = dlnf_all[srows]
    with _x64(True):
        fn = _grad_lnf_fn(flow)
        _l, dlnf_fresh = eval_batched(fn, jnp.asarray(eta_s, dtype=np.float64))
        pg = _phi_grad_fn(phi.phi_model)
        dphi_s, _ = eval_batched(pg, jnp.asarray(eta_s[:, :3], dtype=np.float64))
    _t1s, _t2s, r_stored, _ = cbe_residual(
        eta_s[:, 3:], dlnf_s[:, :3], dlnf_s[:, 3:], dphi_s)
    _t1f, _t2f, r_fresh, _ = cbe_residual(
        eta_s[:, 3:], dlnf_fresh[:, :3], dlnf_fresh[:, 3:], dphi_s)
    d_r = np.abs(r_stored - r_fresh)
    stored_cmp = {
        "n_rows": int(n_stored),
        "radius_range_kpc": [float(np.min(np.linalg.norm(eta_s[:, :3], axis=1) * L_KPC)),
                             float(np.max(np.linalg.norm(eta_s[:, :3], axis=1) * L_KPC))],
        "abs_R_stored_median": float(np.median(np.abs(r_stored))),
        "abs_R_fresh_median": float(np.median(np.abs(r_fresh))),
        "abs_R_stored_minus_fresh_median": float(np.median(d_r)),
        "abs_R_stored_minus_fresh_p99": float(np.percentile(d_r, 99)),
        "delta_over_R_median": float(np.median(d_r / np.maximum(np.abs(r_fresh), CBE_EPS))),
    }

    np.savez_compressed(
        out_dir / "cbe_diagnostics.npz",
        eta=eta, r_kpc=r_rows, lnf=lnf, ln_nu=ln_nu, dlnf=dlnf, dphi=dphi,
        term1=term1, term2=term2, R=resid, relR=rel,
        stored_eta=eta_s, stored_R=r_stored, stored_R_fresh=r_fresh,
        stored_row_indices=srows)

    result = {
        "step": "6.4 CBE residual",
        "definitions": {
            "R": "p . d_q lnF - (d_q phi) . d_p lnF, code units, full DF "
                 "(spatial + conditional), Phi = potential-10",
            "relR": f"|R| / (|term1| + |term2| + {CBE_EPS})",
            "epsilon_justification": "pre-declared round scale 1.0 from "
                                     "stored-data statistics (median |term1| "
                                     "8.17, p16-p84 1.5-24.9)",
            "df_expectation_weights": "w ~ r^3 * nu(q) (importance weights "
                                      "from the log-r x uniform-angle x "
                                      "p(v|q) sampling measure h ~ r^-3 per "
                                      "unit volume; corrected 2026-09-15, "
                                      "was exp(lnf) before)",
            "sampling": f"{n_radii} log radii in [{a2:.6f}, {ROUT_KPC}] kpc x "
                        f"{CBE_N_DIRS} Sobol dirs (seed {SEED_DIRS_CBE}) x "
                        f"{n_vel} velocities from p(v|q) (seed {SEED_VEL})",
            "regime": "x64 host x f32 weights, CPU",
        },
        "provenance": prov,
        "smoke": bool(smoke),
        "n_points": int(len(eta)),
        "overall": {
            "abs_R_median": float(np.median(np.abs(resid))),
            "abs_R_p84": float(np.percentile(np.abs(resid), 84)),
            "abs_R_df_expectation_median": abs_r_wmedian,
            "df_expectation_weight_ess": weight_ess,
            "relR_median": float(np.median(rel)),
            "relR_p84": float(np.percentile(rel, 84)),
            "abs_term1_median": float(np.median(np.abs(term1))),
            "abs_term2_median": float(np.median(np.abs(term2))),
            "df_weight_fraction_inside_30kpc": float(np.sum(w[r_rows < 30.0])),
        },
        "binned_by_radius": binned,
        "stored_training_points": stored_cmp,
        "elapsed_s": time.time() - t0,
    }
    _dump_json(out_dir / "cbe_diagnostics.json", result)
    print(f"[step6.cbe] {run_name}: |R| median {result['overall']['abs_R_median']:.3f} "
          f"(term1 med {result['overall']['abs_term1_median']:.3f}, term2 med "
          f"{result['overall']['abs_term2_median']:.3f}), relR median "
          f"{result['overall']['relR_median']:.4f}; stored-vs-fresh dR median "
          f"{stored_cmp['abs_R_stored_minus_fresh_median']:.4e}")
    print(f"[step6.cbe] done -> {out_dir / 'cbe_diagnostics.json'} "
          f"({result['elapsed_s']:.1f} s)")
    return result


# --------------------------------------------------------------------------
# step 6.5-6.6: SVD local force constraints + comparison with Phi gradient
# --------------------------------------------------------------------------

def local_force_audit(audit_dir, run_name="baseline", smoke=False):
    import jax
    import jax.numpy as jnp

    out_dir = Path(audit_dir) / "step6"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prov = _provenance([run_name])
    a2 = _audit_region()
    n_radii = 4 if smoke else FORCE_N_RADII
    n_dirs = 2 if smoke else FORCE_N_DIRS
    m_tiers = (8, 32) if smoke else FORCE_M_TIERS

    radii = np.geomspace(a2, ROUT_KPC, n_radii)
    dirs = sobol_directions(n_dirs, scramble_seed=SEED_DIRS_FORCE)
    positions = [(r_kpc / L_KPC) * d for r_kpc in radii for d in dirs]
    # nominal design radii (exact grid values; norm(q)*L would only
    # re-approximate them and split the grid under float rounding)
    pos_r = np.array([r_kpc for r_kpc in radii for _ in dirs])

    flow, phi = _load_models(run_name)

    per_tier, npz = {}, {"positions": np.array(positions), "pos_r_kpc": pos_r}
    eta_last = dlnf_last = None
    m_last = None
    for tier_i, m in enumerate(m_tiers):
        As, bs, diags, g_phis = [], [], [], []
        key = jax.random.key(SEED_VEL + 100 + tier_i)
        with _x64(False):  # f32 conditional sampling
            vel_blocks = []
            for q in positions:
                key, sub = jax.random.split(key)
                vel_blocks.append(_sample_velocities(flow, q, m, sub))
        eta = np.concatenate(
            [np.concatenate([np.tile(q, (m, 1)), v], axis=1)
             for q, v in zip(positions, vel_blocks)], axis=0).astype(np.float64)
        with _x64(True):
            fn = _grad_lnf_fn(flow)
            _lnf, dlnf = eval_batched(fn, jnp.asarray(eta))
            pg = _phi_grad_fn(phi.phi_model)
            dphi_pos, _ = eval_batched(
                pg, jnp.asarray(np.array(positions), dtype=np.float64))
        finite_or_fail(dlnf, "dlnf")
        for pi in range(len(positions)):
            sl = slice(pi * m, (pi + 1) * m)
            A = dlnf[sl, 3:]
            b = np.sum(eta[sl, 3:] * dlnf[sl, :3], axis=1)
            d = local_force_svd(A, b)
            d["g_phi"] = dphi_pos[pi]
            # same-metric comparison of the current Phi with the local LS
            # optimum (independent review 2026-09-15: the earlier |g_hat-g_phi|
            # vs resid_rel comparison mixed different spaces/denominators)
            b_norm = max(np.linalg.norm(b), 1e-300)
            d["res_phi"] = float(
                np.linalg.norm(A @ dphi_pos[pi] - b) / b_norm)
            if d["g_hat"] is not None:
                d["extra_res"] = float(np.linalg.norm(
                    A @ (dphi_pos[pi] - d["g_hat"])) / b_norm)
                d["eliminable_sq_share"] = float(
                    1.0 - (d["resid_rel"] / d["res_phi"]) ** 2)
            As.append(A.astype(np.float32)); bs.append(b.astype(np.float64))
            diags.append(d); g_phis.append(dphi_pos[pi])
        eta_last, dlnf_last, m_last = eta, dlnf, m

        well = [d for d in diags if d["rank"] == 3 and d["cond"] < COND_MAX]
        well_mask = np.array(
            [d["rank"] == 3 and d["cond"] < COND_MAX for d in diags])
        g_rel_all = np.full(len(diags), np.nan)
        for i in np.where(well_mask)[0]:
            d = diags[i]
            g_rel_all[i] = (np.linalg.norm(d["g_hat"] - d["g_phi"])
                            / np.linalg.norm(d["g_phi"]))
        g_rel = g_rel_all[well_mask]
        outer = pos_r > 60.0
        res_phi_arr = np.array([d["res_phi"] for d in diags])
        elim = np.array([d.get("eliminable_sq_share", np.nan)
                         for d in diags])
        per_tier[f"M{m}"] = {
            "n_positions": len(positions),
            "n_velocities": int(m),
            "rank_distribution": {
                str(k): int(sum(1 for d in diags if d["rank"] == k))
                for k in sorted({d["rank"] for d in diags})},
            "cond_median": float(np.median([d["cond"] for d in diags])),
            "cond_p84": float(np.percentile([d["cond"] for d in diags], 84)),
            "resid_rel_median": float(np.median([d["resid_rel"] for d in diags])),
            "resid_rel_p84": float(np.percentile([d["resid_rel"] for d in diags], 84)),
            "res_phi_median": float(np.median(res_phi_arr)),
            "res_phi_p84": float(np.percentile(res_phi_arr, 84)),
            "extra_res_median": float(np.median(
                [d["extra_res"] for d in diags if "extra_res" in d])),
            "eliminable_sq_share_median": float(np.nanmedian(elim)),
            "eliminable_sq_share_p84": float(np.nanpercentile(elim, 84)),
            "n_eliminating_gt25pct": int(np.sum(elim > 0.25)),
            "n_well_conditioned": len(well),
            "g_vs_phi_rel_median": (float(np.median(g_rel)) if len(g_rel) else None),
            "g_vs_phi_rel_p84": (float(np.percentile(g_rel, 84)) if len(g_rel) else None),
            "g_vs_phi_rel_max": (float(np.max(g_rel)) if len(g_rel) else None),
            "g_vs_phi_rel_median_outer_r_gt_60": (
                float(np.nanmedian(g_rel_all[outer]))
                if outer.any() and np.any(np.isfinite(g_rel_all[outer]))
                else None),
        }
        npz[f"M{m}__A"] = np.array(As)
        npz[f"M{m}__b"] = np.array(bs)
        npz[f"M{m}__sigma"] = np.array([d["sigma"] for d in diags])
        npz[f"M{m}__cond"] = np.array([d["cond"] for d in diags])
        npz[f"M{m}__rank"] = np.array([d["rank"] for d in diags], dtype=int)
        npz[f"M{m}__resid_rel"] = np.array([d["resid_rel"] for d in diags])
        npz[f"M{m}__res_phi"] = res_phi_arr
        npz[f"M{m}__g_hat"] = np.array(
            [d["g_hat"] if d["g_hat"] is not None else np.full(3, np.nan)
             for d in diags])
        npz[f"M{m}__g_phi"] = np.array(g_phis)
        print(f"[step6.force] M={m}: ranks {per_tier[f'M{m}']['rank_distribution']}, "
              f"resid_rel median {per_tier[f'M{m}']['resid_rel_median']:.4f}, "
              f"res_phi median {per_tier[f'M{m}']['res_phi_median']:.4f}, "
              f"eliminable median {100*per_tier[f'M{m}']['eliminable_sq_share_median']:.1f}%, "
              f"g-vs-phi rel median {per_tier[f'M{m}']['g_vs_phi_rel_median']}")

    # cross-tier stability of the force estimate (independent draws)
    g32 = npz[f"M{m_tiers[0]}__g_hat"]; g128 = npz[f"M{m_tiers[1]}__g_hat"]
    ok = ~(np.isnan(g32).any(axis=1) | np.isnan(g128).any(axis=1))
    stab = np.linalg.norm(g32[ok] - g128[ok], axis=1) / np.maximum(
        np.linalg.norm(g128[ok], axis=1), 1e-30)
    stability = {
        "tiers": [int(m) for m in m_tiers],
        "n_comparable_positions": int(ok.sum()),
        "g_rel_change_median": float(np.median(stab)) if ok.any() else None,
        "g_rel_change_p84": float(np.percentile(stab, 84)) if ok.any() else None,
    }

    # independent-draw cross evaluation: the force fitted on the tier-0
    # velocities evaluated on the tier-1 velocities at the same positions,
    # compared with the current Phi on the same tier-1 constraints
    # (same metric ||A g - b|| / ||b|| throughout)
    A1 = npz[f"M{m_tiers[1]}__A"].astype(np.float64)
    b1 = npz[f"M{m_tiers[1]}__b"].astype(np.float64)
    cross_res, cross_better = [], 0
    for i in range(len(positions)):
        bn = max(np.linalg.norm(b1[i]), 1e-300)
        rc = float(np.linalg.norm(A1[i] @ g32[i] - b1[i]) / bn)
        cross_res.append(rc)
        if rc < npz[f"M{m_tiers[1]}__res_phi"][i]:
            cross_better += 1
    cross_res = np.array(cross_res)
    res_phi_1 = npz[f"M{m_tiers[1]}__res_phi"]
    cross_eval = {
        "fit_tier_n_velocities": int(m_tiers[0]),
        "eval_tier_n_velocities": int(m_tiers[1]),
        "resid_rel_median": float(np.median(cross_res)),
        "res_phi_median": float(np.median(res_phi_1)),
        "n_positions_better_than_phi": int(cross_better),
        "sq_residual_improvement_vs_phi_median": float(
            np.median(1.0 - (cross_res / res_phi_1) ** 2)),
    }

    # strict-ODE scores at the SAME (q, p) as the last tier: does the ODE
    # tolerance noise materially change A, b, the minimum residual, or the
    # force estimate?  (closes the review gap that precision checks never
    # reached the outer region / the local-constraint points)
    flow_strict = _with_ode_tolerance(flow, ODE_STRICT_RTOL, ODE_STRICT_ATOL)
    with _x64(True):
        fns = _grad_lnf_fn(flow_strict)
        _l_strict, dlnf_strict = eval_batched(fns, jnp.asarray(eta_last))
    m = m_last
    score_rel, res_min_strict, g_rel_change, res_phi_strict = [], [], [], []
    g_hat_strict = np.full((len(positions), 3), np.nan)
    floors_all = component_floors(dlnf_last)   # (6,) per-component floor
    for pi in range(len(positions)):
        sl = slice(pi * m, (pi + 1) * m)
        A_s = dlnf_strict[sl, 3:]
        b_s = np.sum(eta_last[sl, 3:] * dlnf_strict[sl, :3], axis=1)
        d_s = local_force_svd(A_s, b_s)
        bn = max(np.linalg.norm(b_s), 1e-300)
        g_phi_i = npz[f"M{m_last}__g_phi"][pi]
        res_min_strict.append(d_s["resid_rel"])
        res_phi_strict.append(float(
            np.linalg.norm(A_s @ g_phi_i - b_s) / bn))
        if d_s["g_hat"] is not None:
            g_hat_strict[pi] = d_s["g_hat"]
            g_ref = npz[f"M{m_last}__g_hat"][pi]
            if np.all(np.isfinite(g_ref)):
                g_rel_change.append(float(
                    np.linalg.norm(d_s["g_hat"] - g_ref)
                    / max(np.linalg.norm(g_ref), 1e-30)))
        score_rel.append(float(np.median(rel_with_floor(
            dlnf_strict[sl] - dlnf_last[sl], dlnf_last[sl],
            floors_all).ravel())))
    res_min_ref = npz[f"M{m_last}__resid_rel"]
    strict_same_points = {
        "n_velocities": int(m_last),
        "score_change_floored_rel_median_of_medians": float(
            np.median(score_rel)),
        "res_min_strict_vs_default_median": float(
            np.median(np.array(res_min_strict) - res_min_ref)),
        "res_min_strict_median": float(np.median(res_min_strict)),
        "res_phi_strict_median": float(np.median(res_phi_strict)),
        "g_hat_rel_change_median": (float(np.median(g_rel_change))
                                    if g_rel_change else None),
        "g_hat_rel_change_p84": (float(np.percentile(g_rel_change, 84))
                                 if g_rel_change else None),
    }
    npz[f"M{m_last}__g_hat_strict"] = g_hat_strict
    npz[f"M{m_last}__resid_rel_strict"] = np.array(res_min_strict)
    npz[f"M{m_last}__res_phi_strict"] = np.array(res_phi_strict)

    np.savez_compressed(out_dir / "local_force_constraints.npz", **npz)
    result = {
        "step": "6.5-6.6 local force constraints",
        "definitions": {
            "A_j": "d_p lnF(q, p_j) (row j), velocities drawn from p(v|q)",
            "b_j": "p_j . d_q lnF(q, p_j)",
            "svd": f"effective rank at sigma > {RCOND_SVD} sigma_max; "
                   "g_hat suppressed when rank < 3",
            "well_conditioned": f"rank 3 and cond < {COND_MAX}",
            "g_vs_phi_rel": "|g_hat - grad phi(q)| / |grad phi(q)| "
                            "(supplementary; the primary same-metric "
                            "comparison is res_phi vs resid_rel)",
            "res_phi": "||A g_phi - b|| / ||b|| — the current Phi under the "
                       "SAME metric as the local LS minimum resid_rel",
            "eliminable_sq_share": "1 - (resid_rel/res_phi)^2 — share of the "
                                   "Phi squared residual a local LS force "
                                   "would remove at the same position",
            "sampling": f"{n_radii} log radii x {n_dirs} Sobol dirs "
                        f"(seed {SEED_DIRS_FORCE}); velocity seeds "
                        f"{SEED_VEL + 100}+tier (independent draws per tier)",
            "regime": "x64 host x f32 weights, CPU; velocities sampled f32",
        },
        "provenance": prov,
        "smoke": bool(smoke),
        "per_tier": per_tier,
        "cross_tier_stability": stability,
        "cross_tier_fit_evaluation": cross_eval,
        "strict_ode_same_points": strict_same_points,
        "elapsed_s": time.time() - t0,
    }
    _dump_json(out_dir / "local_force_constraints.json", result)
    print(f"[step6.force] cross-tier fit-eval: resid {cross_eval['resid_rel_median']:.4f} "
          f"vs Phi {cross_eval['res_phi_median']:.4f}, better at "
          f"{cross_eval['n_positions_better_than_phi']}/{len(positions)}")
    print(f"[step6.force] strict ODE same points: score change median "
          f"{strict_same_points['score_change_floored_rel_median_of_medians']:.2e}, "
          f"res_min change median {strict_same_points['res_min_strict_vs_default_median']:.2e}, "
          f"g_hat change median {strict_same_points['g_hat_rel_change_median']}")
    print(f"[step6.force] done -> {out_dir / 'local_force_constraints.json'} "
          f"({result['elapsed_s']:.1f} s)")
    return result


# --------------------------------------------------------------------------

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("step", choices=["score-audit", "cbe-residual",
                                         "local-force"])
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--run", type=str, default="baseline",
                        choices=list(RUNS),
                        help="run whose DF/Phi/stored gradients are audited "
                             "(score-audit always covers all three runs)")
    parser.add_argument("--smoke", action="store_true",
                        help="256-point protocol (declared in the plan)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing products")
    args = parser.parse_args()

    targets = {"score-audit": ["score_audit.json", "score_audit.npz"],
               "cbe-residual": ["cbe_diagnostics.json", "cbe_diagnostics.npz"],
               "local-force": ["local_force_constraints.json",
                               "local_force_constraints.npz"]}
    out_dir = args.audit_dir / "step6"
    for name in targets[args.step]:
        target = out_dir / name
        if target.exists() and not args.force:
            raise SystemExit(f"{target} exists; use --force to replace it")

    import jax
    print(f"JAX devices: {jax.devices()}")

    if args.step == "score-audit":
        score_audit(args.audit_dir, smoke=args.smoke)
    elif args.step == "cbe-residual":
        cbe_residual_audit(args.audit_dir, run_name=args.run, smoke=args.smoke)
    else:
        local_force_audit(args.audit_dir, run_name=args.run, smoke=args.smoke)


if __name__ == "__main__":
    main()
