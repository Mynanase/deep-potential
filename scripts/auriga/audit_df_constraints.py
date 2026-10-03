#!/usr/bin/env python
"""T1 score numerical-chain audit for the frozen Phase-2 control model.

Numerical audit; historical input identity is recorded in the T0 manifest
docs/history/nf-score-audit/t0-manifest.json.  This is a REWRITE of the 2026-09-15
step-6 auditor: the old run bindings (runs/halo12-baseline, truth anchors),
engineering thresholds and checkpoint indices are deliberately NOT carried
over.  Everything below is re-declared against the control lineage pinned in
the T0 manifest (run 2b32eb04, source 5f76faa, sha256-pinned checkpoints).

Units and conventions (code units throughout, as in AGENTS.md):

  q = x / L,  p = v / V        (eta = [q, p], L = 10 kpc, V = 100 km/s)
  score  s(eta) = grad_eta log F(eta)     -- dlnf_deta in df_gradients.h5
  alpha(q) = - grad_q phi(q)              -- ACCELERATION convention of this
                                             audit line (plan appendix A1);
                                             the old local_force_svd solved
                                             g = +grad_q phi and is replaced.

Numerical budget candidates declared by T0 (verified feasible by the pilot
before adoption; engineering criteria, NOT significance levels):

  TH_CHAIN_MEDIAN / TH_CHAIN_P99 : stored-vs-recompute floored relative
                      error of the six score components, x64-strict ODE
                      reference vs the f32 GPU values stored in
                      df_gradients.h5 (floor = per-component median |stored|
                      over the audited rows).
  TH_ODE_MEDIAN   : default-tolerance vs strict-tolerance ODE recompute
                      (both x64) -- solver-tolerance sensitivity.
  TH_FD_LEVEL     : central-difference vs autodiff plateau level; the gate
                      requires an ADJACENT-STEP stable interval (>= 2
                      consecutive step sizes under the level, with the two
                      FD gradients also agreeing under the level), never a
                      single best step.
  TH_IDENTITY_*   : chain identities (lnf split, p-block equality) between
                      independently evaluated code paths.

Control integrity gates (stop conditions from the T1 card):
  - every sha256 pinned in t0-manifest.json must match the file on disk;
  - the deployed flow-21 checkpoint must carry bitwise the trained spatial
    flow of flow_pos_only-10 (single flow pair, no ensemble);
  - zero non-finite values in any compared score array.

Products (never overwritten without --force):
  <out-dir>/audit_score_chain.json / .npz

Run inside an orchestration snapshot on the GPU host, from the repo root:
  JAX_PLATFORMS=cuda python scripts/auriga/audit_df_constraints.py \
      score-chain --control-repo /home/qiutao/.orx/runs/<run>/repo \
                  --control-run-dir <...>/repo/runs/orx
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

L_KPC, V_KMS = 10.0, 100.0

# ---- T1 engineering gates (re-declared 2026-09-30 from the T0 numerical ----
# ---- budget candidates; not inherited from the 2026-09-15 audit)         ----
TH_CHAIN_MEDIAN = 1e-2     # stored vs x64-strict, per group (spatial/velocity)
TH_CHAIN_P99 = 5e-2
TH_ODE_MEDIAN = 5e-3       # x64 default vs x64 strict ODE tolerance
TH_FD_LEVEL = 5e-2         # FD vs autodiff plateau, adjacent-step stability
TH_IDENTITY_MEDIAN = 1e-4  # independently evaluated chain identities
TH_IDENTITY_P99 = 1e-3
TH_PSCORE_STORED_MEDIAN = 1e-3  # stored dlnf_deta p-block vs dlnp_deta p-block

# FD step ladder, geometric ratio ~1:3.  With x64-strict ODE solves the
# useful window sits between truncation (large h) and the solver noise floor
# (small h); the gate wants an adjacent-step run inside TH_FD_LEVEL.
FD_STEPS = np.array([3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4])

# Production CNF settings (flow_vector_fields.py at 5f76faa): Tsit5 with
# PIDController(rtol=1e-4, atol=1e-5).  Strict reference tolerance below.
ODE_STRICT_RTOL, ODE_STRICT_ATOL = 1e-7, 1e-8

# Pilot sizes (distinct audited points; the strict/FD subsets are subsets
# of the audited rows, so the total distinct-point budget stays at 1024).
AUDIT_N_ROWS = 1024        # radius-stratified rows of df_gradients.h5
AUDIT_N_STRICT = 256       # subset re-solved at strict ODE tolerance (x64)
AUDIT_N_FD = 32            # subset for the finite-difference scan

# Checkpoint indices of the DEPLOYED control pair (T0 manifest: fixed
# 256-epoch cosine schedule, final checkpoint deployed):
#   spatial flow flow_pos_only-10, conditional velocity flow flow-21,
#   potential-10.  These come from the T0 manifest lineage, not from any
# 2026-09-15 run.
FLOW_CKPT_INDEX = 21
SPATIAL_CKPT_INDEX = 10
PHI_CKPT_INDEX = 10

BATCH = 256                # production grad_batch_size (options.json)

# Model-defining files of the control snapshot whose sha256 is recorded in
# every audit/cache manifest (derivative-source lineage).
MODEL_CODE_FILES = [
    "scripts/fit_all.py",
    "scripts/flow_ot_flow_matching_conditional.py",
    "scripts/flow_vector_fields.py",
    "scripts/flow_sampling.py",
    "scripts/flow_matching_conditional.py",
    "scripts/potential.py",
    "scripts/utils.py",
]


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
    subset of a radius-ordered row list.  NOT a prefix (prefix slices were
    the 2026-09-15 review finding: they certified only the inner kpc)."""
    return np.linspace(0, n_total - 1, n_pick).round().astype(int)


def group_summary(per_group):
    """per_group: {'spatial': arr, 'velocity': arr} -> flat summary dict."""
    out = {}
    for g, a in per_group.items():
        a = np.asarray(a)
        out[f"{g}_median"] = float(np.median(a))
        out[f"{g}_p99"] = float(np.percentile(a, 99))
        out[f"{g}_max"] = float(np.max(a))
    return out


def acceleration_from_grad_phi(dphi_dq):
    """alpha = -grad_q phi.  The single sanctioned conversion between the
    potential-gradient convention and the acceleration convention of this
    audit line (plan appendix A1)."""
    return -np.asarray(dphi_dq)


def combine_split_grads(dln_nu_dq, dlnp_deta):
    """Full-eta score from the split evaluation: the q-block is
    d ln n / dq + d ln P / dq (the conditional term carries the
    condition-normalization Jacobian), the p-block is d ln P / dp alone
    because ln n does not depend on p."""
    dln_nu_dq = np.asarray(dln_nu_dq)
    dlnp_deta = np.asarray(dlnp_deta)
    return np.concatenate([dln_nu_dq + dlnp_deta[:, :3], dlnp_deta[:, 3:]],
                          axis=-1)


def cbe_residual_alpha(p, dlnf_dq, dlnf_dp, alpha, eps=1.0):
    """Steady-state CBE residual in acceleration convention:

        R = p . s_q + alpha . s_p ,   alpha = -grad_q phi,

    equivalent to the gradient form R = p . s_q - (grad_q phi) . s_p.
    Returns (term1, term2, R, relR); all inputs (..., 3)."""
    term1 = np.sum(np.asarray(p) * np.asarray(dlnf_dq), axis=-1)
    term2 = np.sum(np.asarray(alpha) * np.asarray(dlnf_dp), axis=-1)
    r = term1 + term2
    rel = np.abs(r) / (np.abs(term1) + np.abs(term2) + eps)
    return term1, term2, r, rel


def local_force_svd_alpha(C, y, weights=None, rcond=1e-6):
    """Weighted SVD solve of the local force constraint  C alpha ~ y  in the
    acceleration convention, with C[i] = s_p,i^T (velocity-block score rows)
    and y[i] = -p_i . s_q,i (from R = 0).  Non-negative weights W rescale
    rows to D = W^{1/2} C before the SVD.

    Returns dict(sigma, rank, cond, resid_rel, alpha_hat, y_norm); alpha_hat
    is None when rank-deficient: no stable force estimate is emitted."""
    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        w = np.ones(len(y))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
    D = np.sqrt(w)[:, None] * C
    yw = np.sqrt(w) * y
    s = np.linalg.svd(D, compute_uv=False)
    rank = int(np.sum(s > rcond * s[0])) if s[0] > 0 else 0
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    alpha, _res, _rank, _s = np.linalg.lstsq(D, yw, rcond=rcond)
    resid_rel = float(np.linalg.norm(D @ alpha - yw)
                      / max(np.linalg.norm(yw), 1e-300))
    return {
        "sigma": s,
        "rank": rank,
        "cond": cond,
        "resid_rel": resid_rel,
        "alpha_hat": alpha if rank == C.shape[1] else None,
        "y_norm": float(np.linalg.norm(yw)),
    }


def absorbable_split(C, e, weights=None, rcond=1e-6):
    """Split a constraint-residual vector e (row space) into the part a
    force perturbation can absorb, W^{-1/2} (D D^+) W^{1/2} e with
    D = W^{1/2} C, and the orthogonal remainder (plan appendix A4).

    Returns dict(absorbable, orthogonal, projection_rows) where
    projection_rows[i] = (D D^+)_{ii} is the leverage of constraint i."""
    C = np.asarray(C, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, dtype=np.float64)
    D = np.sqrt(w)[:, None] * C
    ew = np.sqrt(w) * e
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    keep = s > rcond * (s[0] if s[0] > 0 else 0.0)
    Uk = U[:, keep]
    absorbable = (Uk @ (Uk.T @ ew)) / np.sqrt(w)   # (D D^+) e in row space
    return {
        "absorbable": absorbable,
        "orthogonal": e - absorbable,
        "projection_rows": np.einsum("ij,ij->i", U[:, keep], U[:, keep]),
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


def fd_scan(fn_scalar, rows, ad_grads, floors, steps=FD_STEPS):
    """FD-vs-autodiff error AND raw FD gradients per step size.

    Returns (err, fd_grads): err[h][c] = median_i |fd - ad| / max(|ad|,
    floor_c); fd_grads has shape (n_steps, n_rows, n_comp) so adjacent-step
    agreement can be checked without re-evaluating the model."""
    n, n_c = ad_grads.shape
    err = np.full((len(steps), n_c), np.nan)
    fd_grads = np.empty((len(steps), n, n_c))
    for hi, h in enumerate(steps):
        for i in range(n):
            fd_grads[hi, i] = central_fd_grad(fn_scalar, rows[i], h)
        rel = np.abs(fd_grads[hi] - ad_grads) / np.maximum(np.abs(ad_grads), floors)
        err[hi] = np.median(rel, axis=0)
    return err, fd_grads


def fd_stable_intervals(err, steps, level):
    """Per component: maximal runs of >= 2 consecutive steps whose median
    floored-rel error is under 'level'.  Returns (per_component_intervals,
    all_components_stable).  A single isolated minimum does NOT pass."""
    steps = np.asarray(steps)
    intervals = []
    for c in range(err.shape[1]):
        ok = np.append(err[:, c] < level, False)
        runs, start = [], None
        for i, good in enumerate(ok):
            if good and start is None:
                start = i
            elif not good and start is not None:
                if i - start >= 2:
                    runs.append((float(steps[start]), float(steps[i - 1])))
                start = None
        intervals.append(runs)
    return intervals, all(len(r) > 0 for r in intervals)


def fd_pair_agreement(fd_grads, ad_grads, floors, err, steps, level):
    """Adjacent-step FD agreement inside the stable region: for the first
    qualifying adjacent pair per component, the two FD gradients must also
    agree with each other under 'level' (floored median over rows)."""
    agree = np.zeros(err.shape[1], dtype=bool)
    pairs = [None] * err.shape[1]
    for c in range(err.shape[1]):
        scale = max(float(np.median(np.abs(ad_grads[:, c]))), float(floors[c]))
        for i in range(len(steps) - 1):
            if err[i, c] < level and err[i + 1, c] < level:
                rel = float(np.median(np.abs(fd_grads[i, :, c] - fd_grads[i + 1, :, c]) / scale))
                pairs[c] = {"steps": [float(steps[i]), float(steps[i + 1])],
                            "median_floored_rel": rel}
                agree[c] = bool(rel < level)
                break
    return pairs, bool(np.all(agree))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_labels(r_q, edges):
    """Index of r_q into the frozen q-band partition; points outside the
    outermost edges are clipped into the edge bands."""
    return np.clip(np.searchsorted(edges, r_q, side="right") - 1, 0, len(edges) - 2)


# --------------------------------------------------------------------------
# control lineage: manifest hash gate + model loading from the control
# snapshot (source 5f76faa), never from the audit branch's own copies
# --------------------------------------------------------------------------

def load_t0_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "dpjax.nf-score-audit.t0-manifest.v1":
        raise RuntimeError(f"unexpected manifest schema in {path}")
    return manifest


def verify_control_hashes(manifest, control_run_dir, control_repo=None):
    """Re-hash every file pinned in t0-manifest.json and fail loudly on any
    mismatch (T1 stop condition).  Returns the provenance dict for the
    audit/cache manifests.

    Pinned keys are relative to the control run artifacts dir, except the
    options.json entry which the manifest stores relative to the snapshot
    repo root (run 56b8c01a finding); each key resolves against the run dir
    first, then the repo root."""
    control_run_dir = Path(control_run_dir)
    bases = [control_run_dir] + ([Path(control_repo)] if control_repo else [])
    prov = {}
    for rel, frozen_sha in manifest["control"]["checkpoints_sha256"].items():
        path = next((b / rel for b in bases if (b / rel).exists()), None)
        if path is None:
            raise RuntimeError(f"pinned control file missing: {rel} "
                               f"(looked under {[str(b) for b in bases]})")
        sha = sha256_file(path)
        if sha != frozen_sha:
            raise RuntimeError(
                f"{path} sha256 {sha[:12]} != manifest {frozen_sha[:12]} "
                "(T1 stop condition: hash mismatch)")
        prov[rel] = {"path": str(path), "sha256": sha}
    return prov


def model_code_hashes(control_repo):
    """sha256 of the model-defining files of the control snapshot that the
    audit actually imports (derivative-source lineage)."""
    control_repo = Path(control_repo)
    out = {}
    for rel in MODEL_CODE_FILES:
        p = control_repo / rel
        if not p.exists():
            raise RuntimeError(f"control snapshot file missing: {p}")
        out[rel] = sha256_file(p)
    return out


def load_control_models(control_repo, control_run_dir):
    """Load the deployed control flow pair + Phi from the control snapshot's
    own code (source commit 5f76faa pinned in the T0 manifest).

    Returns (flow, flow_spatial_reference, phi).  flow_spatial_reference is
    the flow_pos_only-10 load used for the single-flow-pair integrity
    cross-check."""
    control_scripts = str(Path(control_repo) / "scripts")
    if control_scripts not in sys.path:
        sys.path.insert(0, control_scripts)
    import jax
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        import fit_all
        from flow_ot_flow_matching_conditional import ConditionalPhaseSpaceFlow
        flow_dir = Path(control_run_dir) / "models" / "df" / "flow"
        flow = fit_all.load_flow(flow_dir, checkpoint_index=FLOW_CKPT_INDEX)
        flow_spatial_ref, _ = ConditionalPhaseSpaceFlow.load(
            flow_dir, load_index=SPATIAL_CKPT_INDEX, load_prefix="flow_pos_only")
        phi = fit_all.load_potential(
            Path(control_run_dir) / "models" / "Phi", checkpoint_index=PHI_CKPT_INDEX)
    finally:
        jax.config.update("jax_enable_x64", prev)
    return flow, flow_spatial_ref, phi


def flow_pair_integrity(flow, flow_spatial_ref):
    """The deployed flow-21 must carry bitwise the trained spatial flow of
    flow_pos_only-10 (control = single flow pair, no ensemble)."""
    import jax
    import equinox as eqx
    a = jax.tree_util.tree_leaves(eqx.filter(flow.spatial_flow, eqx.is_array))
    b = jax.tree_util.tree_leaves(eqx.filter(flow_spatial_ref.spatial_flow, eqx.is_array))
    if len(a) != len(b):
        return {"n_leaves": [len(a), len(b)], "max_abs_diff": float("inf"),
                "bitwise_identical": False}
    dmax = max((float(jax.numpy.max(jax.numpy.abs(x - y))) for x, y in zip(a, b)),
               default=0.0)
    return {"n_leaves": len(a), "max_abs_diff": dmax,
            "bitwise_identical": bool(dmax == 0.0)}


def _with_ode_tolerance(flow, rtol, atol):
    """Same DF weights, stricter diffrax PIDController on both sub-flows.

    The controller is a static field of the VectorField bijection, so
    eqx.tree_at refuses to graft a replacement.  A deepcopy of the flow is
    mutated in place via object.__setattr__ instead; the caller's original
    flow is untouched."""
    import diffrax

    strict = diffrax.PIDController(rtol=rtol, atol=atol)
    flow2 = copy.deepcopy(flow)
    for sub in (flow2.spatial_flow, flow2.conditional_velocity_flow):
        chain = sub.flow.bijection
        vf = chain[0] if hasattr(chain, "__getitem__") else chain.layers[0]
        object.__setattr__(vf, "stepsize_controller", strict)
    return flow2


def _x64(on):
    import jax

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
    (flow_sampling.value_and_grad_lnf_fn), plus jit."""
    import jax
    import equinox as eqx
    return jax.jit(jax.vmap(eqx.filter_value_and_grad(flow.log_prob)))


def _split_fns(flow):
    """vmapped position log-density and conditional-velocity log-density
    with their eta-gradients, for the chain identities:
    lnF = ln n(q) + ln P(p|q), the condition-normalization Jacobian carried
    by autodiff of the conditional term."""
    import jax
    import equinox as eqx
    vg_pos = jax.jit(jax.vmap(eqx.filter_value_and_grad(flow.log_prob_position)))

    def _lp(eta):
        return flow.log_prob_velocity_given_position(eta)
    vg_vel = jax.jit(jax.vmap(eqx.filter_value_and_grad(_lp)))
    return vg_pos, vg_vel


def _lnf_fn_scalar(flow):
    import jax

    def fn(x):
        return flow.log_prob(x)
    return jax.jit(fn)


def eval_batched(fn, arr, batch=BATCH):
    """Apply a vmapped fn in batches; fn must return a tuple of arrays with
    leading batch axis."""
    outs = []
    for i in range(0, len(arr), batch):
        outs.append(fn(arr[i:i + batch]))
    return tuple(
        np.concatenate([np.asarray(o[k]) for o in outs], axis=0)
        for k in range(len(outs[0])))


def _finite_or_fail(arr, name):
    a = np.asarray(arr)
    if not np.all(np.isfinite(a)):
        n_bad = int(np.sum(~np.isfinite(a)))
        raise RuntimeError(f"non-finite values in {name}: {n_bad} (T1 stop condition)")
    return a


def _dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# T1 stage: score numerical-chain audit on the frozen control
# --------------------------------------------------------------------------

def score_chain_audit(control_repo, control_run_dir, manifest_path, out_dir,
                      n_rows=AUDIT_N_ROWS, n_strict=AUDIT_N_STRICT,
                      n_fd=AUDIT_N_FD, smoke=False, force=False):
    import h5py
    import jax
    import jax.numpy as jnp

    out_dir = Path(out_dir)
    json_path = out_dir / "audit_score_chain.json"
    if json_path.exists() and not force:
        raise RuntimeError(f"{json_path} exists (pass --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = load_t0_manifest(manifest_path)
    prov = verify_control_hashes(manifest, control_run_dir, control_repo)
    code_hashes = model_code_hashes(control_repo)

    if smoke:
        n_rows, n_strict, n_fd = 64, 16, 4

    print(f"[t1.audit] loading stored gradients from {control_run_dir}/data/df_gradients.h5")
    with h5py.File(Path(control_run_dir) / "data" / "df_gradients.h5", "r") as f:
        eta_all = f["eta"][:]
        lnf_st_all = f["lnf"][:]
        dlnf_st_all = f["dlnf_deta"][:]
        dlnp_st_all = f["dlnp_deta"][:]
        stored_attrs = dict(f.attrs)
    n_stored_total = len(eta_all)
    rows = stratified_rows_by_radius(eta_all, n_rows)
    eta = eta_all[rows].astype(np.float64)
    lnf_st = lnf_st_all[rows].astype(np.float64)
    dlnf_st = dlnf_st_all[rows].astype(np.float64)
    dlnp_st = dlnp_st_all[rows].astype(np.float64)
    strict_idx = stratified_subset_indices(n_rows, n_strict)
    fd_idx = stratified_subset_indices(n_rows, n_fd)

    print("[t1.audit] loading control models from the pinned snapshot")
    flow, flow_spatial_ref, _phi = load_control_models(control_repo, control_run_dir)
    integrity = flow_pair_integrity(flow, flow_spatial_ref)
    print(f"[t1.audit] flow-pair integrity: {integrity}")

    # stored-data identity (no recompute): the p-block of dlnf must equal
    # the p-block of dlnp because log n(q) does not depend on p
    pscore_rel = rel_with_floor(dlnf_st[:, 3:] - dlnp_st[:, 3:], dlnp_st[:, 3:],
                                component_floors(dlnp_st[:, 3:]))
    pscore_stored = group_summary({"velocity": pscore_rel})

    # regime f32 (production-matching arithmetic on this GPU), plus a
    # same-input repeat for the device-determinism record
    t_reg = time.time()
    with _x64(False):
        fn32 = _grad_lnf_fn(flow)
        lnf_32, dlnf_32 = eval_batched(fn32, jnp.asarray(eta, dtype=jnp.float32))
        lnf_32_rep, dlnf_32_rep = eval_batched(fn32, jnp.asarray(eta, dtype=jnp.float32))
    t_f32 = time.time() - t_reg

    # regime x64 (f64 inputs x f32 weights), default ODE tolerance
    t_reg = time.time()
    with _x64(True):
        fn64 = _grad_lnf_fn(flow)
        lnf_64, dlnf_64 = eval_batched(fn64, jnp.asarray(eta, dtype=jnp.float64))
    t_x64 = time.time() - t_reg

    # strict ODE tolerance on the stratified subset (x64)
    t_reg = time.time()
    flow_strict = _with_ode_tolerance(flow, ODE_STRICT_RTOL, ODE_STRICT_ATOL)
    with _x64(True):
        fns = _grad_lnf_fn(flow_strict)
        lnf_strict, dlnf_strict = eval_batched(
            fns, jnp.asarray(eta[strict_idx], dtype=jnp.float64))
    t_strict = time.time() - t_reg

    # chain identities on the strict subset: split evaluation must
    # reproduce the full log_prob and its gradient (Jacobian included)
    with _x64(True):
        vg_pos, vg_vel = _split_fns(flow_strict)
        eta_s64 = jnp.asarray(eta[strict_idx], dtype=jnp.float64)
        ln_nu, dln_nu_dq = eval_batched(vg_pos, jnp.asarray(eta[strict_idx][:, :3]))
        lnp, dlnp_full = eval_batched(vg_vel, eta_s64)
    ident_val = np.abs((ln_nu + lnp) - lnf_strict)
    ident_scale = max(float(np.median(np.abs(lnf_strict))), 1e-30)
    ident_grad = rel_with_floor(combine_split_grads(dln_nu_dq, dlnp_full)
                                - dlnf_strict,
                                dlnf_strict, component_floors(dlnf_strict))
    pscore_recompute = rel_with_floor(dlnf_strict[:, 3:] - dlnp_full[:, 3:],
                                      dlnp_full[:, 3:], component_floors(dlnp_full[:, 3:]))

    # FD scan (x64, strict ODE): validates the autodiff gradient against
    # the implemented lnF itself, with the autodiff reference evaluated at
    # exactly the FD rows
    t_reg = time.time()
    with _x64(True):
        fd_rows = np.asarray(eta[fd_idx], dtype=np.float64)
        fns_fd = _grad_lnf_fn(flow_strict)
        _l_fd, ad_fd = eval_batched(fns_fd, jnp.asarray(fd_rows))
        scalar = _lnf_fn_scalar(flow_strict)
        fd_err, fd_grads = fd_scan(scalar, fd_rows, np.asarray(ad_fd),
                                   component_floors(np.asarray(ad_fd)))
    t_fd = time.time() - t_reg
    fd_intervals, fd_all_stable = fd_stable_intervals(fd_err, FD_STEPS, TH_FD_LEVEL)
    fd_pairs, fd_pair_ok = fd_pair_agreement(fd_grads, np.asarray(ad_fd),
                                             component_floors(np.asarray(ad_fd)),
                                             fd_err, FD_STEPS, TH_FD_LEVEL)

    # comparisons against the stored production values; the total-chain gate
    # is evaluated on the strict subset (x64-strict reference vs stored f32)
    floors_st = component_floors(dlnf_st)
    cmp_chain = rel_with_floor(dlnf_strict - dlnf_st[strict_idx], dlnf_st[strict_idx],
                               component_floors(dlnf_st[strict_idx]))
    cmp_ode = rel_with_floor(dlnf_strict - dlnf_64[strict_idx], dlnf_64[strict_idx],
                             component_floors(dlnf_64[strict_idx]))
    cmp_prec = rel_with_floor(dlnf_64 - dlnf_st, dlnf_st, floors_st)

    s_chain = group_summary({"spatial": cmp_chain[:, :3], "velocity": cmp_chain[:, 3:]})
    s_ode = group_summary({"spatial": cmp_ode[:, :3], "velocity": cmp_ode[:, 3:]})
    s_prec = group_summary({"spatial": cmp_prec[:, :3], "velocity": cmp_prec[:, 3:]})
    ident_summary = {
        "value_abs_over_scale_median": float(np.median(ident_val / ident_scale)),
        "value_abs_over_scale_p99": float(np.percentile(ident_val / ident_scale, 99)),
        "grad_group": group_summary({"spatial": ident_grad[:, :3],
                                     "velocity": ident_grad[:, 3:]}),
    }
    determinism = {
        "lnf_max_abs_diff": float(np.max(np.abs(lnf_32 - lnf_32_rep))),
        "dlnf_max_abs_diff": float(np.max(np.abs(dlnf_32 - dlnf_32_rep))),
    }

    # radius-band stratification of the total chain error (T0 partition)
    band_edges_q = np.array([0.1, 0.2, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0])
    r_q = np.linalg.norm(eta[strict_idx][:, :3], axis=1)
    bands = band_labels(r_q, band_edges_q)
    per_band = []
    for b in range(len(band_edges_q) - 1):
        m = bands == b
        if not np.any(m):
            continue
        per_band.append({
            "q_band": [float(band_edges_q[b]), float(band_edges_q[b + 1])],
            "r_kpc_range": [float(np.min(r_q[m]) * L_KPC), float(np.max(r_q[m]) * L_KPC)],
            "n_rows": int(np.sum(m)),
            "chain_floored_rel_median": float(np.median(cmp_chain[m])),
            "chain_floored_rel_p99": float(np.percentile(cmp_chain[m], 99)),
        })

    gates = {
        "hashes_ok": True,
        "flow_pair_integrity_ok": bool(integrity["bitwise_identical"]),
        "chain_median_ok": bool(max(s_chain["spatial_median"], s_chain["velocity_median"])
                                < TH_CHAIN_MEDIAN),
        "chain_p99_ok": bool(max(s_chain["spatial_p99"], s_chain["velocity_p99"])
                             < TH_CHAIN_P99),
        "ode_tolerance_ok": bool(max(s_ode["spatial_median"], s_ode["velocity_median"])
                                 < TH_ODE_MEDIAN),
        "fd_adjacent_stable_ok": bool(fd_all_stable and fd_pair_ok),
        "identity_ok": bool(ident_summary["value_abs_over_scale_median"] < TH_IDENTITY_MEDIAN
                            and ident_summary["value_abs_over_scale_p99"] < TH_IDENTITY_P99
                            and max(ident_summary["grad_group"]["spatial_median"],
                                    ident_summary["grad_group"]["velocity_median"])
                            < TH_IDENTITY_P99),
        "pscore_stored_ok": bool(pscore_stored["velocity_median"] < TH_PSCORE_STORED_MEDIAN),
    }
    gates["all_ok"] = bool(all(gates.values()))

    _finite_or_fail(dlnf_32, "dlnf_ad_f32")
    _finite_or_fail(dlnf_64, "dlnf_ad_x64")
    _finite_or_fail(dlnf_strict, "dlnf_ad_strict")
    _finite_or_fail(fd_grads, "fd_grads")

    result = {
        "stage": "T1 score numerical-chain audit",
        "contract": {
            "units": {"L_kpc": L_KPC, "V_kms": V_KMS,
                      "score": "grad_eta log F with eta=[q,p]",
                      "acceleration": "alpha = -grad_q phi"},
            "control": manifest["control"],
            "gates": {
                "chain_median": f"stored-vs-x64strict floored rel median < {TH_CHAIN_MEDIAN} (spatial and velocity)",
                "chain_p99": f"... and p99 < {TH_CHAIN_P99}",
                "ode_tolerance": f"x64 strict-vs-default floored rel median < {TH_ODE_MEDIAN}",
                "fd_adjacent": f">=2 adjacent FD steps with err < {TH_FD_LEVEL} and FD-pair agreement < {TH_FD_LEVEL} in every component",
                "identity": f"lnf split / grad identities < {TH_IDENTITY_MEDIAN} median, < {TH_IDENTITY_P99} p99",
                "pscore_stored": f"stored dlnf vs dlnp p-block floored rel median < {TH_PSCORE_STORED_MEDIAN}",
                "status": "engineering criteria re-declared 2026-09-30 from the T0 numerical budget candidates; not significance levels and not inherited from the 2026-09-15 audit",
            },
        },
        "provenance": {
            "t0_manifest": str(manifest_path),
            "checkpoints": prov,
            "model_code_sha256": code_hashes,
            "control_source_commit": manifest["control"]["source_commit"],
            "stored_df_gradients_attrs": stored_attrs,
        },
        "environment": {
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
            "ode_default": {"rtol": 1e-4, "atol": 1e-5, "solver": "Tsit5"},
            "ode_strict": {"rtol": ODE_STRICT_RTOL, "atol": ODE_STRICT_ATOL},
            "divergence_implementation": {
                "cnf": "exact jacfwd trace (flow_vector_fields._augmented_dynamics_fn)",
                "phi_laplacian": "exact Hessian trace (potential.calc_phi_derivatives)",
                "hutchinson": "training-time Jacobian penalty only (compute_jacobian_penalty); NOT in the evaluated score path",
            },
        },
        "n_points": {"audited_rows": int(n_rows), "strict_subset": int(n_strict),
                     "fd_subset": int(n_fd), "stored_total": int(n_stored_total),
                     "smoke": bool(smoke)},
        "single_flow_pair_record": {
            "deployed": "flow-21 (spatial flow_pos_only-10 carried bitwise inside)",
            "integrity": integrity,
            "ensemble": "none; mixture machinery (flow_sampling.combine_log_prob_and_derivatives) is an identity passthrough at len(flow_list)==1, recorded not re-studied",
        },
        "results": {
            "stored_vs_x64strict": s_chain,
            "x64default_vs_x64strict_ode": s_ode,
            "stored_vs_x64default_precision": s_prec,
            "chain_identities": ident_summary,
            "pscore_stored_identity": pscore_stored,
            "pscore_recompute_identity": group_summary({"velocity": pscore_recompute}),
            "determinism_repeat_f32": determinism,
            "fd_scan": {
                "steps": FD_STEPS.tolist(),
                "median_rel_err_per_step_per_comp": fd_err.tolist(),
                "stable_intervals_per_comp": fd_intervals,
                "pair_agreement_per_comp": fd_pairs,
            },
            "per_band": per_band,
            "radius_range_kpc": [float(np.min(np.linalg.norm(eta[:, :3], axis=1)) * L_KPC),
                                 float(np.max(np.linalg.norm(eta[:, :3], axis=1)) * L_KPC)],
        },
        "gates": gates,
        "timing_s": {"f32_rows": t_f32, "x64_rows": t_x64,
                     "strict_subset": t_strict, "fd_scan": t_fd},
        "elapsed_s": time.time() - t0,
    }

    _dump_json(json_path, result)
    np.savez_compressed(
        out_dir / "audit_score_chain.npz",
        rows=rows, eta=eta, lnf_stored=lnf_st, dlnf_stored=dlnf_st,
        dlnp_stored=dlnp_st, lnf_ad_f32=lnf_32, dlnf_ad_f32=dlnf_32,
        dlnf_ad_x64=dlnf_64, strict_idx=strict_idx, fd_idx=fd_idx,
        dlnf_ad_strict=dlnf_strict, fd_err=fd_err, fd_grads=fd_grads)
    print(f"[t1.audit] gates: {gates}")
    print(f"[t1.audit] done -> {json_path} ({result['elapsed_s']:.1f} s)")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="T1 score numerical-chain audit on the frozen control "
                    "(use the input manifest for this audit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score-chain", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--control-repo", required=True,
                   help="root of the control run snapshot (source 5f76faa)")
    p.add_argument("--control-run-dir", required=True,
                   help="artifacts dir of run 2b32eb04 (contains models/, data/)")
    p.add_argument("--manifest", default="docs/history/nf-score-audit/t0-manifest.json",
                   help="T0 manifest binding the control lineage")
    p.add_argument("--out-dir", default="runs/nf-score-audit/t1-pilot",
                   help="output directory for audit_score_chain.json/.npz")
    p.add_argument("--rows", type=int, default=AUDIT_N_ROWS,
                   help="radius-stratified stored rows to audit")
    p.add_argument("--strict-rows", type=int, default=AUDIT_N_STRICT,
                   help="subset re-solved at strict ODE tolerance")
    p.add_argument("--fd-points", type=int, default=AUDIT_N_FD,
                   help="subset for the finite-difference scan")
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes for plumbing checks")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing outputs")
    args = parser.parse_args()

    score_chain_audit(args.control_repo, args.control_run_dir, args.manifest,
                      args.out_dir, n_rows=args.rows, n_strict=args.strict_rows,
                      n_fd=args.fd_points, smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
