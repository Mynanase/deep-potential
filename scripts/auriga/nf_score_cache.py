#!/usr/bin/env python
"""T1 fixed-point score cache shared by agents B and C (weak score tests /
local force inversion).

Builds the three T0 point sets, evaluates the frozen control model on them
and writes hash-bound caches (plan docs/nf-score-audit-plan.md sec. 4.2):

  heldout           first int(0.25*n) rows of halo12-clean-smooth.h5
                    (the DF validation split; exploratory qualification)
  velocity_probes   7 shells x 16 Sobol directions x 96 shared velocities;
                    proposal = diag Gaussian with the population velocity
                    std, importance weights w = exp(lnF - log G)
  spatial_grid      19 fixed radii x 2048 Sobol directions; Phi products
                    phi / grad_phi / acceleration = -grad_phi / laplacian

Every cache binds: checkpoint sha256s (T0 manifest), control model-code
sha256s, the audit script's code hash, point-order hashes, precisions, ODE
tolerances and units.  A new coordinate/order/precision/numerics setting
means a NEW cache directory, never an overwrite.

Cache certification: the numerical-chain audit (audit_df_constraints.py)
must have run into --audit-json; its gates.all_ok stamps certified=true.
On a failed chain the cache is still written but stamped
science_use="debug only, not for B/C conclusions" (T1 card).

Stage pilot (<= 2048 distinct points total with the audit's 1024):
  heldout 512 stratified rows, probes 2 shells x 16 dirs x 8 velocities,
  grid 2 radii x 128 directions, throughput sweep, extrapolated full-cache
  wall time.  Stage full: the complete sets.

Run inside an orchestration snapshot on the GPU host, from the repo root:
  JAX_PLATFORMS=cuda python scripts/auriga/nf_score_cache.py --stage pilot \\
      --control-repo /home/qiutao/.orx/runs/<run>/repo \\
      --control-run-dir <...>/repo/runs/orx
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

# NOTE: imported here, BEFORE any control-snapshot path is prepended at
# runtime, so the generic helper resolves to THIS branch's copy.
from validate_enclosed_mass import sobol_directions  # noqa: E402
import audit_df_constraints as adc  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0

# ---- frozen point-set designs (T0 contract; do not change silently) ----
Q_BAND_EDGES = np.array([0.1, 0.2, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0])
REPORT_BANDS_KPC = [(2, 10), (10, 30), (30, 50), (50, 70)]
# shell radius per q-band = geometric mean of the band edges, in kpc
SHELL_R_KPC = 10.0 * np.sqrt(Q_BAND_EDGES[:-1] * Q_BAND_EDGES[1:])
N_PROBE_DIRS = 16
N_SHARED_VEL = 96
PROBE_SEED = 3                      # T0 diagnostic-layer candidate: probes
SPATIAL_RADII_KPC = np.concatenate([np.arange(2.0, 30.0, 3.0), np.arange(30.0, 70.1, 5.0)])
N_GRID_DIRS = 2048

# pilot subset design (distinct-point budget: 512 + 256 + 256 = 1024)
PILOT_HELDOUT_ROWS = 512
PILOT_PROBE_SHELLS = [3, 6]         # 24.5 kpc and 65.6 kpc
PILOT_PROBE_VELS = 8
PILOT_GRID_RADII = [35.0, 65.0]
PILOT_GRID_DIRS = 128

CACHE_SCHEMA = "dpjax.nf-score-audit.t1-cache.v1"
UNITS = {"L_kpc": L_KPC, "V_kms": V_KMS,
         "eta": "dimensionless [q, p]", "score": "grad_eta log F",
         "grad_phi": "grad_q phi, code units",
         "acceleration": "alpha = -grad_q phi, code units",
         "alpha_physical": "a_phys = (V^2/L_km) * alpha, L_km = 3.0856775814913673e17 km"}


# --------------------------------------------------------------------------
# point-set builders (numpy only; deterministic given the frozen seeds)
# --------------------------------------------------------------------------

def heldout_slice(pop_path, n_rows=None):
    """Heldout = DF validation split of halo12-clean-smooth.h5: the FIRST
    int(0.25*n) rows of the seed-0 shuffled file (utils.split_data rule).
    n_rows selects a radius-stratified subsample for the pilot."""
    import h5py
    with h5py.File(pop_path, "r") as f:
        n_total = f["eta"].shape[0]
        n_val = int(n_total * 0.25)
        eta = f["eta"][:n_val].astype(np.float64)
        out = {
            "row_index": np.arange(n_val, dtype=np.int64),
            "particle_id": f["particle_id"][:n_val].astype(np.int64),
            "source_index": f["source_index"][:n_val].astype(np.int64),
            "mass": f["mass"][:n_val].astype(np.float64),
            "weights": f["weights"][:n_val].astype(np.float64),
            "eta": eta,
        }
    r_q = np.linalg.norm(eta[:, :3], axis=1)
    out["band_q"] = adc.band_labels(r_q, Q_BAND_EDGES)
    r_kpc = r_q * L_KPC
    out["band_report"] = np.full(n_val, -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(REPORT_BANDS_KPC):
        out["band_report"][(r_kpc >= lo) & (r_kpc < hi)] = i
    if n_rows is not None and n_rows < n_val:
        keep = adc.stratified_rows_by_radius(eta, n_rows)
        for k in out:
            out[k] = out[k][keep]
    return out


def population_velocity_std(pop_path):
    """Per-axis velocity dispersion of the full population in code units;
    the scale of the frozen shared-velocity proposal (recorded, not fitted
    per shell)."""
    import h5py
    with h5py.File(pop_path, "r") as f:
        vel = f["eta"][:, 3:]
        return np.std(vel, axis=0).astype(np.float64)


def velocity_probes(sigma_vel, shells=None, n_vel=None):
    """7 shells x 16 angular Sobol directions (seed PROBE_SEED) x n_vel
    shared velocities.  The SAME velocity set is used at every position so
    B/C comparisons across shells and directions stay paired; the proposal
    is the diag-Gaussian with the population velocity std."""
    shells = np.arange(len(SHELL_R_KPC)) if shells is None else np.asarray(shells)
    n_vel = N_SHARED_VEL if n_vel is None else n_vel
    dirs = sobol_directions(N_PROBE_DIRS, PROBE_SEED)
    rng = np.random.default_rng(PROBE_SEED)
    vel_all = rng.normal(size=(N_SHARED_VEL, 3)) * np.asarray(sigma_vel)
    vel = vel_all[:n_vel]
    log_g = -0.5 * np.sum((vel / np.asarray(sigma_vel)) ** 2, axis=1) \
        - np.sum(np.log(np.asarray(sigma_vel))) - 1.5 * np.log(2.0 * np.pi)
    eta, ids, shell_idx, dir_idx, vel_idx, logp = [], [], [], [], [], []
    for s in shells:
        q = (SHELL_R_KPC[s] / L_KPC) * dirs
        for d in range(N_PROBE_DIRS):
            for v in range(n_vel):
                eta.append(np.concatenate([q[d], vel[v]]))
                ids.append((int(s) * N_PROBE_DIRS + d) * N_SHARED_VEL + v)
                shell_idx.append(int(s)); dir_idx.append(d); vel_idx.append(v)
                logp.append(log_g[v])
    return {
        "point_id": np.asarray(ids, dtype=np.int64),
        "shell_idx": np.asarray(shell_idx, dtype=np.int64),
        "dir_idx": np.asarray(dir_idx, dtype=np.int64),
        "vel_idx": np.asarray(vel_idx, dtype=np.int64),
        "r_kpc": SHELL_R_KPC[np.asarray(shell_idx)],
        "eta": np.asarray(eta, dtype=np.float64),
        "log_proposal": np.asarray(logp, dtype=np.float64),
        "sigma_vel": np.asarray(sigma_vel, dtype=np.float64),
        "shared_velocities": vel_all,
        "directions": dirs,
    }


def spatial_grid(radii=None, n_dirs=None):
    """19 fixed radii x 2048 Sobol directions (same scrambled sequence as
    the probes: its first 16 directions ARE the probe directions, so the
    two designs are prefix-nested at fixed seed)."""
    radii = SPATIAL_RADII_KPC if radii is None else np.asarray(radii, dtype=np.float64)
    n_dirs = N_GRID_DIRS if n_dirs is None else n_dirs
    dirs = sobol_directions(N_GRID_DIRS, PROBE_SEED)[:n_dirs]
    q, ids, r_idx, d_idx = [], [], [], []
    for i, r_kpc in enumerate(radii):
        for d in range(n_dirs):
            q.append((r_kpc / L_KPC) * dirs[d])
            ids.append(int(i) * N_GRID_DIRS + d)
            r_idx.append(int(i)); d_idx.append(d)
    return {
        "point_id": np.asarray(ids, dtype=np.int64),
        "radius_idx": np.asarray(r_idx, dtype=np.int64),
        "dir_idx": np.asarray(d_idx, dtype=np.int64),
        "r_kpc": radii[np.asarray(r_idx)],
        "q": np.asarray(q, dtype=np.float64),
        "radii_kpc": radii,
    }


def point_order_hash(name, ids, eta):
    """sha256 over the canonical byte order of the point set: name, shape,
    int64 ids, float64 coordinates.  Any reorder / value change / precision
    change of the saved order produces a different hash."""
    import hashlib
    hh = hashlib.sha256()
    hh.update(name.encode())
    hh.update(np.asarray(ids, dtype=np.int64).tobytes())
    hh.update(np.asarray(eta, dtype=np.float64).tobytes())
    return hh.hexdigest()


# --------------------------------------------------------------------------
# cache writing (atomic per file; manifest written LAST = completeness mark)
# --------------------------------------------------------------------------

def _write_h5_atomic(path, arrays, attrs):
    import h5py
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for k, v in arrays.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
    tmp.rename(path)
    return adc.sha256_file(path)


def build_cache(control_repo, control_run_dir, manifest_path, population,
                out_dir, stage, batch=2048, heldout_x64_rows=4096,
                audit_json=None, points_only=False, pilot_rows=PILOT_HELDOUT_ROWS):
    import h5py
    import jax
    import jax.numpy as jnp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    manifest = adc.load_t0_manifest(manifest_path)
    prov = adc.verify_control_hashes(manifest, control_run_dir, control_repo)
    code_hashes = adc.model_code_hashes(control_repo)
    pop_sha = adc.sha256_file(population)

    if audit_json is None:
        audit_json = out_dir / "audit_score_chain.json"
    audit_json = Path(audit_json)
    if not points_only:
        if not audit_json.exists():
            raise RuntimeError(f"audit gates missing: {audit_json} (run "
                               "audit_df_constraints.py first)")
        audit = json.loads(audit_json.read_text())
        audit["_path"] = str(audit_json)
        certified = bool(audit.get("gates", {}).get("all_ok"))
    else:
        audit, certified = None, False

    print(f"[t1.cache:{stage}] building point sets")
    pilot = stage == "pilot"
    heldout = heldout_slice(population, n_rows=pilot_rows if pilot else None)
    sigma_vel = population_velocity_std(population)
    probes = velocity_probes(sigma_vel,
                             shells=PILOT_PROBE_SHELLS if pilot else None,
                             n_vel=PILOT_PROBE_VELS if pilot else None)
    grid = spatial_grid(radii=PILOT_GRID_RADII if pilot else None,
                        n_dirs=PILOT_GRID_DIRS if pilot else None)
    print(f"[t1.cache:{stage}] heldout={len(heldout['eta'])} probes={len(probes['eta'])} "
          f"grid={len(grid['q'])} sigma_vel={sigma_vel.tolist()}")

    # ---- points files (hashes computed over the frozen order)
    points = {
        "heldout": (heldout["particle_id"], heldout["eta"],
                    {k: heldout[k] for k in ["row_index", "particle_id", "source_index",
                                             "mass", "weights", "eta", "band_q", "band_report"]}),
        "velocity_probes": (probes["point_id"], probes["eta"],
                            {k: probes[k] for k in ["point_id", "shell_idx", "dir_idx",
                                                    "vel_idx", "r_kpc", "eta", "log_proposal"]}),
        "spatial_grid": (grid["point_id"], grid["q"],
                         {k: grid[k] for k in ["point_id", "radius_idx", "dir_idx",
                                               "r_kpc", "q"]}),
    }
    order_hashes, points_files = {}, {}
    for name, (ids, coords, arrays) in points.items():
        poh = point_order_hash(name, ids, coords)
        order_hashes[name] = poh
        attrs = {"schema": CACHE_SCHEMA, "point_set": name, "stage": stage,
                 "point_order_sha256": poh, "units": json.dumps(UNITS),
                 "control_checkpoints": json.dumps({k: v["sha256"][:16] for k, v in prov.items()},
                                                    sort_keys=True)}
        path = out_dir / f"points_{name}{'_pilot' if pilot else ''}.h5"
        points_files[name] = {"path": str(path),
                              "sha256": _write_h5_atomic(path, arrays, attrs)}
        print(f"[t1.cache:{stage}] wrote {path} (order {poh[:12]})")
    if points_only:
        _write_manifest(out_dir, stage, manifest, prov, code_hashes, pop_sha,
                        order_hashes, points_files, {}, None, False,
                        "points-only build, no model evaluation", t_start)
        return

    # ---- model evaluation
    print("[t1.cache] loading control models")
    flow, _spatial_ref, potential = adc.load_control_models(control_repo, control_run_dir)
    flow_strict = adc._with_ode_tolerance(flow, adc.ODE_STRICT_RTOL, adc.ODE_STRICT_ATOL)

    arrays_files, timing = {}, {}

    # heldout: f32 default for all rows; x64-strict on a stratified subset
    t0 = time.time()
    with adc._x64(False):
        fn32 = adc._grad_lnf_fn(flow)
        lnf_h32, score_h32 = adc.eval_batched(fn32, jnp.asarray(heldout["eta"], dtype=jnp.float32), batch=batch)
    timing["heldout_f32_s"] = time.time() - t0
    x64_rows_idx = np.array([], dtype=np.int64)
    lnf_h64 = score_h64 = None
    if heldout_x64_rows > 0:
        sub = adc.stratified_rows_by_radius(heldout["eta"],
                                            min(heldout_x64_rows, len(heldout["eta"])))
        t0 = time.time()
        with adc._x64(True):
            fns = adc._grad_lnf_fn(flow_strict)
            lnf_h64, score_h64 = adc.eval_batched(fns, jnp.asarray(heldout["eta"][sub], dtype=jnp.float64), batch=min(batch, 512))
        timing["heldout_x64_subset_s"] = time.time() - t0
        x64_rows_idx = np.asarray(sub, dtype=np.int64)

    # throughput sweep on the pilot/f32 path (re-evaluates existing points)
    if pilot:
        sweep = {}
        for bs in (256, 1024, 4096):
            t0 = time.time()
            with adc._x64(False):
                _l, _s = adc.eval_batched(fn32, jnp.asarray(heldout["eta"], dtype=jnp.float32), batch=bs)
            sweep[str(bs)] = {"seconds": time.time() - t0,
                              "points_per_s": len(heldout["eta"]) / (time.time() - t0)}
        timing["throughput_sweep_f32"] = sweep

    path = out_dir / f"arrays_heldout{'_pilot' if pilot else ''}.h5"
    arrays = {"lnf_f32": lnf_h32.astype(np.float32), "score_f32": score_h32.astype(np.float32)}
    if lnf_h64 is not None:
        arrays["x64_row_index"] = x64_rows_idx
        arrays["lnf_x64strict"] = lnf_h64.astype(np.float64)
        arrays["score_x64strict"] = score_h64.astype(np.float64)
    attrs = {"schema": CACHE_SCHEMA, "point_set": "heldout", "stage": stage,
             "point_order_sha256": order_hashes["heldout"],
             "precision": "f32 default-ODE for all rows; f64 strict-ODE on x64_row_index subset",
             "ode": json.dumps({"default": {"rtol": 1e-4, "atol": 1e-5},
                                "strict": {"rtol": adc.ODE_STRICT_RTOL, "atol": adc.ODE_STRICT_ATOL}}),
             "units": json.dumps(UNITS)}
    arrays_files["heldout"] = {"path": str(path), "sha256": _write_h5_atomic(path, arrays, attrs)}
    rate_f32 = len(heldout["eta"]) / timing["heldout_f32_s"]
    print(f"[t1.cache:{stage}] heldout done ({rate_f32:.0f} pts/s f32)")

    # velocity probes: f32 + x64-strict for all probe points
    t0 = time.time()
    with adc._x64(False):
        lnf_p32, score_p32 = adc.eval_batched(fn32, jnp.asarray(probes["eta"], dtype=jnp.float32), batch=batch)
    with adc._x64(True):
        fns_p = adc._grad_lnf_fn(flow_strict)
        lnf_p64, score_p64 = adc.eval_batched(fns_p, jnp.asarray(probes["eta"], dtype=jnp.float64), batch=min(batch, 512))
    timing["probes_s"] = time.time() - t0
    w_raw = np.exp(lnp_clip(lnf_p64) - probes["log_proposal"])
    w_shell = np.empty_like(w_raw)
    for s in np.unique(probes["shell_idx"]):
        m = probes["shell_idx"] == s
        w_shell[m] = w_raw[m] / w_raw[m].sum()
    ess = {}
    for s in np.unique(probes["shell_idx"]):
        m = probes["shell_idx"] == s
        ww = w_shell[m]
        ess[str(int(s))] = float(ww.sum() ** 2 / np.sum(ww ** 2))
    path = out_dir / f"arrays_velocity_probes{'_pilot' if pilot else ''}.h5"
    arrays = {"lnf_f32": lnf_p32.astype(np.float32), "score_f32": score_p32.astype(np.float32),
              "lnf_x64strict": lnf_p64.astype(np.float64), "score_x64strict": score_p64.astype(np.float64),
              "importance_weight_raw": w_raw, "importance_weight_within_shell": w_shell}
    attrs = {"schema": CACHE_SCHEMA, "point_set": "velocity_probes", "stage": stage,
             "point_order_sha256": order_hashes["velocity_probes"],
             "precision": "f32 default-ODE and f64 strict-ODE for all points",
             "ode": json.dumps({"default": {"rtol": 1e-4, "atol": 1e-5},
                                "strict": {"rtol": adc.ODE_STRICT_RTOL, "atol": adc.ODE_STRICT_ATOL}}),
             "proposal": "diag Gaussian, sigma = population velocity std per axis",
             "sigma_vel": probes["sigma_vel"], "units": json.dumps(UNITS)}
    arrays_files["velocity_probes"] = {"path": str(path), "sha256": _write_h5_atomic(path, arrays, attrs)}
    print(f"[t1.cache:{stage}] probes done; per-shell ESS {ess}")

    # spatial grid: Phi products, f32 + f64, acceleration = -grad_phi
    t0 = time.time()
    phi_model = potential.phi_model
    with adc._x64(False):
        phi_f32, grad_f32, lap_f32 = phi_products(phi_model, grid["q"], np.float32, batch)
    with adc._x64(True):
        phi_f64, grad_f64, lap_f64 = phi_products(phi_model, grid["q"], np.float64, batch)
    timing["grid_s"] = time.time() - t0
    path = out_dir / f"arrays_spatial_grid{'_pilot' if pilot else ''}.h5"
    arrays = {"phi_f32": phi_f32.astype(np.float32), "grad_phi_f32": grad_f32.astype(np.float32),
              "acceleration_f32": (-grad_f32).astype(np.float32), "laplacian_f32": lap_f32.astype(np.float32),
              "phi_f64": phi_f64.astype(np.float64), "grad_phi_f64": grad_f64.astype(np.float64),
              "acceleration_f64": (-grad_f64).astype(np.float64), "laplacian_f64": lap_f64.astype(np.float64)}
    attrs = {"schema": CACHE_SCHEMA, "point_set": "spatial_grid", "stage": stage,
             "point_order_sha256": order_hashes["spatial_grid"],
             "precision": "f32 and f64 (f32 weights, f64 activations) for all points",
             "convention": "grad_phi = grad_q phi; acceleration = -grad_phi; laplacian = trace Hess_q phi (exact)",
             "units": json.dumps(UNITS)}
    arrays_files["spatial_grid"] = {"path": str(path), "sha256": _write_h5_atomic(path, arrays, attrs)}
    print(f"[t1.cache:{stage}] grid done ({len(grid['q']) / timing['grid_s']:.0f} pts/s)")

    # Phi finite-difference spot check (machinery validation, f64)
    phi_fd = phi_fd_spot_check(phi_model, grid["q"])

    # ---- budget extrapolation for the full stage
    n_full_heldout = int(1619615 * 0.25)
    extrapolation = {
        "full_heldout_rows": n_full_heldout,
        "full_probes_rows": int(len(SHELL_R_KPC) * N_PROBE_DIRS * N_SHARED_VEL),
        "full_grid_rows": int(len(SPATIAL_RADII_KPC) * N_GRID_DIRS),
        "estimated_full_heldout_f32_s": n_full_heldout / max(rate_f32, 1e-9),
        "estimated_full_probes_s": (len(SHELL_R_KPC) * N_PROBE_DIRS * N_SHARED_VEL)
        * (timing["probes_s"] / max(len(probes["eta"]), 1)),
    }

    metrics = {
        "stage": stage,
        "n_points": {"heldout": int(len(heldout["eta"])), "velocity_probes": int(len(probes["eta"])),
                     "spatial_grid": int(len(grid["q"]))},
        "timing_s": timing,
        "rates": {"heldout_f32_points_per_s": float(rate_f32)},
        "probe_ess_within_shell": ess,
        "phi_fd_spot_check": phi_fd,
        "extrapolation": extrapolation,
        "elapsed_s": time.time() - t_start,
    }
    _dump_metrics(out_dir / "metrics.json", metrics)
    _write_manifest(out_dir, stage, manifest, prov, code_hashes, pop_sha,
                    order_hashes, points_files, arrays_files, audit, certified,
                    "audit gates all_ok" if certified else
                    "audit gates FAILED: cache is debug material, not for B/C science",
                    t_start)
    print(f"[t1.cache:{stage}] certified={certified}; manifest written to {out_dir}/manifest.json")
    return metrics


def lnp_clip(x):
    """Clip log-values before exponentiation to avoid overflow on
    pathologically low-probability probe points (flagged via the raw weight,
    never silently dropped)."""
    return np.clip(np.asarray(x, dtype=np.float64), -700.0, 700.0)


def phi_products(phi_model, q, dtype, batch):
    """phi, grad_q phi and exact-trace laplacian, vmapped from the control
    snapshot's potential.calc_phi_derivatives (production implementation)."""
    import jax
    import jax.numpy as jnp
    from potential import calc_phi_derivatives
    vf = jax.jit(jax.vmap(calc_phi_derivatives, in_axes=(None, 0)))

    def fn(chunk):
        return vf(phi_model, chunk)
    outs_g, outs_l = [], []
    for i in range(0, len(q), batch):
        chunk = jnp.asarray(q[i:i + batch], dtype=dtype)
        g, lap = fn(chunk)
        outs_g.append(np.asarray(g)); outs_l.append(np.asarray(lap))
    grad = np.concatenate(outs_g, axis=0)
    lap = np.concatenate(outs_l, axis=0)
    phi = np.asarray(phi_values(phi_model, q, dtype, batch))
    return phi, grad, lap


def phi_values(phi_model, q, dtype, batch):
    import jax
    import jax.numpy as jnp
    vf = jax.jit(jax.vmap(phi_model))
    outs = []
    for i in range(0, len(q), batch):
        outs.append(np.asarray(vf(jnp.asarray(q[i:i + batch], dtype=dtype))))
    return np.concatenate(outs, axis=0)


def phi_fd_spot_check(phi_model, q, n_points=8, steps=(1e-2, 3e-3, 1e-3)):
    """Central-FD validation of the Phi gradient machinery (f64): each
    gradient component and the axis-second-difference laplacian must agree
    with autodiff on grid points spanning the radius range."""
    import jax
    import jax.numpy as jnp
    from potential import calc_phi_derivatives
    rows = adc.stratified_rows_by_radius(q, n_points)
    report = {"steps": list(steps), "grad_median_rel": [], "lap_median_rel": []}
    with adc._x64(True):
        scalar = jax.jit(phi_model)
        ad = []
        for i in rows:
            g, lap = calc_phi_derivatives(phi_model, jnp.asarray(q[i]))
            ad.append((np.asarray(g), float(np.asarray(lap))))
        for h in steps:
            g_err, l_err = [], []
            for i, (g_ad, lap_ad) in zip(rows, ad):
                g_fd = adc.central_fd_grad(
                    lambda x: float(np.asarray(scalar(jnp.asarray(x)))),
                    np.asarray(q[i], dtype=np.float64), h)
                g_err.append(np.median(np.abs(g_fd - g_ad) / np.maximum(np.abs(g_ad), 1e-6)))
                lap_fd = 0.0
                for d in range(3):
                    e = np.zeros(3); e[d] = h
                    hi = float(np.asarray(scalar(jnp.asarray(q[i] + e))))
                    lo = float(np.asarray(scalar(jnp.asarray(q[i] - e))))
                    ctr = float(np.asarray(scalar(jnp.asarray(q[i]))))
                    lap_fd += (hi - 2.0 * ctr + lo) / (h * h)
                l_err.append(abs(lap_fd - lap_ad) / max(abs(lap_ad), 1e-6))
            report["grad_median_rel"].append(float(np.median(g_err)))
            report["lap_median_rel"].append(float(np.median(l_err)))
    report["grad_ok"] = bool(min(report["grad_median_rel"]) < 1e-4)
    report["lap_ok"] = bool(min(report["lap_median_rel"]) < 1e-2)
    return report


def _dump_metrics(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def _write_manifest(out_dir, stage, manifest, prov, code_hashes, pop_sha,
                    order_hashes, points_files, arrays_files, audit, certified,
                    certification_note, t_start):
    entry = {
        "schema": CACHE_SCHEMA,
        "stage": stage,
        "created_utc": datetime.datetime.utcnow().isoformat(),
        "control": {
            "training_run": manifest["control"]["training_run"],
            "source_commit": manifest["control"]["source_commit"],
            "single_flow_pair": manifest["control"]["single_flow_pair"],
            "checkpoints": prov,
            "model_code_sha256": code_hashes,
        },
        "population": {"path": str(manifest["population"]["file"]),
                       "sha256": pop_sha, "n": manifest["population"]["n"],
                        "split_rule": "heldout = first int(0.25*n) rows (utils.split_data)"},
        "design": {
            "q_band_edges": Q_BAND_EDGES.tolist(),
            "report_bands_kpc": [list(b) for b in REPORT_BANDS_KPC],
            "shell_r_kpc": SHELL_R_KPC.tolist(),
            "probe_dirs": N_PROBE_DIRS, "shared_velocities": N_SHARED_VEL,
            "probe_seed": PROBE_SEED,
            "spatial_radii_kpc": SPATIAL_RADII_KPC.tolist(),
            "grid_dirs": N_GRID_DIRS,
            "proposal": "shared velocities ~ diag N(0, sigma_pop^2); importance w = exp(lnF_x64strict - log G)",
        },
        "numerics": {
            "ode_default": {"rtol": 1e-4, "atol": 1e-5, "solver": "Tsit5"},
            "ode_strict": {"rtol": adc.ODE_STRICT_RTOL, "atol": adc.ODE_STRICT_ATOL},
            "trace": "no stochastic trace in the evaluated chain (exact jacfwd/Hessian traces); fixed-probe variance check not applicable, recorded",
        },
        "points": points_files,
        "arrays": arrays_files,
        "point_order_sha256": order_hashes,
        "audit": ({"path": str(audit.get("_path", "")), "gates": audit.get("gates")} if audit else None),
        "certification": {"certified": bool(certified), "note": certification_note},
        "units": UNITS,
        "elapsed_s": time.time() - t_start,
    }
    path = Path(out_dir) / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, indent=2, default=float))
    tmp.rename(path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="T1 fixed-point score cache for agents B/C "
                    "(plan docs/nf-score-audit-plan.md sec. 4.2, card T1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--control-repo", required=True,
                        help="root of the control run snapshot (source 5f76faa)")
    parser.add_argument("--control-run-dir", required=True,
                        help="artifacts dir of run 2b32eb04 (contains models/, data/)")
    parser.add_argument("--manifest", default="docs/nf-score-audit/t0-manifest.json")
    parser.add_argument("--population",
                        default="/localdisk/kosmos/my-deep-potential/data/auriga/halo12-clean-smooth.h5")
    parser.add_argument("--out-dir", default=None,
                        help="default runs/nf-score-audit/t1-pilot or t1-cache")
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--heldout-x64-rows", type=int, default=4096,
                        help="x64-strict stratified subset inside heldout (0 disables)")
    parser.add_argument("--audit-json", default=None,
                        help="audit_score_chain.json certifying this cache")
    parser.add_argument("--points-only", action="store_true",
                        help="write the point sets without model evaluation")
    parser.add_argument("--pilot-rows", type=int, default=PILOT_HELDOUT_ROWS)
    args = parser.parse_args()

    out_dir = args.out_dir or f"runs/nf-score-audit/t1-{args.stage}"
    build_cache(args.control_repo, args.control_run_dir, args.manifest,
                args.population, out_dir, args.stage, batch=args.batch,
                heldout_x64_rows=args.heldout_x64_rows, audit_json=args.audit_json,
                points_only=args.points_only, pilot_rows=args.pilot_rows)


if __name__ == "__main__":
    main()
