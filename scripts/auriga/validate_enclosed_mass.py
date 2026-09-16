#!/usr/bin/env python
"""Auriga Halo12 enclosed-mass audit (plan: 2026-09-14, steps 0-5).

Conventions (fixed by the experiment plan, do not change silently):

  q = x / L,  Phi_phys = V^2 * phi          (dimensionless potential phi)
  rho(x)    = V^2 / (4 pi G L^2) * lap_q phi(q)
  M_flux(<r) = r^2 V^2 / (G L) * < n_hat . grad_q phi(r n_hat / L) >_Omega

  Sphere average  <h>_Omega = (1/4 pi) int h dOmega.
  Volume path:  Delta M_rho(r; a) = 4 pi int_a^r s^2 <rho(s,Omega)>_Omega ds
  Flux path:    Delta M_flux(r; a) = M_flux(<r>) - M_flux(<a>)

  Radial integration: trapezoidal rule applied IN log r on the transformed
  integrand g(u) = 4 pi s^3 <rho>(s), u = ln s.  Nodes always include a and
  every truth shell outer edge used for comparison.

  Truth: data/auriga/halo_12_total_density.hdf5, M_cum_total = cumsum over
  shells, bound to shell OUTER edges r_edges[1:].  The primary comparison
  quantity is Delta M(r; 0.5) because the truth cumulative mass excludes
  r < 0.5 kpc.

All numeric thresholds below are ENGINEERING CRITERIA fixed by the plan;
they are not statistical significance levels.

Run from the repo root with JAX_PLATFORMS=cpu for analysis steps.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def _dump_json(path, obj):
    """Strict JSON dump: non-finite values raise instead of writing NaN."""
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))

RUNS = {
    "baseline": REPO / "runs/halo12-baseline",
    "rin2": REPO / "runs/halo12-boundary-rin2",
    "rout65": REPO / "runs/halo12-boundary-rout65",
}
TRUTH_PATH = REPO / "data/auriga/halo_12_total_density.hdf5"

# Gravitational constant in kpc (km/s)^2 / Msun.
# IAU-style value 4.300917270e-6; the plan quotes 4.30091e-6.  Kept
# configurable; all checks below use the value recorded in the output JSON.
G_KPC_KMS2_MSUN = 4.30091e-6

# Engineering criteria (plan-fixed, not significance levels).
TH_PLUMMER_REL = 1e-3       # step 1: spherical mass relative error after refinement
TH_NONSPH_REL = 1e-2        # step 1: anisotropic mass vs independent reference
TH_CONVERGENCE = 1e-2       # step 3: last-two-level change and closure
TH_SHELL_MISMATCH = 5e-3    # step 1 check 4: relative offset that flags mispairing

DF_CKPT_INDEX = 21          # flow-21_model.eqx (latest at audit time)
PHI_CKPT_INDEX = 10         # potential-10_model.eqx (latest at audit time)

BLOCK_ROWS = 65536          # row block for large-array comparisons


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def sha256_file(path, chunk=8 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_or_fail(values, context):
    """Return np.asarray(values); raise if any non-finite entry (plan rule:
    non-finite values must be flagged and fail the check, never dropped)."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        raise ValueError(f"non-finite values in {context} at indices {bad[:5].tolist()}")
    return arr


def sobol_directions(n, scramble_seed):
    """First n points of a scrambled 2-D Sobol sequence mapped to the sphere.

    mu = 2u-1 (uniform in cos(theta)), azimuth = 2 pi v.  Equal weights in
    dOmega, so the sphere average is the plain mean over directions.  A fresh
    scipy Sobol engine per call makes the first n points prefix-nested across
    different n at fixed scramble_seed.
    """
    from scipy.stats import qmc
    sob = qmc.Sobol(d=2, scramble=True, seed=int(scramble_seed))
    uv = sob.random(int(n))
    mu = 2.0 * uv[:, 0] - 1.0
    az = 2.0 * np.pi * uv[:, 1]
    r_sin = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
    return np.column_stack([r_sin * np.cos(az), r_sin * np.sin(az), mu])


# --------------------------------------------------------------------------
# model-side evaluation (shared by analytic checks and real models)
# --------------------------------------------------------------------------

def laplacian_phi_batch(phi_func, q_batch):
    """Laplacian of the dimensionless potential at q (N,3) -> (N,)."""
    import jax
    import jax.numpy as jnp
    from potential import calc_phi_derivatives  # training-time implementation

    def lap_one(q):
        _, lap = calc_phi_derivatives(phi_func, q)
        return lap

    return jax.vmap(lap_one)(jnp.asarray(q_batch))


def grad_phi_dot_n_batch(phi_func, q_batch, n_hat):
    """n_hat . grad_q phi at q (N,3); n_hat (N,3) unit vectors -> (N,)."""
    import jax
    import jax.numpy as jnp

    def dn_one(q, n):
        return jnp.dot(jax.grad(phi_func)(q), n)

    return jax.vmap(dn_one)(jnp.asarray(q_batch), jnp.asarray(n_hat))


def rho_from_phi(phi_func, q_batch, L_kpc, V_kms, G=G_KPC_KMS2_MSUN):
    """Physical density [Msun/kpc^3] from the dimensionless potential."""
    lap = np.asarray(laplacian_phi_batch(phi_func, q_batch))
    return lap * V_kms ** 2 / (4.0 * np.pi * G * L_kpc ** 2)


def m_flux_at_radius(phi_func, r_kpc, dirs, L_kpc, V_kms, G=G_KPC_KMS2_MSUN):
    """M_flux(<r) from the sphere-averaged radial potential gradient.

    dirs: (N,3) unit vectors; q = r * n_hat / L.  Returns a scalar [Msun].
    """
    q = dirs * (r_kpc / L_kpc)
    dn = np.asarray(grad_phi_dot_n_batch(phi_func, q, dirs))
    return float(np.mean(dn)) * r_kpc ** 2 * V_kms ** 2 / (G * L_kpc)


# --------------------------------------------------------------------------
# integration machinery (shared by analytic checks and real models)
# --------------------------------------------------------------------------

def angular_mean_at_nodes(rho_of_q, r_nodes_kpc, dirs, L_kpc, batch=200_000):
    """<rho>_Omega at each radial node, physical units [Msun/kpc^3].

    rho_of_q: callable (N,3) dimensionless q -> (N,) physical density.
    Returns (n_r,) array.  Equal-weight mean over the direction set.
    """
    r_nodes_kpc = np.asarray(r_nodes_kpc, dtype=np.float64)
    out = np.empty(r_nodes_kpc.size, dtype=np.float64)
    for j, r in enumerate(r_nodes_kpc):
        q = dirs * (r / L_kpc)
        vals = np.empty(q.shape[0], dtype=np.float64)
        for i in range(0, q.shape[0], batch):
            vals[i:i + batch] = np.asarray(rho_of_q(q[i:i + batch]))
        out[j] = vals.mean()
    return out


def cumulative_volume_mass(r_nodes_kpc, rho_mean, r_inner_kpc):
    """Delta M_rho(r; r_inner) via trapezoidal rule in log r.

    g(u) = 4 pi s^3 <rho>(s), u = ln s; cumulative trapezoid from the first
    node.  r_nodes must start exactly at r_inner (asserted).
    Returns (n_r,) array with Delta M at each node; value at r_inner is 0.
    """
    r = np.asarray(r_nodes_kpc, dtype=np.float64)
    rho = np.asarray(rho_mean, dtype=np.float64)
    if not np.isclose(r[0], float(r_inner_kpc), rtol=1e-12):
        raise ValueError(
            f"radial nodes must start at r_inner={r_inner_kpc}, got {r[0]}")
    u = np.log(r)
    g = 4.0 * np.pi * r ** 3 * rho
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * np.diff(u) * (g[1:] + g[:-1]))])
    return cum


def cumulative_flux_mass(m_flux_nodes):
    """Delta M_flux(r; r_inner) from M_flux evaluated at nodes (first node
    must be r_inner)."""
    m = np.asarray(m_flux_nodes, dtype=np.float64)
    return m - m[0]


def make_radial_nodes(r_inner, r_outer, per_interval, boundaries=()):
    """Log-spaced nodes from r_inner to r_outer with `per_interval` sub-intervals
    between consecutive boundary radii (boundaries include r_inner/r_outer).
    Guarantees exact inclusion of every boundary value."""
    bnds = sorted(set([float(r_inner), float(r_outer)]
                      + [float(b) for b in boundaries]))
    nodes = []
    for lo, hi in zip(bnds[:-1], bnds[1:]):
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), per_interval + 1))
        nodes.append(edges[:-1])
    nodes.append(np.array([bnds[-1]]))
    return np.concatenate(nodes)


# --------------------------------------------------------------------------
# truth handling
# --------------------------------------------------------------------------

def load_truth(path=TRUTH_PATH):
    import h5py
    with h5py.File(path, "r") as f:
        out = {
            "r_edges": np.asarray(f["r_edges"][:], dtype=np.float64),
            "r_center": np.asarray(f["r_center"][:], dtype=np.float64),
            "M_cum_total": np.asarray(f["M_cum_total"][:], dtype=np.float64),
            "M_shell_total": np.asarray(f["M_shell_total"][:],
                                        dtype=np.float64),
        }
        # per-component cumulative profiles (used by the step-5 truth side)
        for g in ("PartType0", "PartType1", "PartType4"):
            out[g] = {"M_cum": np.asarray(f[f"{g}/M_cum"][:],
                                          dtype=np.float64)}
        return out


def truth_delta_mass(truth, r_inner, r_query_edges=None):
    """Truth Delta M(r; r_inner) for shell OUTER edges r > r_inner.

    Convention: M_cum_total[i] is the mass inside r_edges[i+1], i.e. the
    cumulative array binds to the OUTER edges r_edges[1:].
    Returns (r, dM) arrays.
    """
    r_edges = truth["r_edges"]
    m_cum = truth["M_cum_total"]
    if not (r_edges[0] <= r_inner <= r_edges[-1]):
        raise ValueError(f"r_inner={r_inner} outside truth edge range")
    i_inner = int(np.searchsorted(r_edges, r_inner))  # first edge >= r_inner
    if not np.isclose(r_edges[i_inner], r_inner):
        raise ValueError(f"r_inner={r_inner} is not a truth shell edge")
    m_inner = m_cum[i_inner - 1] if i_inner >= 1 else 0.0
    mask = r_edges > r_inner
    idx_out = np.flatnonzero(mask)
    if idx_out.size == 0 or idx_out[0] == 0:
        raise ValueError("no truth shell edge lies beyond r_inner")
    r_out = r_edges[idx_out]
    dM = m_cum[idx_out - 1] - m_inner
    return r_out, dM


# --------------------------------------------------------------------------
# analytic test problems (step 1)
# --------------------------------------------------------------------------

def plummer_phi_func(M_msun=1.0e12, b_kpc=10.0):
    """Dimensionless Plummer potential phi(q) = -GM/(V^2 sqrt((Lq)^2+b^2)).

    Uses the module constants L=10 kpc, V=100 km/s (plan values); the caller
    must pass the same L/V used everywhere else.
    """
    L_kpc, V_kms = 10.0, 100.0

    def phi(q):
        import jax.numpy as jnp
        r2 = jnp.dot(q, q) * L_kpc ** 2
        return -G_KPC_KMS2_MSUN * M_msun / (V_kms ** 2 * jnp.sqrt(r2 + b_kpc ** 2))

    return phi, M_msun, b_kpc


def plummer_mass_inside(r_kpc, M_msun=1.0e12, b_kpc=10.0):
    r = np.asarray(r_kpc, dtype=np.float64)
    return M_msun * r ** 3 / (r ** 2 + b_kpc ** 2) ** 1.5


def plummer_density(r_kpc, M_msun=1.0e12, b_kpc=10.0):
    r = np.asarray(r_kpc, dtype=np.float64)
    return (3.0 * M_msun / (4.0 * np.pi * b_kpc ** 3)) * (
        1.0 + r ** 2 / b_kpc ** 2) ** (-2.5)


def anisotropic_gauss_density(q_batch, L_kpc=10.0,
                              comps=((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))):
    """Physical density [Msun/kpc^3] of two coaxial anisotropic Gaussians.

    Component i: M_i (2 pi)^{-3/2} a_i^{-2} c_i^{-1}
                 exp(-((x^2+y^2)/a_i^2 + z^2/c_i^2)/2)
    with axis ratio c_i/a_i != 1 so direction medians are biased.
    """
    q = np.asarray(q_batch, dtype=np.float64)
    x, y, z = q[:, 0] * L_kpc, q[:, 1] * L_kpc, q[:, 2] * L_kpc
    out = np.zeros(q.shape[0], dtype=np.float64)
    for M, a, c in comps:
        norm = M * (2.0 * np.pi) ** (-1.5) / (a ** 2 * c)
        out += norm * np.exp(-0.5 * ((x ** 2 + y ** 2) / a ** 2 + z ** 2 / c ** 2))
    return out


def anisotropic_gauss_mass(comps=((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))):
    return float(sum(M for M, _, _ in comps))


def anisotropic_gauss_sphere_mean(r_kpc, comps=((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0)),
                                  n_mu=200):
    """Independent high-accuracy <rho>_Omega(s): Gauss-Legendre in mu only,
    because both components are axisymmetric."""
    from numpy.polynomial.legendre import leggauss
    mu, w = leggauss(int(n_mu))
    r = np.asarray(r_kpc, dtype=np.float64)[:, None]
    s_sin = r * np.sqrt(1.0 - mu[None, :] ** 2)  # cylindrical R
    zz = r * mu[None, :]
    mean = np.zeros(r.size, dtype=np.float64)
    for M, a, c in comps:
        norm = M * (2.0 * np.pi) ** (-1.5) / (a ** 2 * c)
        vals = norm * np.exp(-0.5 * (s_sin ** 2 / a ** 2 + zz ** 2 / c ** 2))
        mean += 0.5 * (vals * w[None, :]).sum(axis=1)
    return mean


# --------------------------------------------------------------------------
# step 0: manifest
# --------------------------------------------------------------------------

def _load_phi_smoke(run_dir):
    """Load the audit's Phi checkpoint once to prove it is loadable; return
    (callable phi(q), param dtype string).  The on-disk checkpoint is f32,
    so the load/eval runs with the default dtype (x64 off) inside this
    function and restores the previous setting afterwards."""
    import jax
    import jax.numpy as jnp
    import fit_all
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        phi = fit_all.load_potential(run_dir / "models" / "Phi",
                                     checkpoint_index=PHI_CKPT_INDEX)
        leaf_dtype = str(jax.tree_util.tree_leaves(
            eqx_filter_arrays(phi))[0].dtype)
        val = float(phi.phi_model(jnp.array([0.1, 0.1, 0.1])))
        return (lambda q: phi.phi_model(jnp.asarray(q)), leaf_dtype, val)
    finally:
        jax.config.update("jax_enable_x64", prev)


def eqx_filter_arrays(module):
    import equinox as eqx
    return eqx.filter(module, eqx.is_array)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def _blockwise_equal(ds_a, ds_b, rows):
    """Compare two h5py datasets row-block-wise; return dict summary."""
    if ds_a.shape != ds_b.shape:
        return {"equal": False, "reason": f"shape {ds_a.shape} vs {ds_b.shape}"}
    n = ds_a.shape[0]
    ndiff = 0
    first = None
    for i in range(0, n, BLOCK_ROWS):
        a = np.asarray(ds_a[i:i + BLOCK_ROWS])
        b = np.asarray(ds_b[i:i + BLOCK_ROWS])
        ne = int(np.count_nonzero(a != b))
        if ne and first is None:
            flat = (a != b)
            if flat.ndim > 1:
                flat = flat.any(axis=1)
            j = int(np.flatnonzero(flat)[0])
            first = i + j
        ndiff += ne
    return {"equal": ndiff == 0, "n_diff_entries": ndiff,
            "first_diff_row": first, "rows_compared": int(n)}


def compare_h5_files(path_a, path_b, datasets):
    import h5py
    out = {}
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        for name in datasets:
            if name in fa and name in fb:
                out[name] = _blockwise_equal(fa[name], fb[name], BLOCK_ROWS)
            else:
                out[name] = {"equal": False, "reason": "missing dataset"}
    return out


def step0_manifest(audit_dir):
    import h5py
    import jax
    import equinox

    t0 = time.time()
    audit_dir = Path(audit_dir)
    (audit_dir / "step0").mkdir(parents=True, exist_ok=True)

    manifest = {
        "plan": "2026-09-14 Halo12 enclosed-mass audit, steps 0-5",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": str(REPO),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status_short": _git("status", "--short"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "equinox": equinox.__version__,
            "numpy": np.__version__,
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
        },
        "constants": {"G_kpc_kms2_msun": G_KPC_KMS2_MSUN,
                      "L_kpc_expected": 10.0, "V_kms_expected": 100.0},
        "checkpoints": {"df_index": DF_CKPT_INDEX, "phi_index": PHI_CKPT_INDEX},
        "runs": {},
        "df_checkpoint_comparison": {},
        "df_gradients_comparison": {},
        "split_availability": {},
    }

    # source.diff: tracked working-tree modifications (frozen for this audit)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout
    (audit_dir / "step0" / "source.diff").write_text(diff)
    untracked = [l for l in _git("ls-files", "--others", "--exclude-standard").splitlines() if l]
    (audit_dir / "step0" / "untracked_files.txt").write_text("\n".join(untracked))
    manifest["git"]["n_untracked"] = len(untracked)

    input_paths = {}
    for name, run in RUNS.items():
        info = {"run_dir": str(run)}
        opt = run / "options.json"
        info["options_sha256"] = sha256_file(opt) if opt.exists() else None
        if opt.exists():
            with open(opt) as f:
                o = json.load(f)
            info["options_keys"] = {
                "df.seed": o["df"]["seed"],
                "flow_sampling.seed": o["flow_sampling"]["seed"],
                "flow_sampling.n_samples": o["flow_sampling"]["n_samples"],
                "Phi.seed": o["Phi"]["seed"],
                "validation_frac": o["df"]["validation_frac"],
            }
        # input file from run_all.sh (full script frozen: it carries the
        # per-run configuration that options.json cannot distinguish)
        input_path = None
        script = run / "run_all.sh"
        if script.exists():
            info["run_all_sh_sha256"] = sha256_file(script)
            info["run_all_sh_text"] = script.read_text()
            for line in script.read_text().splitlines():
                if line.startswith("INPUT="):
                    input_path = REPO / line.split("=", 1)[1].strip()
        info["input_h5"] = str(input_path) if input_path else None
        if input_path and input_path.exists():
            info["input_h5_sha256"] = sha256_file(input_path)
            with h5py.File(input_path, "r") as f:
                info["input_attrs"] = {
                    k: (v.tolist() if hasattr(v, "tolist") else str(v))
                    for k, v in f.attrs.items()
                    if k in ("length_scale_kpc", "velocity_scale_kms",
                             "phi_r_min_kpc", "phi_r_max_kpc", "train_r_max_kpc",
                             "shuffle_seed", "weighting", "n_selected", "n_source")
                }
                info["eta_attrs"] = {k: str(v) for k, v in f["eta"].attrs.items()}
        input_paths[name] = input_path

        # checkpoints actually used
        phi_dir, df_dir = run / "models/Phi", run / "models/df/flow"
        phi_file = phi_dir / f"potential-{PHI_CKPT_INDEX}_model.eqx"
        df_file = df_dir / f"flow-{DF_CKPT_INDEX}_model.eqx"
        info["phi_checkpoint"] = {"path": str(phi_file),
                                  "sha256": sha256_file(phi_file) if phi_file.exists() else None}
        info["df_checkpoint"] = {"path": str(df_file),
                                 "sha256": sha256_file(df_file) if df_file.exists() else None}
        meta = phi_dir / "potential-metadata.json"
        if meta.exists():
            with open(meta) as f:
                md = json.load(f)
            fs = md.get("frameshift_params") or {}
            info["phi_metadata"] = {
                "num_parameters": md.get("num_parameters"),
                "frameshift_all_zero": all(float(fs.get(k, 0)) == 0 for k in
                                           ("omega", "r0", "v0_x", "v0_y", "v0_z")),
                "phi_params_type": (md.get("phi_params") or {}).get("type"),
            }
        # split availability (plan step 0 item 6)
        splits = {}
        if input_path and input_path.exists():
            with h5py.File(input_path, "r") as f:
                splits["input_has_eta_train_val"] = bool(
                    "eta_train" in f and "eta_val" in f)
        splits["run_dir_split_files"] = sorted(
            p.name for p in run.rglob("*") if p.is_file() and
            any(k in p.name.lower() for k in ("split", "train_idx", "val_idx")))
        manifest["split_availability"][name] = splits
        # pipeline completion state
        plog = run / "pipeline.log"
        info["pipeline_log_last_line"] = (plog.read_text().splitlines()[-1]
                                          if plog.exists() else None)
        manifest["runs"][name] = info

    # DF checkpoint hash comparison across runs (first divergence index).
    # Covers BOTH training stages: flow_pos_only-* (spatial) and flow-*
    # (conditional velocity).  Loss-JSON hashes are stored alongside so the
    # divergence analysis is reproducible from the manifest alone.
    df_hashes = {n: {} for n in RUNS}
    loss_hashes = {n: {} for n in RUNS}
    for name, run in RUNS.items():
        d = run / "models/df/flow"
        for f in sorted(d.glob("*_model.eqx")):
            idx = int(f.stem.split("-")[1].split("_")[0])
            df_hashes[name][idx] = sha256_file(f)
            lf = f.with_name(f.stem.replace("_model", "") + "_loss.json")
            if lf.exists():
                loss_hashes[name][idx] = sha256_file(lf)
    idxs = sorted(set().union(*[set(h) for h in df_hashes.values()]))
    first_div = None
    table = {}
    for i in idxs:
        hs = {n: df_hashes[n].get(i) for n in RUNS}
        same = len(set(hs.values())) == 1
        table[i] = {"same": same,
                    "hashes": {n: h[:12] for n, h in hs.items()},
                    "loss_sha256": {n: (loss_hashes[n].get(i) or "")[:12]
                                    for n in RUNS}}
        if not same and first_div is None:
            first_div = i
    # persist the loss evidence at the first divergence (epoch-level values)
    first_div_loss = None
    first_div_prefix = None
    if first_div is not None:
        import json as _json
        first_div_loss = {}
        for n, run in RUNS.items():
            d = run / "models/df/flow"
            lf = next(d.glob(f"*-{first_div}_loss.json"), None)
            if lf is not None:
                first_div_prefix = lf.stem.split(f"-{first_div}")[0]
                with open(lf) as fh:
                    lh = _json.load(fh)
                first_div_loss[n] = {
                    k: [float(x) for x in v[:3]] + [float(v[-1])]
                    for k, v in lh.items() if isinstance(v, list) and v
                }
    manifest["df_checkpoint_comparison"] = {
        "first_divergent_index": first_div,
        "first_divergent_prefix": first_div_prefix,
        "note": "identical hash => identical serialized weights; different hash "
                "does not imply functionally different DF (float nondeterminism)",
        "per_index": table,
        "loss_history_at_first_divergence": first_div_loss,
    }

    # DF gradient sample comparison (blockwise, avoids loading full arrays)
    datasets = ("eta", "weights", "particle_id", "source_index", "mass")
    input_comparison = {}
    names = list(RUNS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            input_comparison[f"{a}_vs_{b}"] = compare_h5_files(
                input_paths[a], input_paths[b], datasets)
    (audit_dir / "step0" / "input_comparison.json").write_text(
        json.dumps(input_comparison, indent=2))

    grad_comparison = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ga = RUNS[a] / "data/df_gradients.h5"
            gb = RUNS[b] / "data/df_gradients.h5"
            if ga.exists() and gb.exists():
                grad_comparison[f"{a}_vs_{b}"] = compare_h5_files(
                    ga, gb, ("eta", "dlnf_deta", "lnf"))
    manifest["df_gradients_comparison"] = grad_comparison

    manifest["step0_elapsed_s"] = round(time.time() - t0, 1)

    # plan gate for entering step 1: checkpoints loadable, units consistent,
    # coordinate center consistent.
    gate = {"units_consistent": {}, "checkpoint_load_smoke": {}}
    for name, run in RUNS.items():
        attrs = manifest["runs"][name].get("input_attrs", {})
        gate["units_consistent"][name] = (
            float(attrs.get("length_scale_kpc", -1)) == manifest["constants"]["L_kpc_expected"]
            and float(attrs.get("velocity_scale_kms", -1)) == manifest["constants"]["V_kms_expected"])
    try:
        phi_fn, phi_dtype, phi_val = _load_phi_smoke(RUNS["baseline"])
        gate["checkpoint_load_smoke"] = {
            "baseline_phi_loadable": True,
            "phi_checkpoint_param_dtype": phi_dtype,
            "phi_test_value_q0.1": phi_val,
            "note": "checkpoint params are stored in this dtype; with "
                    "jax_enable_x64 the host accumulates in f64 but the "
                    "network weights remain at their stored dtype",
        }
    except Exception as e:
        gate["checkpoint_load_smoke"] = {"baseline_phi_loadable": False,
                                         "error": repr(e)}
    # coordinate center: the three inputs' particle arrays were compared
    # blockwise above and are identical, so any centering choice is shared.
    inputs_identical = all(
        all(v.get("equal", False) for v in input_comparison[f"{a}_vs_{b}"].values())
        for a, b in ((names[0], names[1]), (names[0], names[2])))
    gate["coordinate_center_consistent"] = bool(inputs_identical)
    gate["ready_for_step1"] = bool(
        all(gate["units_consistent"].values())
        and gate["checkpoint_load_smoke"].get("baseline_phi_loadable", False)
        and gate["coordinate_center_consistent"])
    manifest["step0_gate"] = gate

    (audit_dir / "step0" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False))
    print(f"step0 done -> {audit_dir / 'step0' / 'manifest.json'} "
          f"(ready_for_step1={gate['ready_for_step1']})")
    return manifest


# --------------------------------------------------------------------------
# step 1: analytic checks
# --------------------------------------------------------------------------

def _check_plummer():
    """Check 1: spherical Plummer through the full potential->density path."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    L, V = 10.0, 100.0
    phi, M, b = plummer_phi_func()
    r_inner, r_outer = 0.5, 70.0
    results = {}

    # volume path with refinement (per-direction density via laplacian)
    dirs = sobol_directions(256, scramble_seed=0)
    for n_nodes in (100, 400, 800):
        r_nodes = make_radial_nodes(r_inner, r_outer, per_interval=n_nodes)
        q = dirs[None, :, :] * (r_nodes[:, None, None] / L)  # (n_r, N, 3)
        lap = np.asarray(laplacian_phi_batch(
            phi, jnp.asarray(q.reshape(-1, 3)))).reshape(q.shape[:2])
        rho_mean = finite_or_fail(
            lap.mean(axis=1) * V ** 2 / (4.0 * np.pi * G_KPC_KMS2_MSUN * L ** 2),
            f"plummer rho_mean nodes={n_nodes}")
        dM = cumulative_volume_mass(r_nodes, rho_mean, r_inner)
        # analytic reference at the same nodes (exclude the inner node where
        # both sides are exactly zero)
        dM_true = plummer_mass_inside(r_nodes) - plummer_mass_inside(r_inner)
        body = dM_true > 0
        rel = np.abs(dM[body] / dM_true[body] - 1.0)
        results[f"volume_nodes_{n_nodes}"] = {
            "max_rel_err": float(rel.max()),
            "rel_err_at_70kpc": float(rel[-1]),
        }

    # flux path
    m_flux = np.array([m_flux_at_radius(phi, r, dirs, L, V)
                       for r in r_nodes])
    dM_flux = cumulative_flux_mass(finite_or_fail(m_flux, "plummer m_flux"))
    dM_true = plummer_mass_inside(r_nodes) - plummer_mass_inside(r_inner)
    body = dM_true > 0
    rel_f = np.abs(dM_flux[body] / dM_true[body] - 1.0)
    results["flux_nodes_800"] = {"max_rel_err": float(rel_f.max()),
                                 "rel_err_at_70kpc": float(rel_f[-1])}

    # units: central density analytic vs machinery at r->0 (small r node)
    r_tiny = np.array([1e-3])
    q_tiny = jnp.asarray(np.tile(r_tiny[0] / L * np.array([1.0, 0.0, 0.0]), (1, 1)))
    lap0 = float(np.asarray(laplacian_phi_batch(phi, q_tiny))[0])
    rho0 = lap0 * V ** 2 / (4.0 * np.pi * G_KPC_KMS2_MSUN * L ** 2)
    rho0_true = plummer_density(0.0)
    results["central_density_rel_err"] = float(abs(rho0 / rho0_true - 1.0))

    ok = (results["volume_nodes_800"]["max_rel_err"] < TH_PLUMMER_REL
          and results["flux_nodes_800"]["max_rel_err"] < TH_PLUMMER_REL
          and results["central_density_rel_err"] < TH_PLUMMER_REL)
    return {"check": "plummer_spherical", "status": "pass" if ok else "fail",
            "threshold": TH_PLUMMER_REL, "results": results}


def _check_nonspherical():
    """Check 2: angular MEAN recovers total mass of an anisotropic field;
    direction MEDIAN does not.  Independent reference: Gauss-Legendre."""
    L = 10.0
    comps = ((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))
    m_true = anisotropic_gauss_mass(comps)
    r_inner, r_outer = 0.5, 70.0
    r_nodes = make_radial_nodes(r_inner, r_outer, per_interval=200)

    # machinery path: Sobol directions + log-r trapezoid (pure numpy density)
    dirs = sobol_directions(1024, scramble_seed=0)
    q = dirs[None, :, :] * (r_nodes[:, None, None] / L)
    rho_dirs = anisotropic_gauss_density(q.reshape(-1, 3), L, comps).reshape(
        q.shape[:2])
    rho_mean = finite_or_fail(rho_dirs.mean(axis=1), "anisotropic rho_mean")
    dM = cumulative_volume_mass(r_nodes, rho_mean, r_inner)
    rel_mean = abs(dM[-1] / m_true - 1.0)

    # per-direction enclosed mass: median vs mean at r_outer
    # (same log-r trapezoid as cumulative_volume_mass, kept per direction;
    #  rho_dirs is (n_r, N_dir) because q was built as (n_r, N_dir, 3))
    u = np.log(r_nodes)
    g = 4.0 * np.pi * r_nodes[None, :] ** 3 * rho_dirs.T     # (N_dir, n_r)
    inc = 0.5 * np.diff(u)[None, :] * (g[:, 1:] + g[:, :-1])
    m_per_dir = np.concatenate([np.zeros((dirs.shape[0], 1)),
                                np.cumsum(inc, axis=1)], axis=1)
    m_mean = m_per_dir[:, -1].mean()
    m_median = float(np.median(m_per_dir[:, -1]))
    m_p16, m_p84 = np.percentile(m_per_dir[:, -1], [16, 84])

    # independent reference: Gauss-Legendre mu (200) x log-Gauss nodes?  Use
    # plain Gauss-Legendre in s on [0.5,70] with 400 nodes (independent of the
    # machinery's log-trapezoid).
    from numpy.polynomial.legendre import leggauss
    s_nodes, ws = leggauss(400)
    s_kpc = 0.5 * (r_outer - r_inner) * s_nodes + 0.5 * (r_outer + r_inner)
    w_s = 0.5 * (r_outer - r_inner) * ws
    mean_ref = anisotropic_gauss_sphere_mean(s_kpc, comps, n_mu=200)
    m_ref = float(np.sum(4.0 * np.pi * s_kpc ** 2 * mean_ref * w_s))
    rel_ref = abs(m_ref / m_true - 1.0)

    ok = (rel_mean < TH_NONSPH_REL and rel_ref < TH_NONSPH_REL
          and abs(dM[-1] / m_ref - 1.0) < TH_NONSPH_REL)
    return {"check": "anisotropic_gaussians", "status": "pass" if ok else "fail",
            "threshold": TH_NONSPH_REL,
            "total_mass_msun": m_true,
            "machinery_rel_err_mean": float(rel_mean),
            "independent_ref_rel_err": float(rel_ref),
            "machinery_vs_independent_ref_rel": float(abs(dM[-1] / m_ref - 1.0)),
            "independent_ref_mass": m_ref,
            "per_direction_at_70kpc": {
                "mean_rel_err": float(m_mean / m_true - 1.0),
                "median_rel_err": float(m_median / m_true - 1.0),
                "p16": float(m_p16), "p84": float(m_p84)},
            "note": "mean of per-direction enclosed mass equals the angular-mean "
                    "integral; median has no such guarantee (shown numerically)"}


def _check_constant_linear():
    """Check 3: adding a constant or a linear term to the potential must not
    change the density; the linear term's net sphere flux must vanish."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    L, V = 10.0, 100.0
    phi0, M, b = plummer_phi_func()
    c0, gvec = 123.0, (0.3, -0.7, 0.15)

    def phi_pert(q):
        return phi0(q) + jnp.asarray(c0) + (gvec[0] * q[0] + gvec[1] * q[1]
                                            + gvec[2] * q[2])

    dirs = sobol_directions(1024, scramble_seed=0)
    r_test = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 70.0])
    # density equality at sample points
    rng = np.random.default_rng(1)
    q_rand = rng.normal(size=(512, 3)) * 3.0
    lap0 = np.asarray(laplacian_phi_batch(phi0, jnp.asarray(q_rand)))
    lap1 = np.asarray(laplacian_phi_batch(phi_pert, jnp.asarray(q_rand)))
    scale = np.abs(lap0).max()
    rho_rel_diff = float(np.abs(lap1 - lap0).max() / max(scale, 1e-30))

    # net flux of the linear term over the full sphere (should be ~0)
    lin_flux = []
    for r in r_test:
        q = dirs * (r / L)
        dn = np.asarray(grad_phi_dot_n_batch(
            lambda qq: gvec[0] * qq[0] + gvec[1] * qq[1] + gvec[2] * qq[2],
            jnp.asarray(q), jnp.asarray(dirs)))
        lin_flux.append(float(np.mean(dn)) * r ** 2 * V ** 2 / (G_KPC_KMS2_MSUN * L))
    lin_flux = np.array(lin_flux)
    # scale of the Plummer flux at the same radii for a relative comparison
    plum_flux = np.array([m_flux_at_radius(phi0, r, dirs, L, V) for r in r_test])
    rel = np.abs(lin_flux) / np.maximum(np.abs(plum_flux), 1e-30)

    ok = rho_rel_diff < 1e-12 and rel.max() < TH_PLUMMER_REL
    return {"check": "constant_linear_invariance",
            "status": "pass" if ok else "fail",
            "max_rel_density_diff": rho_rel_diff,
            "linear_net_flux_over_plummer_flux_max": float(rel.max()),
            "linear_net_flux_msun": lin_flux.tolist(),
            "radii_kpc": r_test.tolist()}


def _check_shell_truth_pairing():
    """Check 4: cumulative truth mass binds to r_edges[1:]; pairing with
    r_center must be detectable."""
    truth = load_truth()
    # synthetic shell masses from the Plummer profile on the truth edges
    m_shell = np.diff(plummer_mass_inside(truth["r_edges"]))
    m_cum = np.cumsum(m_shell)  # binds to r_edges[1:]
    i = 20  # arbitrary interior shell
    edge = truth["r_edges"][i + 1]
    center = truth["r_center"][i]
    # mass you would wrongly infer at radius `center` if you read the
    # edge-bound cumulative value: compare M(<center) with M(<edge)
    m_edge = plummer_mass_inside(edge)
    m_center = plummer_mass_inside(center)
    rel = abs(m_center - m_edge) / m_edge
    flagged = rel > TH_SHELL_MISMATCH
    return {"check": "shell_truth_pairing", "status": "pass" if flagged else "fail",
            "note": "pass means the mispairing IS detected (status refers to the "
                    "detection contract, not to the model)",
            "shell_index": i, "edge_kpc": float(edge), "center_kpc": float(center),
            "relative_offset_if_mispaired": float(rel),
            "threshold": TH_SHELL_MISMATCH}


def step1_analytic(audit_dir):
    audit_dir = Path(audit_dir)
    (audit_dir / "step1").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "jax_x64_forced": True,
           "checks": []}
    for fn in (_check_plummer, _check_nonspherical,
               _check_constant_linear, _check_shell_truth_pairing):
        try:
            res = fn()
        except Exception as e:  # record, never silently drop
            res = {"check": fn.__name__, "status": "fail", "exception": repr(e)}
        out["checks"].append(res)
        print(f"{res['check']}: {res['status']}")
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["all_pass"] = all(c["status"] == "pass" for c in out["checks"])
    _dump_json(audit_dir / "step1" / "analytic_checks.json", out)
    print(f"step1 done -> {audit_dir / 'step1' / 'analytic_checks.json'}")
    return out


# --------------------------------------------------------------------------
# step 2: legacy 2x2 ablation (median-vs-mean, center-vs-edge)
# --------------------------------------------------------------------------

# Exact replication of the legacy verify_enclosed_mass.py sampling (needed
# so curve A reproduces the old JSON bit-for-bit up to explained tolerance).
def legacy_directions(n_dir=256, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_dir, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def legacy_radial_edges(r_min=0.5, r_max=70.0, n_shell=40):
    return np.logspace(np.log10(r_min), np.log10(r_max), n_shell)


def _per_direction_enclosed(r_nodes, rho_dirs):
    """Per-direction cumulative enclosed mass via log-r trapezoid.
    rho_dirs: (n_dir, n_r).  Returns (n_dir, n_r)."""
    u = np.log(r_nodes)
    g = 4.0 * np.pi * r_nodes[None, :] ** 3 * rho_dirs
    inc = 0.5 * np.diff(u)[None, :] * (g[:, 1:] + g[:, :-1])
    return np.concatenate([np.zeros((rho_dirs.shape[0], 1)),
                           np.cumsum(inc, axis=1)], axis=1)


def step2_ablation(audit_dir, run="baseline"):
    """2x2 ablation on the legacy baseline samples.

    Curves (plan step 2):
      A: legacy replication   -> direction MEDIAN, truth paired at r_center
      B: fixed statistic only -> direction MEAN,   truth paired at r_center
      C: fixed radius only    -> direction MEDIAN, truth paired at outer edge
      D: both fixed           -> direction MEAN,   truth paired at outer edge
    A/B truth pairing is the DELIBERATELY KEPT legacy error, for attribution
    only; they are not scientific mass results.
    """
    import h5py
    audit_dir = Path(audit_dir)
    out_dir = audit_dir / "step2"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import jax
    import jax.numpy as jnp
    import fit_all

    run_dir = RUNS[run]
    # Load with the default dtype first (checkpoint params are f32 on disk);
    # then run the legacy numerics with x64 off, exactly like the old script.
    prev_x64 = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        phi = fit_all.load_potential(run_dir / "models" / "Phi",
                                     checkpoint_index=PHI_CKPT_INDEX)
        phi_func = phi.phi_model

        # Legacy sampling: f32 q vectors, x64 OFF (matches the old script's
        # numerics; recorded in the output).
        L, V = 10.0, 100.0
        dirs = legacy_directions()
        r_nodes = legacy_radial_edges()
        q = jnp.asarray((dirs[None, :, :] * (r_nodes[:, None, None] / L))
                        .reshape(-1, 3), dtype=jnp.float32)
        lap = np.asarray(laplacian_phi_batch(
            phi_func, q).reshape(len(r_nodes), dirs.shape[0]))
        rho = lap.T * V ** 2 / (4.0 * np.pi * G_KPC_KMS2_MSUN * L ** 2)  # (n_dir, n_r)
        rho = finite_or_fail(rho.astype(np.float64), "step2 rho")
    finally:
        jax.config.update("jax_enable_x64", prev_x64)

    m_per_dir = _per_direction_enclosed(r_nodes, rho)   # (n_dir, n_r)

    truth = load_truth()
    r_center, r_edges, m_cum = (truth["r_center"], truth["r_edges"],
                                truth["M_cum_total"])
    # common coverage: complete shells whose outer edge <= model r_max and
    # whose inner edge >= model r_min; compare at OUTER edges only
    mask = (r_edges[1:] <= r_nodes[-1] * 1.001) & (r_edges[:-1] >= r_nodes[0] - 1e-9)
    # legacy compared at r_center for all truth shells with r_center<=70;
    # keep the same shell set for all four curves (complete shells only).
    idx = np.flatnonzero(mask)
    r_edge_cmp = r_edges[1:][idx]          # outer edges (fixed-radius curves)
    r_center_cmp = r_center[idx]           # shell centres (legacy pairing)
    m_truth_edge = m_cum[idx]              # M_cum binds to outer edges

    def interp_curve(stat, x_query):
        """stat: (n_r,) model enclosed mass on r_nodes -> values at the
        query radii.  The legacy script used np.interp in r (linear) with
        x = r_center; curve A must reproduce it.  Curves C/D read the model
        at the truth OUTER edge instead (that is the radius fix being
        ablated); B shares A's readout so the 2x2 isolates each factor."""
        return np.interp(x_query, r_nodes, stat)

    m_med = np.median(m_per_dir, axis=0)
    m_mean = m_per_dir.mean(axis=0)
    p16 = np.percentile(m_per_dir, 16, axis=0)
    p84 = np.percentile(m_per_dir, 84, axis=0)

    curves = {
        "A_legacy": {"stat": "median", "pairing": "r_center",
                     "m_model": interp_curve(m_med, r_center_cmp),
                     "r_cmp": r_center_cmp},
        "B_stat_only": {"stat": "mean", "pairing": "r_center",
                        "m_model": interp_curve(m_mean, r_center_cmp),
                        "r_cmp": r_center_cmp},
        "C_radius_only": {"stat": "median", "pairing": "r_edge",
                          "m_model": interp_curve(m_med, r_edge_cmp),
                          "r_cmp": r_edge_cmp},
        "D_both_fixed": {"stat": "mean", "pairing": "r_edge",
                         "m_model": interp_curve(m_mean, r_edge_cmp),
                         "r_cmp": r_edge_cmp},
    }
    for c in curves.values():
        c["ratio"] = c["m_model"] / m_truth_edge   # truth always at outer edge

    # attribution: deltas of the ratio curves
    rA, rB, rC, rD = (curves[k]["ratio"] for k in
                      ("A_legacy", "B_stat_only", "C_radius_only", "D_both_fixed"))
    attribution = {
        "B_minus_A": (rB - rA).tolist(),
        "C_minus_A": (rC - rA).tolist(),
        "D_minus_A": (rD - rA).tolist(),
        "interaction_D_minus_B_minus_C_plus_A": (rD - rB - rC + rA).tolist(),
        "max_abs": {
            "B_minus_A": float(np.abs(rB - rA).max()),
            "C_minus_A": float(np.abs(rC - rA).max()),
            "D_minus_A": float(np.abs(rD - rA).max()),
            "interaction": float(np.abs(rD - rB - rC + rA).max()),
        },
    }

    # anchor: compare curve A against the persisted legacy JSON
    legacy_json = RUNS["rin2"] / "analysis" / "enclosed_mass_comparison.json"
    with open(legacy_json) as f:
        legacy = json.load(f)
    leg = legacy["runs"][run]
    leg_r = np.asarray(legacy["truth"]["r_center"])
    leg_ratio = np.asarray(leg["ratio_median"])
    # legacy r_cmp includes shells below the model grid start? it starts at
    # r_center[0]=0.529 > 0.5 so all are covered; match by radius value
    anchor = {"legacy_json": str(legacy_json), "matched": 0, "max_ratio_diff": None}
    diffs = []
    for r0, ratio_old in zip(leg_r, leg_ratio):
        j = int(np.argmin(np.abs(curves["A_legacy"]["r_cmp"] - r0)))
        if abs(curves["A_legacy"]["r_cmp"][j] - r0) < 1e-6:
            anchor["matched"] += 1
            diffs.append(abs(curves["A_legacy"]["ratio"][j] - ratio_old))
    if diffs:
        anchor["max_ratio_diff"] = float(max(diffs))

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": run,
        "phi_checkpoint": str(run_dir / "models/Phi" /
                              f"potential-{PHI_CKPT_INDEX}_model.eqx"),
        "numerics": {"jax_enable_x64_during_model_eval": False,
                     "q_dtype": "float32",
                     "note": "legacy numerics replicated on purpose; the "
                             "refined convergence study is step 3"},
        "n_directions": int(dirs.shape[0]),
        "n_radial_nodes": int(r_nodes.size),
        "r_nodes_kpc": r_nodes.tolist(),
        "comparison_shells": {
            "n": int(idx.size),
            "outer_edges_kpc": r_edge_cmp.tolist(),
            "centers_kpc_legacy_pairing": r_center_cmp.tolist(),
            "truth_delta_from_zero": False,
            "note": "truth is M_cum_total at outer edges; A/B pair the model "
                    "value at r_center with it (deliberate legacy error)",
        },
        "curves": {k: {"stat": v["stat"], "pairing": v["pairing"],
                       "r_cmp": v["r_cmp"].tolist(),
                       "m_model": v["m_model"].tolist(),
                       "ratio": v["ratio"].tolist()}
                   for k, v in curves.items()},
        "direction_band_at_outer_edges": {
            "p16": np.interp(r_center_cmp, r_nodes,
                             np.percentile(m_per_dir, 16, axis=0)).tolist(),
            "p84": np.interp(r_center_cmp, r_nodes,
                             np.percentile(m_per_dir, 84, axis=0)).tolist(),
            "p16_ratio": (np.interp(r_center_cmp, r_nodes,
                          np.percentile(m_per_dir, 16, axis=0)) / m_truth_edge).tolist(),
            "p84_ratio": (np.interp(r_center_cmp, r_nodes,
                          np.percentile(m_per_dir, 84, axis=0)) / m_truth_edge).tolist(),
            "legend_label": "direction distribution (NOT statistical uncertainty)",
        },
        "attribution": attribution,
        "legacy_anchor": anchor,
        "elapsed_s": round(time.time() - t0, 1),
    }

    np.savez_compressed(
        out_dir / "legacy_ablation.npz",
        r_nodes=r_nodes, rho_dirs=rho, m_per_dir=m_per_dir,
        dirs=dirs, r_edge_cmp=r_edge_cmp, r_center_cmp=r_center_cmp,
        m_truth_edge=m_truth_edge,
        **{f"m_model_{k}": v["m_model"] for k, v in curves.items()},
        **{f"ratio_{k}": v["ratio"] for k, v in curves.items()})
    _dump_json(out_dir / "legacy_ablation.json", result)
    print(f"step2 done -> {out_dir / 'legacy_ablation.json'}")
    print(f"  anchor: matched {anchor['matched']}/{len(leg_r)} points, "
          f"max |ratio diff| vs legacy JSON = {anchor['max_ratio_diff']}")
    for k, v in attribution["max_abs"].items():
        print(f"  {k}: {v:.4f}")
    return result


# --------------------------------------------------------------------------
# step 3: dual-path mass + convergence (Sobol directions x radial refinement)
# --------------------------------------------------------------------------

# Sweep levels fixed by the plan: directions 256/512/1024 with 4 independent
# scrambles (nested prefixes within one scramble); radial refinement 1/2/4/8
# log-r sub-intervals between consecutive truth edges.
STEP3_N_DIRS = (256, 512, 1024)
STEP3_SCRAMBLE_SEEDS = (0, 1, 2, 3)
STEP3_RADIAL_LEVELS = (1, 2, 4, 8)
STEP3_DIR_GRID_P = 4           # radial grid held fixed for the direction sweep
STEP3_RAD_GRID_N = 1024        # direction count held fixed for the radial sweep
STEP3_RAD_GRID_SEED = 0
# Focus region for the engineering criteria: truth shells fully inside the
# intersection of the three Phi support regions (plan step 4 definition).
STEP3_FOCUS_RMIN, STEP3_FOCUS_RMAX = 2.0, 65.0
# Relative closure is only reported where the truth delta-mass is not ~0
# (plan: never report relative errors where the mass difference -> 0).
STEP3_REL_MASS_MIN_FRAC = 1e-3


def per_direction_rho_and_dn(phi_func, r_nodes_kpc, dirs, L_kpc, V_kms,
                             G=G_KPC_KMS2_MSUN, chunk_nodes=48):
    """Per-direction density and radial force term at every radial node.

    One model evaluation per (node, direction) pair, node-major chunks.
    Returns (rho_dir, dn_dir), each (n_dir, n_r) float64:
      rho_dir[k, j] = physical density at r_j along direction k   [Msun/kpc^3]
      dn_dir[k, j]  = n_k . grad_q phi at q = r_j n_k / L         [dimensionless]
    """
    import jax.numpy as jnp
    r = np.asarray(r_nodes_kpc, dtype=np.float64)
    nd = dirs.shape[0]
    rho = np.empty((nd, r.size), dtype=np.float64)
    dn = np.empty((nd, r.size), dtype=np.float64)
    for i in range(0, r.size, chunk_nodes):
        rc = r[i:i + chunk_nodes]
        q = (rc[:, None, None] / L_kpc) * dirs[None, :, :]      # (nc, nd, 3)
        flat = q.reshape(-1, 3)
        n_hat = np.tile(dirs, (rc.size, 1))
        lap = np.asarray(laplacian_phi_batch(phi_func, jnp.asarray(flat)))
        g = np.asarray(grad_phi_dot_n_batch(
            phi_func, jnp.asarray(flat), jnp.asarray(n_hat)))
        rho[:, i:i + chunk_nodes] = finite_or_fail(
            lap.reshape(q.shape[:2]).T * V_kms ** 2 / (4.0 * np.pi * G * L_kpc ** 2),
            "step3 per-direction rho")
        dn[:, i:i + chunk_nodes] = finite_or_fail(
            g.reshape(q.shape[:2]).T, "step3 per-direction dn")
    return rho, dn


def grid_edge_indices(r_nodes, edges):
    """Node index of every edge value; edges must lie EXACTLY on the grid
    (they are grid boundaries by construction).  Asserts exactness."""
    r = np.asarray(r_nodes, dtype=np.float64)
    idx = np.searchsorted(r, np.asarray(edges, dtype=np.float64) - 1e-12)
    if not np.allclose(r[idx], edges, rtol=0.0, atol=1e-9):
        raise ValueError("truth edges are not exact radial-grid boundaries")
    return idx


def mass_stats_at_prefix(r_nodes, r_inner, rho_dir, dn_dir, L_kpc, V_kms,
                         G=G_KPC_KMS2_MSUN, n_dir=None):
    """Angular-mean mass curves using the first n_dir directions (nested
    prefix of the same scrambled Sobol sequence).  Returns dict with
    rho_mean / dM_rho / m_flux / dM_flux on the node grid."""
    if n_dir is None:
        n_dir = rho_dir.shape[0]
    rho_mean = rho_dir[:n_dir].mean(axis=0)
    dM_rho = cumulative_volume_mass(r_nodes, rho_mean, r_inner)
    m_flux = (dn_dir[:n_dir].mean(axis=0) * r_nodes ** 2
              * V_kms ** 2 / (G * L_kpc))
    dM_flux = cumulative_flux_mass(m_flux)
    return {"rho_mean": rho_mean, "dM_rho": dM_rho,
            "m_flux": m_flux, "dM_flux": dM_flux}


def _load_phi_for_step3(run_dir):
    """Load Phi with x64 OFF (the checkpoint on disk is f32 and equinox
    refuses to deserialise f32 arrays into an f64 `like` tree), then hand
    back the model for x64 evaluation: f64 inputs x f32 weights -> f64
    arithmetic.  The global x64 flag is restored to True afterwards."""
    import jax
    import fit_all
    jax.config.update("jax_enable_x64", False)
    try:
        phi = fit_all.load_potential(run_dir / "models" / "Phi",
                                     checkpoint_index=PHI_CKPT_INDEX)
    finally:
        jax.config.update("jax_enable_x64", True)
    return phi.phi_model


def step3_convergence(audit_dir, models=("baseline", "rin2", "rout65"),
                      n_dirs=STEP3_N_DIRS):
    """Plan step 3: dual-path (volume vs flux) enclosed mass with separate
    direction-count and radial-node convergence checks, then the same final
    evaluation for all three models at the chosen resolution.

    n_dirs: ascending direction levels; the plan caps escalation at 2048."""
    n_dirs = tuple(sorted(int(n) for n in n_dirs))
    if len(n_dirs) < 2:
        raise ValueError("need at least two direction levels")
    n_final = n_dirs[-1]
    audit_dir = Path(audit_dir)
    out_dir = audit_dir / "step3"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    import jax
    import jax.numpy as jnp

    L, V = 10.0, 100.0
    truth = load_truth()
    r_edges = truth["r_edges"]                 # r_edges[0] = 0.5 kpc
    edges_out = r_edges[1:]                    # 60 truth OUTER edges
    dM_truth = finite_or_fail(truth["M_cum_total"], "truth M_cum_total")
    focus_mask = ((r_edges[:-1] >= STEP3_FOCUS_RMIN - 1e-9)
                  & (edges_out <= STEP3_FOCUS_RMAX + 1e-9))
    # relative closure only where the truth delta-mass is not ~0
    rel_ok = dM_truth >= STEP3_REL_MASS_MIN_FRAC * dM_truth[-1]

    # Phi support regions from the frozen step-0 manifest (provenance chain)
    supports = {}
    man_path = audit_dir / "step0" / "manifest.json"
    if man_path.exists():
        with open(man_path) as f:
            man = json.load(f)
        for m in models:
            a = man["runs"][m].get("input_attrs", {})
            supports[m] = [float(a.get("phi_r_min_kpc", 0.0)),
                           float(a.get("phi_r_max_kpc", 1e9))]
    else:
        supports = {m: [0.0, 1e9] for m in models}

    grids = {p: make_radial_nodes(0.5, 70.0, p, boundaries=edges_out)
             for p in STEP3_RADIAL_LEVELS}
    eidx = {p: grid_edge_indices(grids[p], edges_out)
            for p in STEP3_RADIAL_LEVELS}

    model_results = {}
    npz = {
        "r_edges_outer": edges_out, "dM_truth": dM_truth,
        "focus_mask": focus_mask, "rel_closure_ok": rel_ok,
        "scramble_seeds": np.array(STEP3_SCRAMBLE_SEEDS),
        "n_dirs_levels": np.array(n_dirs),
        "radial_levels": np.array(STEP3_RADIAL_LEVELS),
    }

    for m in models:
        print(f"[step3] model {m}: loading Phi ...")
        phi_func = _load_phi_for_step3(RUNS[m])
        timings = {}

        # ---- direction sweep: radial grid fixed at STEP3_DIR_GRID_P ----
        p_dir = STEP3_DIR_GRID_P
        cache = {}                      # seed -> (rho_dir, dn_dir) at N_max
        dir_sweep = {}                  # seed -> N -> stats
        for s in STEP3_SCRAMBLE_SEEDS:
            t0 = time.time()
            dirs = sobol_directions(n_final, scramble_seed=s)
            cache[s] = per_direction_rho_and_dn(
                phi_func, grids[p_dir], dirs, L, V)
            dir_sweep[s] = {
                N: mass_stats_at_prefix(grids[p_dir], 0.5, *cache[s], L, V,
                                        n_dir=N)
                for N in n_dirs}
            timings[f"direction_sweep_seed{s}_s"] = round(time.time() - t0, 1)

        # ---- radial sweep: N fixed, seed fixed ----
        rad_sweep = {p_dir: dir_sweep[STEP3_RAD_GRID_SEED][STEP3_RAD_GRID_N]}
        for p in STEP3_RADIAL_LEVELS:
            if p == p_dir:
                continue
            t0 = time.time()
            dirs = sobol_directions(STEP3_RAD_GRID_N,
                                    scramble_seed=STEP3_RAD_GRID_SEED)
            rd = per_direction_rho_and_dn(phi_func, grids[p], dirs, L, V)
            rad_sweep[p] = mass_stats_at_prefix(grids[p], 0.5, *rd, L, V)
            del rd
            timings[f"radial_sweep_p{p}_s"] = round(time.time() - t0, 1)

        # ---- convergence metrics (max over the FOCUS region) ----
        def ratios(stats, p):
            return (stats["dM_rho"][eidx[p]] / dM_truth,
                    stats["dM_flux"][eidx[p]] / dM_truth)

        dir_conv = {}                   # pair -> seed -> path -> max |dratio|
        for Nlo, Nhi in zip(n_dirs[:-1], n_dirs[1:]):
            key = f"{Nlo}_to_{Nhi}"
            dir_conv[key] = {}
            for s in STEP3_SCRAMBLE_SEEDS:
                vlo, flo = ratios(dir_sweep[s][Nlo], p_dir)
                vhi, fhi = ratios(dir_sweep[s][Nhi], p_dir)
                dv_ = np.abs(vhi - vlo)
                df_ = np.abs(fhi - flo)
                dir_conv[key][str(s)] = {
                    "volume": float(dv_[focus_mask].max()),
                    "flux": float(df_[focus_mask].max()),
                    "volume_argmax_r_kpc": float(
                        edges_out[focus_mask][np.argmax(dv_[focus_mask])]),
                    "flux_argmax_r_kpc": float(
                        edges_out[focus_mask][np.argmax(df_[focus_mask])])}
        dir_last_key = f"{n_dirs[-2]}_to_{n_dirs[-1]}"
        dir_last_ok = all(
            max(v["volume"], v["flux"]) < TH_CONVERGENCE
            for v in dir_conv[dir_last_key].values())

        rad_conv = {}                   # pair -> max |dratio| (volume only)
        for plo, phi_ in zip(STEP3_RADIAL_LEVELS[:-1], STEP3_RADIAL_LEVELS[1:]):
            vlo, _ = ratios(rad_sweep[plo], plo)
            vhi, _ = ratios(rad_sweep[phi_], phi_)
            dr_ = np.abs(vhi - vlo)
            rad_conv[f"{plo}_vs_{phi_}"] = {
                "max": float(dr_[focus_mask].max()),
                "argmax_r_kpc": float(
                    edges_out[focus_mask][np.argmax(dr_[focus_mask])])}
        rad_last_key = (f"{STEP3_RADIAL_LEVELS[-2]}_vs_"
                        f"{STEP3_RADIAL_LEVELS[-1]}")
        rad_last_ok = (rad_conv[rad_last_key]["max"]
                       < TH_CONVERGENCE)

        # closure at (N=1024, p) for the two finest radial levels, seed 0
        closure_by_p = {}
        abs_closure_inner = {}
        for p in (STEP3_DIR_GRID_P, STEP3_RADIAL_LEVELS[-1]):
            st = rad_sweep[p]
            diff = np.abs(st["dM_rho"][eidx[p]] - st["dM_flux"][eidx[p]])
            rel = diff / dM_truth
            closure_by_p[str(p)] = {
                "max": float(rel[focus_mask].max()),
                "argmax_r_kpc": float(
                    edges_out[focus_mask][np.argmax(rel[focus_mask])])}
            inner = ~rel_ok
            abs_closure_inner[str(p)] = float(
                diff[inner].max()) if inner.any() else 0.0

        # ---- resolution choice (plan: last-two-level change < 1%) ----
        p_star = STEP3_DIR_GRID_P if rad_last_ok else STEP3_RADIAL_LEVELS[-1]
        chosen = {"n_dir": int(n_final), "per_interval": int(p_star),
                  "rule": "smallest radial level whose last-two-level change "
                          "and closure are below threshold; directions at the "
                          "finest swept level",
                  "radial_last_two_ok": bool(rad_last_ok),
                  "direction_last_two_ok": bool(dir_last_ok)}

        # ---- final evaluation at (N=n_final, p*) for all scrambles ----
        if p_star == p_dir:
            final = {s: dir_sweep[s][n_final]
                     for s in STEP3_SCRAMBLE_SEEDS}   # reuse, all seeds done
            final_rho = cache
        else:
            final, final_rho = {}, {}
            for s in STEP3_SCRAMBLE_SEEDS:
                if s == STEP3_RAD_GRID_SEED:
                    dirs = sobol_directions(n_final, scramble_seed=s)
                    rd = per_direction_rho_and_dn(
                        phi_func, grids[p_star], dirs, L, V)
                else:
                    t0 = time.time()
                    dirs = sobol_directions(n_final, scramble_seed=s)
                    rd = per_direction_rho_and_dn(
                        phi_func, grids[p_star], dirs, L, V)
                    timings[f"final_p{p_star}_seed{s}_s"] = round(
                        time.time() - t0, 1)
                final_rho[s] = rd
                final[s] = mass_stats_at_prefix(grids[p_star], 0.5, *rd, L, V)

        fin_vol = np.stack([ratios(final[s], p_star)[0]
                            for s in STEP3_SCRAMBLE_SEEDS])   # (seed, edge)
        fin_flux = np.stack([ratios(final[s], p_star)[1]
                             for s in STEP3_SCRAMBLE_SEEDS])
        spread = 0.5 * np.ptp(fin_vol, axis=0)
        clo = np.abs(final[STEP3_RAD_GRID_SEED]["dM_rho"][eidx[p_star]]
                     - final[STEP3_RAD_GRID_SEED]["dM_flux"][eidx[p_star]])
        clo_rel = clo / dM_truth
        closure_final = {
            "max": float(clo_rel[focus_mask].max()),
            "argmax_r_kpc": float(
                edges_out[focus_mask][np.argmax(clo_rel[focus_mask])])}
        # closure per seed at the final resolution (outlier scrambles visible)
        closure_per_seed = {
            str(s): float(np.abs(fin_vol[i] - fin_flux[i])[focus_mask].max())
            for i, s in enumerate(STEP3_SCRAMBLE_SEEDS)}
        closure_final_ok = closure_final["max"] < TH_CONVERGENCE
        gate_model = bool(dir_last_ok and rad_last_ok and closure_final_ok)

        # f32-arithmetic diagnostic (baseline only): same grid/seed, x64 off
        f32_diag = None
        if m == "baseline":
            t0 = time.time()
            jax.config.update("jax_enable_x64", False)
            try:
                dirs = sobol_directions(n_final,
                                        scramble_seed=STEP3_RAD_GRID_SEED)
                rho32, dn32 = per_direction_rho_and_dn(
                    phi_func, grids[p_star], dirs, L, V)
                st32 = mass_stats_at_prefix(grids[p_star], 0.5, rho32, dn32,
                                            L, V)
            finally:
                jax.config.update("jax_enable_x64", True)
            r32, _ = ratios(st32, p_star)
            r64, _ = ratios(final[STEP3_RAD_GRID_SEED], p_star)
            f32_diag = {
                "note": "same (grid, seed) evaluated fully in f32 (legacy "
                        "regime) vs x64 arithmetic with f32 weights; the "
                        "difference bounds the precision-regime sensitivity",
                "max_abs_ratio_diff_focus": float(
                    np.abs(r32 - r64)[focus_mask].max()),
                "elapsed_s": round(time.time() - t0, 1)}

        # ---- persist per-direction primitives at the final resolution ----
        for s in STEP3_SCRAMBLE_SEEDS:
            rho_dir, dn_dir = final_rho[s]
            m_per_dir = _per_direction_enclosed(grids[p_star],
                                                rho_dir)[:, eidx[p_star]]
            npz[f"{m}__seed{s}__rho_dir"] = rho_dir
            npz[f"{m}__seed{s}__dM_per_dir_edges"] = m_per_dir
        npz[f"{m}__r_nodes"] = grids[p_star]
        npz[f"{m}__edge_node_idx"] = eidx[p_star]
        npz[f"{m}__support_kpc"] = np.array(supports[m])
        for s in STEP3_SCRAMBLE_SEEDS:
            npz[f"{m}__seed{s}__dM_rho"] = final[s]["dM_rho"]
            npz[f"{m}__seed{s}__dM_flux"] = final[s]["dM_flux"]
            npz[f"{m}__seed{s}__m_flux_nodes"] = final[s]["m_flux"]
            npz[f"{m}__seed{s}__rho_mean"] = final[s]["rho_mean"]
        npz[f"{m}__ratio_vol_seeds"] = fin_vol
        npz[f"{m}__ratio_flux_seeds"] = fin_flux
        # sweep curves vs radius (for the closure / refinement-difference fig)
        for N in n_dirs:
            npz[f"{m}__dirsweep_ratio_vol_N{N}"] = np.stack(
                [ratios(dir_sweep[s][N], p_dir)[0]
                 for s in STEP3_SCRAMBLE_SEEDS])
            npz[f"{m}__dirsweep_ratio_flux_N{N}"] = np.stack(
                [ratios(dir_sweep[s][N], p_dir)[1]
                 for s in STEP3_SCRAMBLE_SEEDS])
        for p in STEP3_RADIAL_LEVELS:
            npz[f"{m}__radsweep_ratio_vol_p{p}"] = ratios(rad_sweep[p], p)[0]

        model_results[m] = {
            "phi_checkpoint": str(RUNS[m] / "models" / "Phi" /
                                  f"potential-{PHI_CKPT_INDEX}_model.eqx"),
            "support_kpc": supports[m],
            "timings_s": timings,
            "direction_convergence": dir_conv,
            "direction_last_two_ok": bool(dir_last_ok),
            "scramble_spread_half_ptp_focus": {
                "max": float(spread[focus_mask].max()),
                "label": "quadrature repeatability (NOT a model confidence "
                         "interval)"},
            "radial_convergence": rad_conv,
            "radial_last_two_ok": bool(rad_last_ok),
            "closure_max_focus": closure_by_p,
            "closure_abs_msun_inner": abs_closure_inner,
            "chosen_resolution": chosen,
            "closure_final": closure_final,
            "closure_final_per_seed": closure_per_seed,
            "closure_final_ok": bool(closure_final_ok),
            "final_ratio_volume_focus_range": [
                float(fin_vol[:, focus_mask].min()),
                float(fin_vol[:, focus_mask].max())],
            "final_ratio_flux_focus_range": [
                float(fin_flux[:, focus_mask].min()),
                float(fin_flux[:, focus_mask].max())],
            "f32_diagnostic": f32_diag,
            "gate_ok": gate_model,
        }
        print(f"[step3] model {m}: dir_ok={dir_last_ok} rad_ok={rad_last_ok} "
              f"closure_ok={closure_final_ok} -> gate {gate_model}")

    np.savez_compressed(out_dir / "mass_profiles.npz", **npz)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_step": 3,
        "models": list(models),
        "constants": {
            "G_kpc_kms2_msun": G_KPC_KMS2_MSUN, "L_kpc": L, "V_kms": V,
            "r_inner_kpc": 0.5, "r_outer_kpc": 70.0,
            "threshold": TH_CONVERGENCE,
        },
        "numerics": {
            "jax_enable_x64_host": True,
            "phi_params_dtype": "float32",
            "model_eval": "f64 inputs x f32 weights -> f64 arithmetic; "
                          "x64 switched off only to deserialise the "
                          "checkpoint (equinox dtype guard)",
            "radial_rule": "trapezoid in log r on g(u) = 4 pi s^3 <rho>(s); "
                           "nodes include 0.5 kpc and every truth outer edge "
                           "exactly",
            "direction_rule": "scrambled Sobol (mu=2u-1, az=2 pi v), shared "
                              "across models, prefix-nested within a "
                              "scramble; 4 independent scrambles per level",
            "signed_density": True,
        },
        "focus_region": {
            "rule": "truth shells fully inside [2, 65] kpc (intersection of "
                    "the three Phi support regions)",
            "r_min_kpc": STEP3_FOCUS_RMIN, "r_max_kpc": STEP3_FOCUS_RMAX,
            "n_shells": int(focus_mask.sum()),
            "rel_closure_min_frac_of_total": STEP3_REL_MASS_MIN_FRAC,
        },
        "sweeps": {
            "n_directions": list(n_dirs),
            "scramble_seeds": list(STEP3_SCRAMBLE_SEEDS),
            "radial_levels": list(STEP3_RADIAL_LEVELS),
            "direction_sweep_grid_p": STEP3_DIR_GRID_P,
            "radial_sweep_n_dir": STEP3_RAD_GRID_N,
            "radial_sweep_seed": STEP3_RAD_GRID_SEED,
        },
        "models_detail": model_results,
        "gate": {
            "threshold": TH_CONVERGENCE,
            "criteria": "max over focus region of (last-two-level direction "
                        "change, both paths) < 1%; (last-two-level radial "
                        "change, volume) < 1%; closure |dM_rho-dM_flux|/"
                        "dM_truth at the chosen resolution < 1%",
            "per_model_ok": {m: model_results[m]["gate_ok"] for m in models},
            "all_ok": bool(all(model_results[m]["gate_ok"] for m in models)),
            "escalation_note": "if failed, the plan allows at most N=2048 "
                               "directions or 16 radial sub-intervals before "
                               "stopping at a numerical problem",
        },
        "elapsed_s": round(time.time() - t_start, 1),
    }
    _dump_json(out_dir / "convergence.json", result)
    print(f"[step3] done -> {out_dir / 'convergence.json'} "
          f"(all_ok={result['gate']['all_ok']}, "
          f"elapsed {result['elapsed_s']}s)")
    return result


# --------------------------------------------------------------------------
# step 4: inner-anchor comparison DeltaM(r;a) + per-shell increments
# --------------------------------------------------------------------------

# Plan step 4: a = grid start (0.5 kpc, where the truth cumulative is zero)
# plus the first truth shell edge >= 2 kpc and >= 5 kpc (robustness check).
STEP4_ANCHOR_REQUESTS = (0.5, 2.0, 5.0)
# Common region (plan): truth shells fully inside the intersection of the
# three Phi support regions [2, 65] kpc.
STEP4_COMMON_RMIN, STEP4_COMMON_RMAX = 2.0, 65.0
STEP4_BOUNDARY_NEDGE = 3     # first/last focus shells counted as near-boundary
STEP4_SHELL_MIN_ABOVE = 10   # min above-floor shells before "systematic" verdict
STEP4_SHELL_FRAC_ABOVE = 2.0 / 3.0   # above-floor fraction = "systematic"


def anchor_edge_and_node(truth_r_edges, edge_node_idx, r_nodes, request_kpc):
    """Anchor definition (plan step 4): the first truth shell edge >= request;
    request <= 0.5 maps to the grid start, where the truth cumulative is zero
    by construction (it excludes r < 0.5 kpc).

    Returns dict with the anchor radius, its position among the truth OUTER
    edges (None for the grid start) and its exact radial-grid node index.
    """
    edges_out = np.asarray(truth_r_edges)[1:]
    if request_kpc <= float(truth_r_edges[0]) + 1e-12:
        return {"request_kpc": float(request_kpc),
                "r_kpc": float(truth_r_edges[0]), "edge_pos": None,
                "node_idx": 0, "m_truth_at_anchor": 0.0,
                "rule": "grid start; M_truth(0.5)=0 (truth cumulative "
                        "excludes r<0.5 kpc)"}
    j = int(np.searchsorted(edges_out, request_kpc - 1e-12))
    if j >= edges_out.size:
        raise ValueError(f"no truth shell edge >= {request_kpc} kpc")
    r_a = float(edges_out[j])
    node = int(edge_node_idx[j])
    if abs(float(r_nodes[node]) - r_a) > 1e-9:
        raise ValueError(f"anchor {r_a} kpc is not an exact radial-grid node")
    return {"request_kpc": float(request_kpc), "r_kpc": r_a, "edge_pos": j,
            "node_idx": node, "m_truth_at_anchor": None,
            "rule": f"first truth shell edge >= {request_kpc:g} kpc"}


def shell_increments(edge_values):
    """Per-shell mass from cumulative values bound to the K+1 radii
    [r_start, e_1, ..., e_K] (grid start + truth outer edges)."""
    return np.diff(np.asarray(edge_values, dtype=np.float64))


def _longest_true_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def step4_anchor(audit_dir):
    """Plan step 4: anchor the enclosed-mass comparison at
    a in {0.5, first edge >= 2, first edge >= 5} kpc and add per-shell
    increments, to separate central extrapolation error from systematic
    bias in the data-constrained shells.

    Reads ONLY the persisted step-3 products (no model reload):
    mass_profiles.npz at the chosen resolution (N=2048, p=4, 4 scrambles)
    and convergence.json for the certified numerical floors.  The flux
    path is the primary estimator; the volume path is an independent
    corroboration whose outer-region single-scramble floor (~1-1.5%) is
    carried along explicitly.  Anchored volume differences use the exact
    telescoping identity on the shared radial grid and are cross-checked
    against a direct integration from the anchor node.
    """
    audit_dir = Path(audit_dir)
    out_dir = audit_dir / "step4"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    npz_path = audit_dir / "step3" / "mass_profiles.npz"
    conv_path = audit_dir / "step3" / "convergence.json"
    missing = [str(p) for p in (npz_path, conv_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("step4 needs the step-3 products: "
                                + ", ".join(missing))
    conv = json.loads(Path(conv_path).read_text())
    d = np.load(npz_path)
    truth = load_truth()

    # ---- persisted arrays vs the truth file (provenance cross-check) ----
    edges_out = truth["r_edges"][1:]
    assert np.array_equal(d["r_edges_outer"], edges_out)
    assert np.allclose(d["dM_truth"], truth["M_cum_total"], rtol=1e-12)
    shell_truth = finite_or_fail(truth["M_shell_total"], "truth M_shell_total")
    assert np.allclose(np.diff(np.concatenate([[0.0], truth["M_cum_total"]])),
                       shell_truth, rtol=1e-9, atol=0.0)

    models = list(conv["models"])
    r_nodes = {m: d[f"{m}__r_nodes"] for m in models}
    for m in models:
        assert r_nodes[m][0] == truth["r_edges"][0]
    eidx = d[f"{models[0]}__edge_node_idx"]
    for m in models[1:]:
        assert np.array_equal(eidx, d[f"{m}__edge_node_idx"])

    # ---- regions: common region + supplementary outer point ----
    focus = ((truth["r_edges"][:-1] >= STEP4_COMMON_RMIN - 1e-9)
             & (edges_out <= STEP4_COMMON_RMAX + 1e-9))
    assert np.array_equal(focus, d["focus_mask"].astype(bool))
    j_sup = int(np.searchsorted(edges_out, STEP4_COMMON_RMAX + 1e-9))
    r_sup = float(edges_out[j_sup])
    supports = {m: [float(x) for x in d[f"{m}__support_kpc"]] for m in models}
    extrapolation_for = [m for m in models if r_sup > supports[m][1] + 1e-9]

    # ---- certified numerical floors from the step-3 direction sweep ----
    n_levels = conv["sweeps"]["n_directions"]
    n_lo, n_hi = int(n_levels[-2]), int(n_levels[-1])
    floors = {}
    for m in models:
        fv = np.abs(d[f"{m}__dirsweep_ratio_flux_N{n_lo}"]
                    - d[f"{m}__dirsweep_ratio_flux_N{n_hi}"])
        vv = np.abs(d[f"{m}__dirsweep_ratio_vol_N{n_lo}"]
                    - d[f"{m}__dirsweep_ratio_vol_N{n_hi}"])
        floors[m] = {
            "flux_seed_max": float(fv[:, focus].max()),
            "vol_seed_max": float(vv[:, focus].max()),
            "vol_mean_curve": float(np.abs(
                d[f"{m}__dirsweep_ratio_vol_N{n_hi}"].mean(axis=0)
                - d[f"{m}__dirsweep_ratio_vol_N{n_lo}"].mean(axis=0))[focus].max()),
            "closure_final": float(
                conv["models_detail"][m]["closure_final"]["max"]),
        }

    # ---- anchors (plan: only real boundaries; print the final ranges) ----
    anchors = [anchor_edge_and_node(truth["r_edges"], eidx,
                                    r_nodes[models[0]], req)
               for req in STEP4_ANCHOR_REQUESTS]
    for a in anchors:
        if a["edge_pos"] is not None:
            a["m_truth_at_anchor"] = float(truth["M_cum_total"][a["edge_pos"]])
    print(f"[step4] anchors: "
          + ", ".join(f"a={a['r_kpc']:g} kpc ({a['rule']})" for a in anchors))
    print(f"[step4] common region: {int(focus.sum())} complete shells, "
          f"outer edges {edges_out[focus][0]:g}-{edges_out[focus][-1]:g} kpc; "
          f"supplementary outer point {r_sup:g} kpc "
          f"(extrapolation for {extrapolation_for or 'none'})")

    seeds = [int(s) for s in d["scramble_seeds"]]
    m_cum_edges = np.concatenate([[0.0], truth["M_cum_total"]])  # at [0.5, e_1..]
    # relative-mass guard for every RATIO statistic (plan: never report a
    # relative error where the mass difference -> 0, e.g. at the edge just
    # beyond an anchor); absolute dM stats stay unguarded
    rel_guard = STEP3_REL_MASS_MIN_FRAC * truth["M_cum_total"][-1]

    z = {
        "anchor_request_kpc": np.array([a["request_kpc"] for a in anchors]),
        "anchor_r_kpc": np.array([a["r_kpc"] for a in anchors]),
        "r_edges_inner": truth["r_edges"][:-1],
        "r_edges_outer": edges_out,
        "r_center": truth["r_center"],
        "focus_mask": focus,
        "shell_truth": shell_truth,
        "scramble_seeds": np.array(seeds),
        "supplement_edge_pos": np.array(j_sup),
    }

    def anchor_summary(dm_seeds, dm_truth_a, valid, floor):
        """Ratio + absolute-difference stats over the common-region edges
        beyond the anchor (plan: compare dM(r), not only the ratio).  Ratio
        statistics apply the relative-mass guard; dM statistics do not."""
        sel = valid & focus
        rel_ok = sel & (np.abs(dm_truth_a) >= rel_guard)
        mean_curve = dm_seeds.mean(axis=0)
        d_abs = mean_curve[sel] - dm_truth_a[sel]
        ratio = mean_curve[rel_ok] / dm_truth_a[rel_ok]
        dev = np.abs(ratio - 1.0)
        i = int(np.argmax(dev)) if dev.size else 0
        return {
            "n_edges": int(sel.sum()),
            "n_edges_relmass_guarded": int(rel_ok.sum()),
            "relmass_guard": float(rel_guard),
            "max_abs_ratio_dev": float(dev.max()) if dev.size else None,
            "argmax_r_kpc": float(edges_out[rel_ok][i]) if dev.size else None,
            "mean_abs_ratio_dev": float(dev.mean()) if dev.size else None,
            "dM_abs_mean_msun": float(d_abs.mean()),
            "dM_abs_std_msun": float(d_abs.std()),
            "dM_abs_cv_std_over_abs_mean": float(
                d_abs.std() / max(abs(d_abs.mean()), 1e-30)),
            "half_ptp_spread_max": float(
                (0.5 * np.ptp(dm_seeds[:, sel], axis=0)).max()),
            "floor": floor,
            "above_floor": bool(dev.size > 0
                                and dev.max() > max(TH_CONVERGENCE,
                                                    5.0 * floor)),
        }

    def shell_summary(dev, floor_shell):
        sel = focus
        above = np.abs(dev[sel]) > floor_shell[sel]
        n_above = int(above.sum())
        pos = int((dev[sel][above] > 0).sum())
        neg = n_above - pos
        same = max(pos, neg) / n_above if n_above else 0.0
        head = slice(None, STEP4_BOUNDARY_NEDGE)
        tail = slice(-STEP4_BOUNDARY_NEDGE, None)
        ratio_to_floor = np.abs(dev[sel]) / floor_shell[sel]
        return {
            "n_focus_shells": int(sel.sum()),
            "dev_median": float(np.median(dev[sel])),
            "dev_max_abs": float(np.abs(dev[sel]).max()),
            "n_above_floor": n_above,
            "frac_same_sign_above_floor": float(same),
            "longest_same_sign_run": int(max(
                _longest_true_run(above & (dev[sel] > 0)),
                _longest_true_run(above & (dev[sel] < 0)))),
            "interior_max_abs_dev_over_floor": float(
                ratio_to_floor[STEP4_BOUNDARY_NEDGE:-STEP4_BOUNDARY_NEDGE].max()),
            "boundary_max_abs_dev_over_floor": float(
                max(ratio_to_floor[head].max(), ratio_to_floor[tail].max())),
            "floor_rule": "max(4-scramble half-ptp, |N2048-N1024| level "
                          "change), both RELATIVE to the shell mass; the "
                          "independent-sum bound floor_cum x (M_in(e_i)+"
                          "M_in(e_{i-1}))/M_shell_i is recorded separately "
                          "as an over-conservative upper bound (it ignores "
                          "the error cancellation between adjacent edges)",
        }

    per_model = {}
    for m in models:
        mf_seeds = np.stack([d[f"{m}__seed{s}__m_flux_nodes"] for s in seeds])
        drho_seeds = np.stack([d[f"{m}__seed{s}__dM_rho"] for s in seeds])
        mf_edges = mf_seeds[:, eidx]        # (n_seed, 60) anchor-independent
        mrho_edges = drho_seeds[:, eidx]

        # ---- anchored curves DeltaM(r;a) for every anchor ----
        anchor_res = {}
        for a in anchors:
            node = a["node_idx"]
            valid = edges_out > a["r_kpc"] + 1e-9
            dm_flux = mf_edges - mf_seeds[:, node][:, None]
            dm_vol = mrho_edges - drho_seeds[:, node][:, None]
            # cross-check the telescoping identity against a direct
            # integration from the anchor node (guards the persisted npz)
            rho_mean0 = d[f"{m}__seed{seeds[0]}__rho_mean"]
            direct = cumulative_volume_mass(r_nodes[m][node:],
                                            rho_mean0[node:], a["r_kpc"])
            later = eidx > node
            assert np.allclose(dm_vol[0][later],
                               direct[eidx[later] - node], rtol=1e-10, atol=0.0)
            dm_truth_a = truth["M_cum_total"] - a["m_truth_at_anchor"]
            key = f"a{a['request_kpc']:g}"
            zkey = f"{a['request_kpc']:g}"
            z[f"{m}__anchor{zkey}__dM_flux_seeds"] = dm_flux
            z[f"{m}__anchor{zkey}__dM_vol_seeds"] = dm_vol
            z[f"{m}__anchor{zkey}__dM_truth"] = dm_truth_a
            z[f"{m}__anchor{zkey}__valid"] = valid
            anchor_res[key] = {
                "r_kpc": a["r_kpc"],
                "n_edges_beyond_anchor": int(valid.sum()),
                "flux": anchor_summary(dm_flux, dm_truth_a, valid,
                                       floors[m]["flux_seed_max"]),
                "volume": anchor_summary(dm_vol, dm_truth_a, valid,
                                         floors[m]["vol_seed_max"]),
            }

        # ---- per-shell increments (anchor-free differences) ----
        # prepend the cumulative mass at the grid start (node 0 = 0.5 kpc):
        # truth shell 0 spans 0.5 -> 0.559 kpc, so the model increment must
        # subtract M(0.5), not zero (volume path has M_rho(0.5) = 0 anyway)
        shell_flux = shell_increments(
            np.concatenate([mf_seeds[:, :1], mf_edges], axis=1))
        shell_vol = shell_increments(
            np.concatenate([drho_seeds[:, :1], mrho_edges], axis=1))
        dev_flux = shell_flux.mean(axis=0) / shell_truth - 1.0
        dev_vol = shell_vol.mean(axis=0) / shell_truth - 1.0
        spread_flux = 0.5 * np.ptp(shell_flux, axis=0) / shell_truth
        spread_vol = 0.5 * np.ptp(shell_vol, axis=0) / shell_truth
        # empirical level-change floor: per-shell increments rebuilt from the
        # persisted N=1024 vs N=2048 ratio curves, per scramble.  Unlike the
        # independent-sum propagation this keeps the error cancellation
        # between adjacent edges that the shared Sobol directions provide.
        def _level_shell_change(path):
            sh = {}
            for n in (n_lo, n_hi):
                ratio = d[f"{m}__dirsweep_ratio_{path}_N{n}"]
                sh[n] = shell_increments(
                    np.concatenate([np.zeros((len(seeds), 1)),
                                    ratio * truth["M_cum_total"][None, :]],
                                   axis=1))
            return (np.abs(sh[n_hi] - sh[n_lo]) / shell_truth).max(axis=0)
        level_flux = _level_shell_change("flux")
        level_vol = _level_shell_change("vol")
        # over-conservative independent-sum bound, recorded NOT used
        prop_flux = floors[m]["flux_seed_max"] * (
            m_cum_edges[1:] + m_cum_edges[:-1]) / shell_truth
        prop_vol = floors[m]["vol_seed_max"] * (
            m_cum_edges[1:] + m_cum_edges[:-1]) / shell_truth
        floor_shell_flux = np.maximum(spread_flux, level_flux)
        floor_shell_vol = np.maximum(spread_vol, level_vol)
        z[f"{m}__shell_flux_seeds"] = shell_flux
        z[f"{m}__shell_vol_seeds"] = shell_vol
        z[f"{m}__shell_dev_flux"] = dev_flux
        z[f"{m}__shell_dev_vol"] = dev_vol
        z[f"{m}__shell_spread_flux"] = spread_flux
        z[f"{m}__shell_spread_vol"] = spread_vol
        z[f"{m}__shell_floor_flux"] = floor_shell_flux
        z[f"{m}__shell_floor_vol"] = floor_shell_vol
        z[f"{m}__shell_propbound_flux"] = prop_flux
        z[f"{m}__shell_propbound_vol"] = prop_vol
        z[f"{m}__floors"] = np.array([floors[m]["flux_seed_max"],
                                      floors[m]["vol_seed_max"],
                                      floors[m]["vol_mean_curve"]])

        # ---- classification (rules recorded verbatim; flux path primary) ----
        b = {k: v["flux"]["max_abs_ratio_dev"] for k, v in anchor_res.items()}
        ffloor = floors[m]["flux_seed_max"]
        thresh = max(TH_CONVERGENCE, 5.0 * ffloor)
        biased_05 = bool(b["a0.5"] > thresh)
        anchor_fixes = {k: bool(b[k] <= 0.5 * b["a0.5"] and b[k] <= thresh)
                        for k in ("a2", "a5")}
        ssum_f = shell_summary(dev_flux, floor_shell_flux)
        # systematic = numerically RESOLVED (above its own floor) in >= 2/3
        # of the focus shells; the sign pattern (same-sign fraction, longest
        # run) is reported separately as radial structure -- a sign change
        # across radius (e.g. inner under / outer over) is still systematic
        shells_biased = bool(
            ssum_f["n_above_floor"] >= STEP4_SHELL_MIN_ABOVE
            and ssum_f["n_above_floor"]
            >= STEP4_SHELL_FRAC_ABOVE * ssum_f["n_focus_shells"])
        boundary_only = bool(ssum_f["n_above_floor"] > 0
                             and ssum_f["interior_max_abs_dev_over_floor"] <= 1.0
                             and ssum_f["boundary_max_abs_dev_over_floor"] > 1.0)
        if shells_biased:
            verdict = ("shell masses systematically deviate after re-anchoring "
                       "-> not a single central constant error; proceed to "
                       "steps 5/6")
        elif biased_05 and any(anchor_fixes.values()):
            verdict = ("inner-region contribution likely dominates; keep the "
                       "central-extrapolation caveat and report constrained-"
                       "shell masses as primary")
        elif boundary_only:
            verdict = ("deviations concentrate near support-region edges; "
                       "boundary/selection effects possible, do not "
                       "generalise across radius")
        elif not biased_05:
            verdict = ("no deviation above the numerical floor in the common "
                       "region (flux path)")
        else:
            verdict = ("0.5-anchored deviation persists at every anchor "
                       "without a systematic shell signature; ambiguous")

        per_model[m] = {
            "support_kpc": supports[m],
            "numerical_floors": floors[m],
            "anchors": anchor_res,
            "shells": {"flux": ssum_f,
                       "volume": shell_summary(dev_vol, floor_shell_vol)},
            "supplement_point": {
                "r_kpc": r_sup,
                "shell_center_kpc": float(truth["r_center"][j_sup]),
                "shell_mass_truth_msun": float(shell_truth[j_sup]),
                "shell_mass_flux_msun": float(shell_flux.mean(axis=0)[j_sup]),
                "dev_flux": float(dev_flux[j_sup]),
                "floor_flux": float(floor_shell_flux[j_sup]),
                "dev_vol": float(dev_vol[j_sup]),
                "floor_vol": float(floor_shell_vol[j_sup]),
                "extrapolation": m in extrapolation_for,
            },
            "classification": {
                "b_flux_max_abs_ratio_dev": b,
                "threshold": thresh,
                "rules": {
                    "biased_at_0.5": "b(a0.5) > max(1%, 5 x flux floor)",
                    "anchor_fixes": "b(a) <= 0.5 b(a0.5) AND b(a) <= "
                                    "max(1%, 5 x flux floor)",
                    "shells_biased": f">= {STEP4_SHELL_MIN_ABOVE} focus shells "
                                     "AND >= "
                                     f"{STEP4_SHELL_FRAC_ABOVE:.2f} of them "
                                     "numerically resolved above their own "
                                     "floor; sign pattern reported separately "
                                     "(radial structure, not noise)",
                    "boundary_only": "above-floor deviations only within the "
                                     f"first/last {STEP4_BOUNDARY_NEDGE} focus "
                                     "shells",
                },
                "biased_at_0.5": biased_05,
                "anchor_fixes": anchor_fixes,
                "shells_systematically_biased": shells_biased,
                "boundary_only": boundary_only,
                "verdict": verdict,
                "caveat": "improvement after re-anchoring is NOT causal "
                          "proof: changing the anchor changes the observed "
                          "quantity; the anchor closest to truth must not be "
                          "kept selectively",
            },
        }
        print(f"[step4] model {m}: b(a0.5)={b['a0.5']:.4f} "
              f"b(a2)={b['a2']:.4f} b(a5)={b['a5']:.4f} | "
              f"shells above floor {ssum_f['n_above_floor']}/"
              f"{ssum_f['n_focus_shells']} (same-sign frac "
              f"{ssum_f['frac_same_sign_above_floor']:.2f}, longest run "
              f"{ssum_f['longest_same_sign_run']}) -> {verdict}")

    z["models"] = np.array(models)
    np.savez_compressed(out_dir / "anchor_comparison.npz", **z)
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_step": 4,
        "question": "does the mass deviation come from the untrained centre "
                    "or from the data-constrained shells?",
        "primary_estimator": "flux path (step-3 certified)",
        "corroboration": "volume path (outer-region single-scramble floor "
                         "~1-1.5% annotated; 4-scramble mean curve <=0.3%)",
        "inputs": {
            "step3_npz": {"path": str(npz_path),
                          "sha256": sha256_file(npz_path)},
            "convergence_json": {"path": str(conv_path),
                                 "sha256": sha256_file(conv_path)},
            "truth_h5": {"path": str(TRUTH_PATH),
                         "sha256": sha256_file(TRUTH_PATH)},
        },
        "resolution": {
            "n_dir": conv["models_detail"][models[0]]["chosen_resolution"]["n_dir"],
            "per_interval": conv["models_detail"][models[0]]["chosen_resolution"]["per_interval"],
            "scramble_seeds": seeds,
            "note": "read from the persisted step-3 final evaluation; no "
                    "model was reloaded for step 4",
        },
        "anchors": anchors,
        "regions": {
            "focus": {
                "rule": "truth shells fully inside [2, 65] kpc "
                        "(intersection of the three Phi support regions)",
                "n_shells": int(focus.sum()),
                "outer_edge_range_kpc": [float(edges_out[focus][0]),
                                         float(edges_out[focus][-1])],
            },
            "supplement_outer": {
                "r_kpc": r_sup,
                "shell_center_kpc": float(truth["r_center"][j_sup]),
                "in_support_for": [x for x in models
                                   if x not in extrapolation_for],
                "extrapolation_for": extrapolation_for,
            },
        },
        "numerical_floors": {
            "provenance": f"step-3 direction sweep N {n_lo}->{n_hi}, max over "
                          "the common region; per-seed values single-scramble",
            "per_model": floors,
            "shell_level_rule": "per-shell floor = max(4-scramble half-ptp, "
                                "level-change |N2048-N1024|), relative to the "
                                "shell mass; shell_propbound_* arrays in the "
                                "npz keep the over-conservative "
                                "independent-sum bound for reference only",
        },
        "per_model": per_model,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _dump_json(out_dir / "anchor_comparison.json", result)
    print(f"[step4] done -> {out_dir / 'anchor_comparison.json'} "
          f"(elapsed {result['elapsed_s']}s)")
    return result


# --------------------------------------------------------------------------
# step 5: separate angular structure from uncertainty
# --------------------------------------------------------------------------

# Plan step 5: explicit radii for the density sky maps ("约 5、10、25、50 kpc"
# as definite radii, all inside the common region [2, 65] kpc).
STEP5_MAP_RADII = (5.0, 10.0, 25.0, 50.0)
# Same direction machinery/resolution as the step-3 final evaluation so the
# direction statistics and the certified quadrature floors stay comparable.
STEP5_N_DIR = 2048
STEP5_SCRAMBLE_SEEDS = (0, 1, 2, 3)
STEP5_LMAX = 4                 # low-order angular structure reported up to l=4
# Sky-map pixel grid: cell CENTRES, uniform in lon and in sin(lat); every
# cell then has the SAME area, so plain cell statistics are sphere averages
STEP5_MAP_NLAT, STEP5_MAP_NLON = 180, 360

# ---- step 5b: particle angular truth from the raw Auriga snapshot ----
# Full snapshot (all 8 chunks + Subfind tables) provided by the user on
# 2026-09-15 as data/halo12_raw.tar.gz; extracted to data/halo12_raw/.
RAW_SNAP_TEMPLATE = REPO / "data/halo12_raw/snapdir_127/snapshot_127.{i}.hdf5"
RAW_SNAP_N_CHUNKS = 8
HUBBLE_PARAM = 0.6777          # asserted against the snapshot Header
# particle truth maps: coarser equal-area grid than the model maps
# (180x360 would leave ~2-4 particles per cell in these shells -- shot noise
# would dominate; 45x90 gives ~30-60 per cell at the four map radii)
STEP5_TRUTH_GRID = (45, 90)
# shell = [r/fw, r*fw]: log-symmetric, +/-~9.1% radial width -- narrow enough
# that the l<=4 angular structure is not smeared by the radial gradient,
# wide enough to keep per-cell Poisson noise at the level above
STEP5_TRUTH_SHELL_FW = 1.1
STEP5_TRUTH_BOOTSTRAP = 8      # particle resamples for A_l/A_0 error bars
STEP5_TRUTH_BOOT_SEED = 20260915
# star convention: formed stars only (SFT > 0), matching the training sample
# (0 of 1652969 training stars have SFT < 0); the all-PartType4 variant is
# kept as a systematic check


def sph_harm_power(values, dirs, l_max):
    """Angular power spectrum A_l of a scalar field sampled on an
    equal-weight direction quadrature (Sobol set, dOmega-uniform).

    a_lm = <values * conj(Y_lm)>_Omega (a 1/4 pi-scaled coefficient; the
    scale cancels in every A_l/A_0 ratio reported here);
    A_l = sqrt(sum_m |a_lm|^2).  Returns array (l_max + 1,);
    A[0] = |a_00| = <values> / sqrt(4 pi).
    """
    from scipy.special import sph_harm_y   # scipy >= 1.17 (old sph_harm removed)
    az = np.arctan2(dirs[:, 1], dirs[:, 0])
    pol = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    out = np.empty(l_max + 1, dtype=np.float64)
    for l in range(l_max + 1):
        acc = 0.0
        for m in range(-l, l + 1):
            alm = np.mean(values * np.conj(sph_harm_y(l, m, pol, az)))
            acc += float(np.abs(alm)) ** 2
        out[l] = np.sqrt(acc)
    return out


def equal_area_map_grid(n_lat=STEP5_MAP_NLAT, n_lon=STEP5_MAP_NLON):
    """Equal-area cell-centre grid on the sphere: uniform in sin(lat) and lon.
    Returns (lat_rad, lon_rad) 1-D centre arrays."""
    mu = 2.0 * (np.arange(n_lat) + 0.5) / n_lat - 1.0     # sin(lat) centres
    lat = np.arcsin(mu)
    lon = 2.0 * np.pi * ((np.arange(n_lon) + 0.5) / n_lon) - np.pi
    return lat, lon


def directions_from_grid(lat, lon):
    """Unit direction vectors for every (lat, lon) cell centre, row-major."""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    cl, sl = np.cos(LAT), np.sin(LAT)
    return np.column_stack([(cl * np.cos(LON)).ravel(),
                            (cl * np.sin(LON)).ravel(), sl.ravel()])


def map_stats(rho_map):
    """Mean and negative fraction of a (n_lat, n_lon) map on the equal-area
    centre grid (uniform in sin(lat) and lon): every cell has the SAME area,
    so the statistics are plain means -- no extra cos(lat) weight (that
    would double-count the area element).  Sign: rho < 0 counts."""
    mean = float(rho_map.mean())
    neg = float((rho_map < 0).mean())
    return mean, neg


def density_map(phi_func, r_kpc, lat, lon, L_kpc, V_kms,
                G=G_KPC_KMS2_MSUN, batch=100_000):
    """Physical density [Msun/kpc^3] on the full (lat, lon) grid at radius r.
    Model-only diagnostic (plan: maps are not compared to a spherical-truth
    angular error)."""
    dirs = directions_from_grid(lat, lon)
    q = dirs * (r_kpc / L_kpc)
    rho = np.empty(q.shape[0], dtype=np.float64)
    for i in range(0, q.shape[0], batch):
        rho[i:i + batch] = rho_from_phi(phi_func, q[i:i + batch], L_kpc, V_kms, G)
    return finite_or_fail(rho, f"density map r={r_kpc}").reshape(
        lat.size, lon.size)


def pooled_direction_band(d, model, seeds, dM_truth):
    """Direction distribution of the per-direction volume-path integrals
    (persisted by step 3), pooled over the scramble seeds.

    Returns percentiles of the per-direction ratio DeltaM_dir(r;0.5)/
    DeltaM_truth, the pooled-mean estimator curve and the per-seed mean
    curves whose min-max is the quadrature-repeatability envelope."""
    per = np.concatenate(
        [d[f"{model}__seed{s}__dM_per_dir_edges"] for s in seeds], axis=0)
    ratio = per / dM_truth[None, :]
    seed_means = np.stack([
        d[f"{model}__seed{s}__dM_per_dir_edges"].mean(axis=0) / dM_truth
        for s in seeds])
    return {
        "p16": np.percentile(ratio, 16, axis=0),
        "p50": np.percentile(ratio, 50, axis=0),
        "p84": np.percentile(ratio, 84, axis=0),
        "mean": ratio.mean(axis=0),
        "seed_means": seed_means,
        "rep_min": seed_means.min(axis=0),
        "rep_max": seed_means.max(axis=0),
        "n_directions_pooled": int(per.shape[0]),
    }


def angular_truth_inventory(data_dir=None, include_raw_snapshot=True):
    """Plan step 5 eligibility scan: does any data file provide particle
    positions for ALL gravitational-source components (gas PartType0,
    high-res dark matter PartType1, stars PartType4) of the truth profile,
    which is the minimum for an angular truth in the model frame?

    Position datasets are detected in both known schemas: Gadget-style
    Coordinates/Positions and the dpjax star export's scalar x/y/z (see
    scripts/auriga/prepare_data.py); the detected names are recorded so the
    evidence table stays faithful to the files.  Spherically averaged
    profiles and star-only samples do not qualify (plan rule).

    The chunked raw snapshot (data/halo12_raw/snapdir_127, 8 Gadget chunks)
    is checked as ONE aggregate entry: a single chunk is not the complete
    halo, only the full set of chunks is.
    """
    import h5py
    data_dir = Path(data_dir or REPO / "data")
    candidates = sorted(
        {p for p in list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))
         + list((data_dir / "auriga").glob("*.h5"))
         + list((data_dir / "auriga").glob("*.hdf5"))})
    pos_names = {"coordinates", "positions", "pos"}
    xyz_names = {"x", "y", "z"}
    files, qualified = {}, []
    required = ("PartType0", "PartType1", "PartType4")
    for p in candidates:
        try:
            rel = str(p.relative_to(REPO))
        except ValueError:
            rel = str(p)                       # outside the repo (tests)
        try:
            with h5py.File(p, "r") as f:
                comps = {}
                for g in sorted(f.keys()):
                    if not (g.startswith("PartType") and hasattr(f[g], "keys")):
                        continue
                    keys = {k.lower() for k in f[g].keys()}
                    det = sorted(pos_names & keys)
                    if not det and xyz_names <= keys:
                        det = ["x", "y", "z"]     # dpjax star-export schema
                    if det:
                        comps[g] = det
                files[rel] = {"components_with_positions": comps}
                if all(c in comps for c in required):
                    qualified.append(rel)
        except OSError as e:
            files[rel] = {"error": repr(e)}
    # aggregate check of the chunked raw snapshot (single chunks are partial)
    snap_files = [Path(str(RAW_SNAP_TEMPLATE).format(i=i))
                  for i in range(RAW_SNAP_N_CHUNKS)]
    if include_raw_snapshot and all(p.exists() for p in snap_files):
        comps, nfiles_hdr = {}, None
        with h5py.File(snap_files[0], "r") as f:
            nfiles_hdr = int(f["Header"].attrs["NumFilesPerSnapshot"])
            for g in required:
                if g in f and "Coordinates" in f[g]:
                    comps[g] = ["Coordinates"]
        key = "data/halo12_raw/snapdir_127/snapshot_127.[0-7].hdf5 (8 chunks)"
        files[key] = {
            "components_with_positions": comps,
            "n_chunks_present": RAW_SNAP_N_CHUNKS,
            "header_NumFilesPerSnapshot": nfiles_hdr,
            "note": "chunked Gadget/Auriga snapshot; complete only when all "
                    "chunks are combined (any single chunk is partial)",
        }
        if (nfiles_hdr == RAW_SNAP_N_CHUNKS
                and all(c in comps for c in required)):
            qualified.append(key)
    return {
        "rule": "angular truth requires positions for ALL of PartType0/1/4 "
                "(the components summed into M_cum_total) in one snapshot; "
                "star-only samples and 60-shell spherical averages do not "
                "qualify",
        "required_components": list(required),
        "files_scanned": files,
        "qualified_files": qualified,
    }


def kabsch_det(A, B, target_det):
    """Least-squares orthogonal map A -> B with a PRESCRIBED determinant
    (+1: rotation, -1: roto-reflection).  Returns (R, center_of_A, resid):
    x_B ~= R (x_A - center_of_A); resid = per-point residual norms.

    The det=-1 branch exists because the pipeline's raw->model star
    transform turned out to be improper (a mirrored frame); a det=+1-only
    fit cannot recover it (residuals ~ kpc)."""
    Ac, Bc = A - A.mean(axis=0), B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc / len(A))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) * float(target_det)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    res = np.linalg.norm((R @ Ac.T).T + B.mean(axis=0) - B, axis=1)
    return R, A.mean(axis=0), res


def load_raw_snapshot(template=RAW_SNAP_TEMPLATE):
    """All snapshot chunks -> per-component positions [kpc physical] and
    masses [Msun]; PartType4 also returns GFM_StellarFormationTime.

    Unit conversion: Coordinates are Mpc/h, masses 1e10 Msun/h; PartType1
    (dark matter) has no per-particle Masses -> uniform MassTable[1]."""
    import h5py
    out, dm_mass = {}, None
    for pt in ("PartType0", "PartType1", "PartType4"):
        pos_l, m_l, sft_l = [], [], []
        for i in range(RAW_SNAP_N_CHUNKS):
            with h5py.File(str(template).format(i=i), "r") as f:
                h = f["Header"].attrs
                if abs(float(h["HubbleParam"]) - HUBBLE_PARAM) > 1e-6:
                    raise AssertionError(
                        f"snapshot chunk {i} HubbleParam {h['HubbleParam']} "
                        f"!= constant {HUBBLE_PARAM}")
                if pt not in f:
                    continue
                grp = f[pt]
                pos_l.append(grp["Coordinates"][:].astype(np.float64))
                if "Masses" in grp:
                    m_l.append(grp["Masses"][:].astype(np.float64))
                dm_mass = float(h["MassTable"][1]) * 1e10 / HUBBLE_PARAM
                if pt == "PartType4":
                    sft_l.append(
                        grp["GFM_StellarFormationTime"][:].astype(np.float64))
        pos = np.concatenate(pos_l) / HUBBLE_PARAM * 1000.0    # kpc physical
        if m_l:
            mass = np.concatenate(m_l) * 1e10 / HUBBLE_PARAM   # Msun
        else:
            mass = np.full(len(pos), dm_mass)
        if pt == "PartType4":
            out[pt] = (pos, mass, np.concatenate(sft_l))
        else:
            out[pt] = (pos, mass)
    out["dm_particle_mass_msun"] = dm_mass
    return out


def derive_model_frame(template=RAW_SNAP_TEMPLATE,
                       train_h5=REPO / "data/auriga/halo12.h5"):
    """Measure the raw-snapshot -> model-frame transform from the training
    star sample itself: halo12.h5 carries particle_id; matching against the
    raw PartType4 ParticleIDs gives exact correspondences, and a
    determinant-free Kabsch fit (both signs tried) returns the map.

    Empirical finding (2026-09-15): the applied transform is IMPROPER
    (det = -1, a mirrored frame) and differs from the header matrix
    header_Tiv_star (det = +1, ~90 deg away) -- the header does not record
    the transform actually applied upstream.  Gravity is invariant under
    reflection, so applying the SAME measured map to all raw particles puts
    the truth in the model's frame and the mirror cancels in comparisons."""
    import h5py
    with h5py.File(train_h5, "r") as f:
        pid = f["particle_id"][:]
        x_model = f["eta"][:, :3].astype(np.float64) * 10.0   # q -> kpc
    ids_l, pos_l = [], []
    for i in range(RAW_SNAP_N_CHUNKS):
        with h5py.File(str(template).format(i=i), "r") as f:
            ids_l.append(f["PartType4/ParticleIDs"][:])
            pos_l.append(f["PartType4/Coordinates"][:].astype(np.float64))
    raw_id = np.concatenate(ids_l)
    raw_pos = np.concatenate(pos_l) / HUBBLE_PARAM * 1000.0
    order = np.argsort(raw_id)
    srt = raw_id[order]
    hit = np.isin(pid, srt)
    if not hit.all():
        raise AssertionError(
            f"{hit.sum()}/{len(pid)} training stars matched the raw "
            f"PartType4 ParticleIDs -- expected all")
    ridx = order[np.searchsorted(srt, pid[hit])]
    A = raw_pos[ridx] - raw_pos[ridx].mean(axis=0)
    B = x_model[hit]
    fits = {}
    for td in (+1, -1):
        R, c, res = kabsch_det(raw_pos[ridx], B, td)
        fits[td] = (R, c, res)
    det_used = min(fits, key=lambda td: np.median(fits[td][2]))
    R, c, res = fits[det_used]
    if np.median(res) > 0.01 or np.percentile(res, 99) > 0.05:
        raise AssertionError(
            f"model-frame fit residual too large: median {np.median(res):.4g} "
            f"p99 {np.percentile(res, 99):.4g} kpc -- transform not rigid")
    return {
        "M_raw_to_model": R, "center_kpc_raw_frame": c, "det": det_used,
        "n_matched": int(hit.sum()), "n_train": int(len(pid)),
        "residual_median_kpc": float(np.median(res)),
        "residual_p99_kpc": float(np.percentile(res, 99)),
        "residual_median_kpc_det_plus1": float(
            np.median(fits[+1][2])),
    }


def to_model_frame(pos, M, center, chunk=2_000_000):
    """In-place x_model = M (x_raw - center), chunked to bound memory."""
    for i in range(0, len(pos), chunk):
        blk = pos[i:i + chunk] - center
        pos[i:i + chunk] = blk @ M.T
    return pos


def shell_cell_assign(pos_model, mass, r0, fw, n_lat, n_lon):
    """Assign shell particles ([r0/fw, r0*fw)) to equal-area cells.

    Returns dict with cell indices, per-cell mass/count arrays (n_lat, n_lon)
    and shell bookkeeping (edges, volume, mean density)."""
    r = np.linalg.norm(pos_model, axis=1)
    sel = (r >= r0 / fw) & (r < r0 * fw)
    x = pos_model[sel]
    m = mass[sel]
    rr = r[sel]
    mu = np.clip(x[:, 2] / rr, -1.0, 1.0)
    lon = np.arctan2(x[:, 1], x[:, 0])
    mu_edges = np.linspace(-1.0, 1.0, n_lat + 1)
    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    ilat = np.clip(np.searchsorted(mu_edges, mu, side="right") - 1,
                   0, n_lat - 1)
    ilon = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1,
                   0, n_lon - 1)
    idx = ilat * n_lon + ilon
    ncell = n_lat * n_lon
    mass_map = np.bincount(idx, weights=m, minlength=ncell).reshape(
        n_lat, n_lon)
    counts = np.bincount(idx, minlength=ncell).reshape(n_lat, n_lon)
    r_in, r_out = r0 / fw, r0 * fw
    vol = 4.0 * np.pi / 3.0 * (r_out ** 3 - r_in ** 3)
    return {
        "idx": idx, "mass": m, "mass_map": mass_map, "counts": counts,
        "n_shell": int(sel.sum()), "r_in": r_in, "r_out": r_out,
        "volume": vol, "mean_density": float(m.sum() / vol),
        "cell_volume": vol / ncell,
    }


def map_angular_power_ratio(mass_map, lat, lon, l_max=STEP5_LMAX):
    """A_l/A_0 of a (n_lat, n_lon) equal-area map (cell-mass quadrature)."""
    A = sph_harm_power(mass_map.ravel(), directions_from_grid(lat, lon),
                       l_max)
    if not np.isfinite(A[0]) or A[0] == 0.0:
        raise AssertionError("A_0 of the truth map is zero/non-finite")
    return A / A[0]


def bootstrap_shell_power(assign, n_boot, seed, n_lat, n_lon):
    """Particle-resampled A_l/A_0 for one shell (Poisson error bars).

    Resampling rows (mass + cell index together) keeps the per-component
    mixture intact; each draw rebuilds the cell-mass map by bincount."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, STEP5_LMAX))
    n = len(assign["idx"])
    for b in range(n_boot):
        take = rng.integers(0, n, n)
        mm = np.bincount(assign["idx"][take], weights=assign["mass"][take],
                         minlength=n_lat * n_lon).reshape(n_lat, n_lon)
        out[b] = map_angular_power_ratio(mm, *equal_area_map_grid(n_lat,
                                                                  n_lon))[1:]
    return out


def rebin_map(a, f):
    """Block-mean of a (n, m) map by an integer factor f on both axes."""
    n, m = a.shape
    if n % f or m % f:
        raise ValueError(f"rebin factor {f} does not divide {a.shape}")
    return a.reshape(n // f, f, m // f, f).mean(axis=(1, 3))


def step5_angular(audit_dir, models=("baseline", "rin2", "rout65")):
    """Plan step 5: separate angular structure from uncertainty.

    Part A (no model reload): the direction distribution of the per-direction
    integrals persisted by step 3 -- p16/p50/p84 kept explicitly, legend worded
    "direction distribution", mean as the primary curve -- shown separately
    from the step-3 quadrature repeatability (cross-scramble spread of the
    mean estimator).

    Part B (model reload): at the explicit radii 5/10/25/50 kpc, sky maps of
    the model density on an equal-area grid, direction statistics (mean,
    median, negative fraction) on the step-3 Sobol directions, and the
    low-order (l<=4) angular power spectrum; plus per-direction radial-force
    curves on the truth-edge grid with the spherical truth force.

    Truth-side angular structure: eligibility scan for all-source particle
    data; with only a star sample and 60-shell spherical averages the
    conclusion stays "real structure and model error cannot be separated".
    """
    import jax
    import fit_all

    audit_dir = Path(audit_dir)
    out_dir = audit_dir / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    npz_path = audit_dir / "step3" / "mass_profiles.npz"
    conv_path = audit_dir / "step3" / "convergence.json"
    man_path = audit_dir / "step0" / "manifest.json"
    missing = [str(p) for p in (npz_path, conv_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("step5 needs the step-3 products: "
                                + ", ".join(missing))
    conv = json.loads(Path(conv_path).read_text())
    d = np.load(npz_path)
    truth = load_truth()
    edges_out = truth["r_edges"][1:]
    dM_truth = d["dM_truth"]
    focus = d["focus_mask"].astype(bool)
    seeds = [int(s) for s in STEP5_SCRAMBLE_SEEDS]
    L, V = 10.0, 100.0

    lat, lon = equal_area_map_grid()
    # radial-force grid: exactly the grid start + the 60 truth outer edges
    # (the truth outermost edge is NOT exactly 70 kpc, so the grid is built
    # by concatenation, not via an outer endpoint), so the truth spherical
    # force is available at every node beyond the first
    r_force = np.concatenate([[truth["r_edges"][0]], edges_out])
    eidx = d[f"{models[0]}__edge_node_idx"]
    force_truth = finite_or_fail(
        -G_KPC_KMS2_MSUN * truth["M_cum_total"] / edges_out ** 2,
        "truth spherical radial force")
    # quadrature bias floor of the harmonic estimator: A_l/A_0 of a CONSTANT
    # field on the same direction sets.  A measured A_l/A_0 below this floor
    # is indistinguishable from zero (the direction sets are shared by all
    # models, so the floor is model-independent).
    harm_floor = []
    for s in seeds:
        A_c = sph_harm_power(np.full(STEP5_N_DIR, 1.0),
                             sobol_directions(STEP5_N_DIR, scramble_seed=s),
                             STEP5_LMAX)
        harm_floor.append(A_c[1:] / A_c[0])
    harm_floor = np.stack(harm_floor)              # (n_seed, l_max)

    z = {
        "r_edges_outer": edges_out, "dM_truth": dM_truth, "focus_mask": focus,
        "scramble_seeds": np.array(seeds), "n_dir": np.array(STEP5_N_DIR),
        "models": np.array(models), "map_radii": np.array(STEP5_MAP_RADII),
        "map_lat": lat, "map_lon": lon,
        "harm_quadrature_floor": harm_floor,
        "force_r_nodes": r_force, "force_truth": force_truth,
    }

    # ---- part A: direction distribution from the persisted step-3 arrays ----
    band_summary = {}
    for m in models:
        band = pooled_direction_band(d, m, seeds, dM_truth)
        for k in ("p16", "p50", "p84", "mean", "seed_means",
                  "rep_min", "rep_max"):
            z[f"{m}__band_{k}"] = finite_or_fail(band[k], f"band {m} {k}")
        half = 0.5 * (band["p84"] - band["p16"])
        j_rad = [int(np.argmin(np.abs(edges_out - r))) for r in STEP5_MAP_RADII]
        band_summary[m] = {
            "n_directions_pooled": band["n_directions_pooled"],
            "direction_halfwidth_p84_p16_focus_max": float(half[focus].max()),
            "argmax_r_kpc": float(edges_out[focus][np.argmax(half[focus])]),
            "direction_halfwidth_at_map_radii": {
                f"{r:g}": float(half[j]) for r, j in zip(STEP5_MAP_RADII, j_rad)},
            "repeatability_halfwidth_focus_max": float(
                (0.5 * np.ptp(band["seed_means"], axis=0))[focus].max()),
            "labels": {
                "band": "direction distribution (angular structure of the "
                        "model field), NOT statistical uncertainty",
                "envelope": "quadrature repeatability of the mean estimator "
                            "(cross-scramble spread), NOT a confidence "
                            "interval"},
        }

    # ---- part B: model reload for maps / direction stats / radial force ----
    fixed_summary = {}
    force_summary = {}
    map_summary = {}
    for m in models:
        print(f"[step5] model {m}: loading Phi ...")
        phi_func = _load_phi_for_step3(RUNS[m])
        dirs_per_seed = {s: sobol_directions(STEP5_N_DIR, scramble_seed=s)
                         for s in seeds}

        # fixed-radius direction statistics + low-order angular power
        fixed_summary[m] = {}
        rho_pool, fr_pool = {}, {}
        harm_seeds = {r: [] for r in STEP5_MAP_RADII}
        stat_seeds = {r: [] for r in STEP5_MAP_RADII}
        for s in seeds:
            rho_dir, dn_dir = per_direction_rho_and_dn(
                phi_func, np.array(STEP5_MAP_RADII), dirs_per_seed[s], L, V)
            a_r = -(V ** 2 / L) * dn_dir        # radial acceleration, inward <
            for j, r in enumerate(STEP5_MAP_RADII):
                stat_seeds[r].append({
                    "mean": float(rho_dir[:, j].mean()),
                    "median": float(np.median(rho_dir[:, j])),
                    "neg_fraction": float((rho_dir[:, j] < 0).mean()),
                })
                harm_seeds[r].append(sph_harm_power(rho_dir[:, j],
                                                    dirs_per_seed[s],
                                                    STEP5_LMAX))
            rho_pool[s], fr_pool[s] = rho_dir, a_r
        for j, r in enumerate(STEP5_MAP_RADII):
            pooled_rho = np.concatenate([rho_pool[s][:, j] for s in seeds])
            pooled_fr = np.concatenate([fr_pool[s][:, j] for s in seeds])
            hs = np.stack(harm_seeds[r])       # (n_seed, l_max+1)
            ratio = hs / hs[:, [0]]
            z[f"{m}__rho_dir_r{r:g}"] = finite_or_fail(pooled_rho, f"{m} rho {r}")
            z[f"{m}__fr_dir_r{r:g}"] = finite_or_fail(pooled_fr, f"{m} fr {r}")
            z[f"{m}__harm_ratio_seeds_r{r:g}"] = finite_or_fail(
                ratio, f"{m} harm {r}")
            smean = np.array([st["mean"] for st in stat_seeds[r]])
            fixed_summary[m][f"{r:g}"] = {
                "mean_rho_msun_kpc3": float(pooled_rho.mean()),
                "median_rho_msun_kpc3": float(np.median(pooled_rho)),
                "p16_rho": float(np.percentile(pooled_rho, 16)),
                "p84_rho": float(np.percentile(pooled_rho, 84)),
                "neg_fraction": float((pooled_rho < 0).mean()),
                "mean_seed_to_seed_rel_spread": float(
                    np.ptp(smean) / smean.mean()) if smean.size else None,
                "angular_power_ratio_A_l_over_A0": {
                    f"l{l}": {"median_over_seeds": float(np.median(ratio[:, l])),
                              "min": float(ratio[:, l].min()),
                              "max": float(ratio[:, l].max())}
                    for l in range(1, STEP5_LMAX + 1)},
                "force_mean": float(pooled_fr.mean()),
                "force_p16": float(np.percentile(pooled_fr, 16)),
                "force_p84": float(np.percentile(pooled_fr, 84)),
            }

        # equal-area sky maps of the model density (diagnostic, model only)
        map_summary[m] = {}
        for r in STEP5_MAP_RADII:
            rho_map = density_map(phi_func, r, lat, lon, L, V)
            wmean, wneg = map_stats(rho_map)
            mean_sobol = fixed_summary[m][f"{r:g}"]["mean_rho_msun_kpc3"]
            z[f"{m}__map_rho_r{r:g}"] = rho_map
            z[f"{m}__map_meansobol_r{r:g}"] = np.array(mean_sobol)
            map_summary[m][f"{r:g}"] = {
                "map_mean": wmean,
                "map_vs_sobol_mean_rel_diff": float(abs(wmean / mean_sobol
                                                        - 1.0)),
                "map_neg_fraction": wneg,
                "map_min": float(rho_map.min()), "map_max": float(rho_map.max()),
            }

        # radial-force curves on the truth-edge grid + step-3 cross-check
        seed_means = []
        pooled_all = []
        for s in seeds:
            _, dn_dir = per_direction_rho_and_dn(
                phi_func, r_force, dirs_per_seed[s], L, V)
            a_r = -(V ** 2 / L) * dn_dir
            seed_means.append(a_r.mean(axis=0))
            pooled_all.append(a_r)
            # consistency: the mean flux implied here must reproduce the
            # persisted step-3 final evaluation at the shared edge nodes
            m_flux_here = (dn_dir.mean(axis=0) * r_force ** 2 * V ** 2
                           / (G_KPC_KMS2_MSUN * L))
            ref = d[f"{m}__seed{s}__m_flux_nodes"][eidx]
            if not np.allclose(m_flux_here[1:], ref, rtol=1e-6, atol=0.0):
                raise AssertionError(
                    f"step5 force evaluation disagrees with the persisted "
                    f"step-3 flux for {m} seed {s}: max rel "
                    f"{np.max(np.abs(m_flux_here[1:] / ref - 1.0)):.3e}")
        pooled = np.concatenate(pooled_all, axis=0)
        seed_means = np.stack(seed_means)
        z[f"{m}__force_mean"] = finite_or_fail(pooled.mean(axis=0), "force mean")
        for q in (16, 50, 84):
            z[f"{m}__force_p{q}"] = finite_or_fail(
                np.percentile(pooled, q, axis=0), f"force p{q}")
        z[f"{m}__force_seedmeans"] = finite_or_fail(seed_means, "force seeds")
        # mean vs spherical truth over the focus region (truth M(<r) at edges)
        rel = np.abs(z[f"{m}__force_mean"][1:] - force_truth) / np.abs(force_truth)
        force_summary[m] = {
            "mean_vs_truth_spherical_focus_max": float(rel[focus].max()),
            "argmax_r_kpc": float(edges_out[focus][np.argmax(rel[focus])]),
            "direction_halfwidth_at_map_radii": {
                f"{r:g}": float(0.5 * np.ptp(np.percentile(
                    pooled[:, int(np.argmin(np.abs(r_force - r)))],
                    [16, 84])))
                for r in STEP5_MAP_RADII},
            "consistency_check": "mean flux at the 60 truth edges reproduces "
                                 "the persisted step-3 m_flux_nodes (rtol "
                                 "1e-6 asserted)",
        }

    inventory = angular_truth_inventory()
    if inventory["qualified_files"]:
        angular_truth_conclusion = (
            "qualified all-source data found: "
            + ", ".join(inventory["qualified_files"])
            + "; the raw->model-frame transform (measured, includes a "
              "mirror) and the particle angular truth at the map radii are "
              "produced by `step5-truth` -> step5/angular_truth.json; the "
              "earlier 'real structure and model error cannot be separated' "
              "conclusion is superseded")
    else:
        angular_truth_conclusion = (
            "no file provides positions for all of PartType0/1/4; the "
            "available particle data are a star-only sample (PartType4) and "
            "the truth HDF5 holds 60-shell SPHERICAL averages per component; "
            "per the plan a star sample cannot serve as the total-density "
            "angular truth, so real angular structure and model error "
            "CANNOT be separated")

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_step": 5,
        "question": "what does the wide direction band mean?",
        "part_a": {
            "definition": "p16/p50/p84 of the per-direction volume-path "
                          "DeltaM(r;0.5)/DeltaM_truth pooled over 4 scrambles "
                          "x 2048 directions; primary curve = direction MEAN; "
                          "the *_at_map_radii summaries are read at the truth "
                          "edge nearest each requested radius",
            "per_model": band_summary,
        },
        "part_b": {
            "radii_kpc": list(STEP5_MAP_RADII),
            "direction_stats_definition": "statistics over the pooled Sobol "
                                          "directions (4 scrambles x 2048), "
                                          "identical machinery to step 3",
            "angular_power_definition": "A_l = sqrt(sum_m |a_lm|^2), a_lm = "
                                       "<rho Y_lm*>_Omega; reported as "
                                       "A_l/A_0 per seed (median/min/max); "
                                       "values below harm_quadrature_floor "
                                       "are indistinguishable from zero",
            "sky_map_definition": "equal-area cell-centre grid (180 x 360, "
                                  "uniform in sin lat), MODEL density only",
            "radial_force_definition": "a_r = -(V^2/L) n.grad_q phi per "
                                       "direction on the truth-edge grid; "
                                       "truth line = spherical field of the "
                                       "spherically averaged truth profile",
            "per_model_fixed_radius": fixed_summary,
            "per_model_maps": map_summary,
            "per_model_force": force_summary,
        },
        "angular_truth_availability": inventory,
        "angular_truth_conclusion": angular_truth_conclusion,
        "numerics": {
            "jax_enable_x64_host": True, "phi_params_dtype": "float32",
            "model_eval": "same regime as step 3: f64 inputs x f32 weights "
                          "-> f64 arithmetic",
            "n_dir": STEP5_N_DIR, "scramble_seeds": list(seeds),
            "l_max": STEP5_LMAX,
            "harm_quadrature_floor_A_l_over_A0": {
                f"l{l+1}": {"median": float(np.median(harm_floor[:, l])),
                            "max": float(harm_floor[:, l].max())}
                for l in range(STEP5_LMAX)},
        },
        "inputs": {
            "step3_npz": {"path": str(npz_path),
                          "sha256": sha256_file(npz_path)},
            "convergence_json": {"path": str(conv_path),
                                 "sha256": sha256_file(conv_path)},
            "truth_h5": {"path": str(TRUTH_PATH),
                         "sha256": sha256_file(TRUTH_PATH)},
        },
        "checkpoints": {},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    if man_path.exists():
        with open(man_path) as f:
            man = json.load(f)
        result["checkpoints"] = {
            m: {"path": man["runs"][m]["phi_checkpoint"]["path"],
                "sha256": man["runs"][m]["phi_checkpoint"]["sha256"]}
            for m in models}

    np.savez_compressed(out_dir / "angular_diagnostics.npz", **z)
    _dump_json(out_dir / "angular_diagnostics.json", result)
    print(f"[step5] direction band halfwidth (focus max): "
          + ", ".join(f"{m}="
                      f"{band_summary[m]['direction_halfwidth_p84_p16_focus_max']:.3f}"
                      for m in models))
    print(f"[step5] force mean vs spherical truth (focus max): "
          + ", ".join(f"{m}="
                      f"{force_summary[m]['mean_vs_truth_spherical_focus_max']:.3f}"
                      for m in models))
    print(f"[step5] angular truth: qualified files = "
          f"{inventory['qualified_files'] or 'none'} -> {angular_truth_conclusion}")
    print(f"[step5] done -> {out_dir / 'angular_diagnostics.json'} "
          f"(elapsed {result['elapsed_s']}s)")
    return result


def step5_truth(audit_dir, models=("baseline", "rin2", "rout65")):
    """Plan step 5, truth side: angular structure of the REAL total density
    from the all-source raw snapshot, in the model frame, compared against
    the step-5 model diagnostics.

    Chain of evidence (all asserted):
    1. frame: raw->model transform measured from ID-matched training stars
       (improper Kabsch; residual asserted < 0.01 kpc median);
    2. spherical: per-component M_cum from the particles reproduces the
       60-shell truth HDF5 (gas/dm asserted < 2% median beyond 1 kpc; the
       stellar convention difference is reported, not asserted);
    3. angular: shell mass maps on an equal-area grid at 5/10/25/50 kpc,
       A_l/A_0 with particle-bootstrap error bars, and Pearson correlation
       against the step-5 model maps (rebinned to the truth grid)."""
    audit_dir = Path(audit_dir)
    out_dir = audit_dir / "step5"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    step5_npz = out_dir / "angular_diagnostics.npz"
    step5_json = out_dir / "angular_diagnostics.json"
    missing = [str(p) for p in (step5_npz, step5_json) if not p.exists()]
    if missing:
        raise FileNotFoundError("step5-truth needs the step-5 products: "
                                + ", ".join(missing))

    # ---- 1. frame ----
    print("[step5-truth] deriving the raw->model frame from training stars")
    frame = derive_model_frame()
    if frame["det"] != -1:
        raise AssertionError(
            f"expected the improper (det=-1) transform; got det="
            f"{frame['det']} -- the frame finding changed, re-examine")
    print(f"[step5-truth] frame: det={frame['det']}, matched "
          f"{frame['n_matched']}/{frame['n_train']} stars, residual median "
          f"{frame['residual_median_kpc']:.4f} kpc (det=+1 fit would give "
          f"{frame['residual_median_kpc_det_plus1']:.3f} kpc)")
    M, c_raw = frame["M_raw_to_model"], frame["center_kpc_raw_frame"]

    # ---- 2. load all components into the model frame ----
    print("[step5-truth] loading snapshot particles (8 chunks)")
    raw = load_raw_snapshot()
    comp_masks = {}
    for pt in ("PartType0", "PartType1", "PartType4"):
        pos, mass = raw[pt][0], raw[pt][1]
        to_model_frame(pos, M, c_raw)
        if pt == "PartType4":
            comp_masks[pt + "_formed"] = raw[pt][2] > 0.0
            comp_masks[pt + "_all"] = np.ones(len(pos), bool)
        else:
            comp_masks[pt] = np.ones(len(pos), bool)
        print(f"[step5-truth]   {pt}: {len(pos)} particles in the model frame")

    # spherical validation against the 60-shell truth HDF5
    truth = load_truth()
    edges = truth["r_edges"][1:]
    spher = {}
    for name, (pt, mask) in {
            "PartType0": ("PartType0", comp_masks["PartType0"]),
            "PartType1": ("PartType1", comp_masks["PartType1"]),
            "PartType4": ("PartType4", comp_masks["PartType4_formed"]),
            "PartType4_all": ("PartType4", comp_masks["PartType4_all"]),
    }.items():
        pos, mass = raw[pt][0], raw[pt][1]
        r = np.linalg.norm(pos[mask], axis=1)
        order = np.argsort(r)
        rb = r[order]
        mc = np.cumsum(mass[mask][order])
        Mc = mc[np.searchsorted(rb, edges, side="right") - 1]
        ref = truth[name if name in truth else "PartType4"]["M_cum"]
        rel = np.abs(Mc - ref) / np.maximum(np.abs(ref), 1.0)
        keep = edges > 1.0        # innermost shells are discreteness-limited
        spher[name] = {
            "M_cum_at_400kpc_msun": float(Mc[-1]),
            "ref_M_cum_at_400kpc_msun": float(ref[-1]),
            "rel_median_beyond_1kpc": float(np.median(rel[keep])),
            "rel_max_beyond_1kpc": float(rel[keep].max()),
        }
    for gas_dm in ("PartType0", "PartType1"):
        if spher[gas_dm]["rel_median_beyond_1kpc"] > 0.02:
            raise AssertionError(
                f"spherical reproduction of {gas_dm} worse than 2% median: "
                f"{spher[gas_dm]['rel_median_beyond_1kpc']:.3e}")
    tot = np.array([spher[k]["M_cum_at_400kpc_msun"] for k in
                    ("PartType0", "PartType1", "PartType4")]).sum()
    spher["total_P0P1P4formed_vs_truth_total"] = {
        "M_cum_at_400kpc_msun": float(tot),
        "rel": float(tot / truth["M_cum_total"][-1] - 1.0),
    }
    print("[step5-truth] spherical check: gas/dm median rel "
          + ", ".join(f"{k}={spher[k]['rel_median_beyond_1kpc']:.1e}"
                      for k in ("PartType0", "PartType1"))
          + f"; stars(formed) {spher['PartType4']['rel_median_beyond_1kpc']:.1e}"
            f" (convention difference, reported not asserted)")

    # ---- 3. shells: angular maps, harmonics, bootstrap ----
    n_lat, n_lon = STEP5_TRUTH_GRID
    tlat, tlon = equal_area_map_grid(n_lat, n_lon)
    z = {
        "truth_grid_lat": tlat, "truth_grid_lon": tlon,
        "map_radii": np.array(STEP5_MAP_RADII),
        "frame_M_raw_to_model": M, "frame_center_kpc_raw_frame": c_raw,
        "frame_det": np.array(frame["det"]),
    }
    shell_summary = {}
    for r0 in STEP5_MAP_RADII:
        per_comp, total_idx, total_mass = {}, [], []
        for name in ("PartType0", "PartType1", "PartType4"):
            mask = comp_masks[name + "_formed"] if name == "PartType4" \
                else comp_masks[name]
            pos, mass = raw[name][0], raw[name][1]
            a = shell_cell_assign(pos[mask], mass[mask], r0,
                                  STEP5_TRUTH_SHELL_FW, n_lat, n_lon)
            per_comp[name] = {
                "n": a["n_shell"], "mass_msun": float(a["mass"].sum()),
                "mean_density": a["mean_density"],
            }
            z[f"truth_map_rho_{name}_r{r0:g}"] = a["mass_map"] / a[
                "cell_volume"]
            total_idx.append(a["idx"])
            total_mass.append(a["mass"])
        assign = {"idx": np.concatenate(total_idx),
                  "mass": np.concatenate(total_mass)}
        mass_map = np.bincount(assign["idx"], weights=assign["mass"],
                               minlength=n_lat * n_lon).reshape(n_lat, n_lon)
        counts = np.bincount(assign["idx"], minlength=n_lat * n_lon
                             ).reshape(n_lat, n_lon)
        rho_map = mass_map / (np.pi * 4.0 / 3.0 * ((r0 * STEP5_TRUTH_SHELL_FW)
                             ** 3 - (r0 / STEP5_TRUTH_SHELL_FW) ** 3)
                             / (n_lat * n_lon))
        ratio = map_angular_power_ratio(mass_map, tlat, tlon)
        boot = bootstrap_shell_power(assign, STEP5_TRUTH_BOOTSTRAP,
                                     STEP5_TRUTH_BOOT_SEED + int(r0),
                                     n_lat, n_lon)
        # systematic variant: all-PartType4 (wind included) total harmonics
        a0 = shell_cell_assign(raw["PartType0"][0][comp_masks["PartType0"]],
                               raw["PartType0"][1][comp_masks["PartType0"]],
                               r0, STEP5_TRUTH_SHELL_FW, n_lat, n_lon)
        a1 = shell_cell_assign(raw["PartType1"][0][comp_masks["PartType1"]],
                               raw["PartType1"][1][comp_masks["PartType1"]],
                               r0, STEP5_TRUTH_SHELL_FW, n_lat, n_lon)
        pos4, mass4 = raw["PartType4"][0], raw["PartType4"][1]
        a4 = shell_cell_assign(pos4, mass4, r0, STEP5_TRUTH_SHELL_FW,
                               n_lat, n_lon)
        mm_all4 = a0["mass_map"] + a1["mass_map"] + a4["mass_map"]
        ratio_all4 = map_angular_power_ratio(mm_all4, tlat, tlon)

        z[f"truth_map_rho_r{r0:g}"] = finite_or_fail(rho_map, f"truth {r0}")
        z[f"truth_counts_r{r0:g}"] = counts
        z[f"truth_A_ratio_r{r0:g}"] = ratio[1:]
        z[f"truth_boot_ratio_r{r0:g}"] = boot
        shell_summary[f"{r0:g}"] = {
            "shell_edges_kpc": [r0 / STEP5_TRUTH_SHELL_FW,
                                r0 * STEP5_TRUTH_SHELL_FW],
            "n_particles": {k: v["n"] for k, v in per_comp.items()},
            "mass_msun": {k: v["mass_msun"] for k, v in per_comp.items()},
            "counts_per_cell": {
                "median": float(np.median(counts)),
                "p05": float(np.percentile(counts, 5)),
                "fraction_below_5": float((counts < 5).mean())},
            "mean_density_msun_kpc3": float(rho_map.mean()),
            "A_l_over_A0": {f"l{l}": float(ratio[l])
                            for l in range(1, STEP5_LMAX + 1)},
            "A_l_over_A0_bootstrap_std": {
                f"l{l}": float(boot[:, l - 1].std())
                for l in range(1, STEP5_LMAX + 1)},
            "A_l_over_A0_allP4_variant": {
                f"l{l}": float(ratio_all4[l])
                for l in range(1, STEP5_LMAX + 1)},
        }
        print(f"[step5-truth] r={r0:g} kpc: N=" +
              "/".join(f"{v['n']}" for v in per_comp.values())
              + f", A2/A0={ratio[2]:.3f}+/-{boot[:, 1].std():.3f}")

    # ---- 4. model comparison (rebin 180x360 -> truth grid by 4x4) ----
    dm5 = np.load(step5_npz)
    with open(step5_json) as f:
        rep5 = json.load(f)
    comparison = {}
    for m in models:
        comparison[m] = {}
        for r0 in STEP5_MAP_RADII:
            model_map = rebin_map(dm5[f"{m}__map_rho_r{r0:g}"], 4)
            truth_map = z[f"truth_map_rho_r{r0:g}"]
            a = model_map / model_map.mean() - 1.0
            b = truth_map / truth_map.mean() - 1.0
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            model_A = rep5["part_b"]["per_model_fixed_radius"][m][
                f"{r0:g}"]["angular_power_ratio_A_l_over_A0"]
            comparison[m][f"{r0:g}"] = {
                "pearson_r_normalized_maps": r,
                "model_over_truth_mean_density": float(
                    dm5[f"{m}__map_meansobol_r{r0:g}"]
                    / truth_map.mean()),
                "A_l_over_A0_model_median": {
                    f"l{l}": model_A[f"l{l}"]["median_over_seeds"]
                    for l in range(1, STEP5_LMAX + 1)},
                "A_l_over_A0_truth": shell_summary[f"{r0:g}"][
                    "A_l_over_A0"],
            }
    # headline numbers for the printout/conclusion
    r5 = comparison[models[0]]["5"]["pearson_r_normalized_maps"]
    a2t = {rr: shell_summary[f"{rr:g}"]["A_l_over_A0"]["l2"]
           for rr in STEP5_MAP_RADII}

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_step": "5b",
        "question": "what is the REAL angular structure, and how much of the "
                    "model's is error?",
        "frame": {
            "method": "improper-Kabsch on ID-matched training stars "
                      "(halo12.h5 particle_id vs raw PartType4 ParticleIDs)",
            "det": frame["det"],
            "matrix_raw_to_model": M.tolist(),
            "center_kpc_raw_frame": c_raw.tolist(),
            "n_matched": frame["n_matched"], "n_train": frame["n_train"],
            "residual_median_kpc": frame["residual_median_kpc"],
            "residual_p99_kpc": frame["residual_p99_kpc"],
            "residual_median_kpc_if_det_plus1":
                frame["residual_median_kpc_det_plus1"],
            "notes": "applied transform is a MIRROR (det=-1) and differs "
                     "from halo12.h5 header_Tiv_star (det=+1, ~90 deg away); "
                     "gravity is reflection-invariant and the SAME map is "
                     "applied to every component, so the mirror cancels in "
                     "model-vs-truth comparisons",
        },
        "spherical_validation": spher,
        "star_convention_caveat": "primary truth uses formed stars "
                                  "(SFT>0, matching the 0-wind training "
                                  "sample); the truth HDF5's PartType4 "
                                  "convention shows a residual 1.5-3% mass "
                                  "difference (upstream selection not in "
                                  "this repo); gas/dm reproduce to <2% "
                                  "median, asserted",
        "shells": {
            "grid": [n_lat, n_lon],
            "shell_width_relative": STEP5_TRUTH_SHELL_FW,
            "bootstrap_draws": STEP5_TRUTH_BOOTSTRAP,
            "per_radius": shell_summary,
        },
        "model_comparison": {
            "definition": "model step-5 maps rebinned 4x4 onto the truth "
                          "grid; Pearson r of mean-normalised maps; "
                          "A_l/A_0 from each side's own estimator",
            "per_model": comparison,
        },
        "angular_truth_conclusion": (
            f"angular truth MEASURED from the all-source snapshot in the "
            f"model frame; truth A2/A0 = "
            + ", ".join(f"{rr:g}kpc:{a2t[rr]:.3f}" for rr in STEP5_MAP_RADII)
            + f"; model-vs-truth normalized-map Pearson r (baseline) = "
              f"{r5:.3f} at 5 kpc; real angular structure and model angular "
              f"error are now SEPARATED at l<=4 (per-radius values in "
              f"model_comparison)"),
        "numerics": {
            "particle_grid": [n_lat, n_lon],
            "shot_noise_note": "per-cell Poisson dominates the pixel maps "
                               "(median counts in shells summary); A_l/A_0 "
                               "error bars are particle bootstrap",
            "hubble_param": HUBBLE_PARAM,
        },
        "inputs": {
            "snapshot_chunks": {
                str(Path(str(RAW_SNAP_TEMPLATE).format(i=i)).relative_to(
                    REPO)): {"sha256": sha256_file(
                        str(RAW_SNAP_TEMPLATE).format(i=i))}
                for i in range(RAW_SNAP_N_CHUNKS)},
            "train_h5": {"path": str(REPO / "data/auriga/halo12.h5"),
                         "sha256": sha256_file(REPO / "data/auriga/"
                                               "halo12.h5")},
            "truth_h5": {"path": str(TRUTH_PATH),
                         "sha256": sha256_file(TRUTH_PATH)},
            "step5_npz": {"path": str(step5_npz),
                          "sha256": sha256_file(step5_npz)},
        },
        "elapsed_s": round(time.time() - t_start, 1),
    }

    np.savez_compressed(out_dir / "angular_truth.npz", **z)
    _dump_json(out_dir / "angular_truth.json", result)
    print(f"[step5-truth] done -> {out_dir / 'angular_truth.json'} "
          f"(elapsed {result['elapsed_s']}s)")
    return result


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("step", choices=["step0-manifest", "step1-analytic",
                                         "step2-ablation", "step3-convergence",
                                         "step4-anchor", "step5-angular",
                                         "step5-truth"])
    parser.add_argument("--audit-dir", type=Path,
                        default=REPO / "runs/halo12-mass-audit-20260914T0453")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output file for this step "
                             "(default: refuse, plan rule 'no overwriting')")
    parser.add_argument("--n-dirs", type=str, default=None,
                        help="step3 only: comma-separated ascending direction "
                             "levels, e.g. 256,512,1024,2048 (plan caps the "
                             "escalation at 2048)")
    args = parser.parse_args()

    # x64 is a global property of the verification machinery; set it once at
    # the entry point so later steps cannot silently run the model in f32
    # while the host accumulates in f64 (plan warning on mixed precision).
    import jax
    jax.config.update("jax_enable_x64", True)

    expected = {"step0-manifest": "step0/manifest.json",
                "step1-analytic": "step1/analytic_checks.json",
                "step2-ablation": "step2/legacy_ablation.json",
                "step3-convergence": "step3/convergence.json",
                "step4-anchor": "step4/anchor_comparison.json",
                "step5-angular": "step5/angular_diagnostics.json",
                "step5-truth": "step5/angular_truth.json"}
    if args.step in expected:
        target = args.audit_dir / expected[args.step]
        if target.exists() and not args.force:
            print(f"REFUSING to overwrite existing product {target}; "
                  "use --force to replace it explicitly")
            return 1

    if args.step == "step0-manifest":
        step0_manifest(args.audit_dir)
    elif args.step == "step1-analytic":
        step1_analytic(args.audit_dir)
    elif args.step == "step2-ablation":
        step2_ablation(args.audit_dir)
    elif args.step == "step3-convergence":
        n_dirs = (tuple(int(x) for x in args.n_dirs.split(","))
                  if args.n_dirs else STEP3_N_DIRS)
        step3_convergence(args.audit_dir, n_dirs=n_dirs)
    elif args.step == "step4-anchor":
        step4_anchor(args.audit_dir)
    elif args.step == "step5-angular":
        step5_angular(args.audit_dir)
    elif args.step == "step5-truth":
        step5_truth(args.audit_dir)
    else:  # pragma: no cover - argparse rejects anything else first
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
