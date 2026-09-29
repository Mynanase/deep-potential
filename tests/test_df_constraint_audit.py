#!/usr/bin/env python
"""Tests for the T1 DF score-chain audit machinery and cache contract.

These validate the AUDITOR on analytic examples (CBE residual algebra in
the alpha convention, weighted SVD local force, absorbable projection,
finite-difference adjacent-step stability, stratification, floored relative
errors, x64 toggling, batching, hash gates, the frozen point-set designs
and the point-order hash), plus the model plumbing (gradient functions,
strict-ODE grafting, split identities) on a tiny locally built CNF of the
same class as the control.  No trained checkpoints are touched.

Run from the repo root:  JAX_PLATFORMS=cpu python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import json

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import audit_df_constraints as adc  # noqa: E402
import nf_score_cache as nsc  # noqa: E402


# ------------------------------------------------------- analytic steady DF
# lnF(q, p) = -|p|^2 / 2 - phi_a(q),  phi_a(q) = 0.5 * ln(1 + q.q)
# s_q = -grad phi_a,  s_p = -p;  with alpha = -grad phi_a the steady-state
# CBE holds exactly in code units.

def _grad_phi_a(q):
    return q / (1.0 + np.sum(q * q, axis=-1, keepdims=True))


@pytest.fixture(scope="module", autouse=True)
def _jax_x64_default():
    import jax
    jax.config.update("jax_enable_x64", True)
    yield


def _make_points(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 3))
    p = rng.normal(size=(n, 3))
    return q, p


# ---------------------------------------------------------------- small utils

def test_rel_with_floor_bounds_spike_at_zero():
    ref = np.array([1.0, 0.0, 1e-9])
    diff = np.array([1e-4, 1e-4, 1e-4])
    out = adc.rel_with_floor(diff, ref, floor=1e-3)
    assert out[0] == 1e-4
    assert out[1] == 0.1
    assert out[2] == 0.1


def test_component_floors_are_medians():
    stored = np.array([[1.0, -10.0], [3.0, 10.0], [100.0, 10.0]])
    np.testing.assert_allclose(adc.component_floors(stored), [3.0, 10.0])


def test_stratified_rows_by_radius_spans_range():
    rng = np.random.default_rng(1)
    eta = np.zeros((1000, 6))
    eta[:, :3] = rng.normal(size=(1000, 3))
    rows = adc.stratified_rows_by_radius(eta, 37)
    assert len(rows) == 37
    r = np.linalg.norm(eta[:, :3], axis=1)
    r_sel = r[rows]
    assert np.all(np.diff(r_sel) >= 0)
    assert r_sel[0] == r.min() and r_sel[-1] == r.max()


def test_band_labels_partition():
    edges = np.array([0.1, 0.2, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0])
    r = np.array([0.05, 0.15, 0.5, 1.5, 2.5, 4.0, 5.0, 6.5, 9.0])
    b = adc.band_labels(r, edges)
    assert b.tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 6]


# ------------------------------------------------- alpha convention and CBE

def test_acceleration_sign_conversion():
    rng = np.random.default_rng(7)
    dphi = rng.normal(size=(5, 3))
    np.testing.assert_allclose(adc.acceleration_from_grad_phi(dphi), -dphi)


def test_cbe_residual_alpha_zero_for_steady_df():
    q, p = _make_points(64)
    alpha = -_grad_phi_a(q)
    term1, term2, r, rel = adc.cbe_residual_alpha(p, -_grad_phi_a(q), -p, alpha)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    np.testing.assert_allclose(rel, 0.0, atol=1e-12)
    # gradient-form equivalence: R = p.s_q - (grad phi).s_p
    r_grad = np.sum(p * -_grad_phi_a(q), axis=-1) - np.sum(_grad_phi_a(q) * -p, axis=-1)
    np.testing.assert_allclose(r, r_grad, atol=1e-12)


def test_cbe_residual_alpha_detects_wrong_acceleration():
    q, p = _make_points(64)
    r_wrong = adc.cbe_residual_alpha(p, -_grad_phi_a(q), -p, 2.0 * -_grad_phi_a(q))[2]
    assert np.median(np.abs(r_wrong)) > 1e-2


# ------------------------------------------------------- SVD local force

def _steady_constraints(q, p, scale=1.0):
    """C rows = s_p = -p; y = -p . s_q with s_q = -scale * grad phi_a."""
    C = -p
    g = _grad_phi_a(np.repeat(q[None], len(p), axis=0))
    y = -np.sum(p * (-scale * g), axis=1)
    return C, y


def test_local_force_svd_alpha_recovers_exact_force():
    rng = np.random.default_rng(2)
    q = rng.normal(size=3) * 0.5
    p = rng.normal(size=(64, 3))
    C, y = _steady_constraints(q, p)
    d = adc.local_force_svd_alpha(C, y)
    assert d["rank"] == 3
    assert d["resid_rel"] < 1e-10
    np.testing.assert_allclose(d["alpha_hat"], -_grad_phi_a(q[None, :])[0], rtol=1e-10)


def test_local_force_svd_alpha_sign_convention():
    # flipping the sign of y solves for -alpha: the y = -p.s_q convention is
    # load-bearing, not cosmetic
    rng = np.random.default_rng(3)
    q = rng.normal(size=3)
    p = rng.normal(size=(64, 3))
    C, y = _steady_constraints(q, p)
    a1 = adc.local_force_svd_alpha(C, y)["alpha_hat"]
    a2 = adc.local_force_svd_alpha(C, -y)["alpha_hat"]
    np.testing.assert_allclose(a2, -a1, rtol=1e-10)


def test_local_force_svd_alpha_weighted_matches_analytic():
    rng = np.random.default_rng(4)
    q = rng.normal(size=3)
    m = 48
    p = rng.normal(size=(m, 3))
    w = rng.uniform(0.2, 5.0, size=m)
    C, y = _steady_constraints(q, p)
    d = adc.local_force_svd_alpha(C, y, weights=w)
    # analytic weighted least squares normal equations
    W = np.diag(w)
    a_ref = np.linalg.solve(C.T @ W @ C, C.T @ W @ y)
    np.testing.assert_allclose(d["alpha_hat"], a_ref, rtol=1e-9)
    d_unit = adc.local_force_svd_alpha(C, y)
    np.testing.assert_allclose(d_unit["alpha_hat"],
                               adc.local_force_svd_alpha(C, y, weights=np.ones(m))["alpha_hat"])


def test_local_force_svd_alpha_rank_deficient_suppresses_alpha():
    rng = np.random.default_rng(5)
    p = rng.normal(size=(32, 3))
    p[:, 2] = 0.0                 # velocities confined to a plane
    alpha_true = np.array([0.3, -0.2, 0.7])
    C = -p
    y = p @ alpha_true
    d = adc.local_force_svd_alpha(C, y)
    assert d["rank"] == 2
    assert d["alpha_hat"] is None  # no stable force estimate emitted


# ------------------------------------------------ absorbable projection

def test_absorbable_split_adds_up_and_orthogonal():
    rng = np.random.default_rng(6)
    C = rng.normal(size=(40, 3))
    e = rng.normal(size=40)
    w = rng.uniform(0.5, 2.0, size=40)
    split = adc.absorbable_split(C, e, weights=w)
    np.testing.assert_allclose(split["absorbable"] + split["orthogonal"], e, atol=1e-12)
    # the absorbable part is exactly reachable by a force perturbation:
    # C @ dalpha reproduces it (in weighted space)
    W2 = np.diag(np.sqrt(w))
    dalpha = np.linalg.lstsq(W2 @ C, W2 @ split["absorbable"], rcond=None)[0]
    np.testing.assert_allclose((W2 @ C) @ dalpha, W2 @ split["absorbable"], atol=1e-9)
    # the orthogonal part cannot be absorbed: LS on it leaves everything
    resid = (W2 @ C) @ np.linalg.lstsq(W2 @ C, W2 @ split["orthogonal"], rcond=None)[0]
    assert np.linalg.norm(resid) < 1e-9 * max(1.0, np.linalg.norm(W2 @ split["orthogonal"]))


# ---------------------------------------------------------------- FD machinery

def test_central_fd_grad_quadratic_exact():
    fn = lambda x: float(np.sum(x * x) + 3.0 * x[0])
    x = np.array([0.3, -1.2, 0.7, 0.0, 2.0, -0.5])
    g = adc.central_fd_grad(fn, x, 1e-3)
    exact = 2.0 * x + np.array([3.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(g, exact, rtol=0, atol=1e-6)


def test_fd_scan_returns_raw_grads_and_err():
    rng = np.random.default_rng(8)
    rows = rng.normal(size=(4, 6))
    fn = lambda x: float(np.sum(x * x * x) + np.sum(x))
    ad = 3.0 * rows ** 2 + 1.0
    steps = np.array([1e-1, 3e-2, 1e-3])
    err, grads = adc.fd_scan(fn, rows, ad, np.full(6, 1e-8), steps=steps)
    assert grads.shape == (3, 4, 6)
    assert err.shape == (3, 6)
    assert np.nanmax(err[-1]) < 1e-5          # smallest step: roundoff-limited
    assert np.nanmax(err[-1]) < np.nanmax(err[0])


def test_fd_stable_intervals_requires_adjacent_run():
    steps = np.array([1e-1, 3e-2, 1e-2, 3e-3])
    # single isolated minimum -> NOT stable
    err = np.array([[1.0, 1.0], [0.01, 1.0], [1.0, 1.0], [1.0, 1.0]])
    intervals, ok = adc.fd_stable_intervals(err, steps, level=0.05)
    assert ok is False
    assert intervals[0] == []
    # adjacent pair under the level -> stable
    err2 = np.array([[1.0, 1.0], [0.01, 0.02], [0.02, 0.01], [1.0, 1.0]])
    intervals2, ok2 = adc.fd_stable_intervals(err2, steps, level=0.05)
    assert ok2 is True
    assert intervals2[0] == [(0.03, 0.01)]


def test_fd_pair_agreement_checks_fd_vs_fd():
    steps = np.array([1e-1, 3e-2, 1e-2])
    ad = np.full((5, 2), 10.0)
    floors = np.full(2, 1.0)
    err = np.array([[0.01, 0.01], [0.01, 0.01], [1.0, 1.0]])
    fd = np.zeros((3, 5, 2))
    fd[0] = 10.0
    fd[1, :, 0] = 10.5      # 5% disagreement -> fails at level 0.05
    fd[1, :, 1] = 10.01     # fine, but the pair must ALSO both pass err
    pairs, ok = adc.fd_pair_agreement(fd, ad, floors, err, steps, level=0.05)
    assert ok is False
    assert pairs[0]["median_floored_rel"] == pytest.approx(0.05, rel=1e-6)


# ------------------------------------------------ provenance / hash gates

def test_verify_control_hashes_gate(tmp_path):
    run_dir = tmp_path / "runs/orx"
    (run_dir / "models").mkdir(parents=True)
    ckpt = run_dir / "models" / "model.eqx"
    ckpt.write_bytes(b"checkpoint-bytes")
    sha = adc.sha256_file(ckpt)
    manifest = {"control": {"checkpoints_sha256": {"models/model.eqx": sha}}}
    prov = adc.verify_control_hashes(manifest, run_dir)
    assert prov["models/model.eqx"]["sha256"] == sha
    manifest_bad = {"control": {"checkpoints_sha256": {"models/model.eqx": "0" * 64}}}
    with pytest.raises(RuntimeError, match="hash mismatch"):
        adc.verify_control_hashes(manifest_bad, run_dir)


def test_load_t0_manifest_rejects_wrong_schema(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"schema": "other"}')
    with pytest.raises(RuntimeError, match="schema"):
        adc.load_t0_manifest(p)


# ---------------------------------------------------------------- jax plumbing

def test_x64_context_toggles_and_restores():
    import jax
    prev = bool(jax.config.jax_enable_x64)
    with adc._x64(not prev):
        assert bool(jax.config.jax_enable_x64) == (not prev)
    assert bool(jax.config.jax_enable_x64) == prev


def test_eval_batched_concatenates_tree():
    import jax
    import jax.numpy as jnp
    fn = jax.jit(jax.vmap(lambda x: (x * x, x + 1.0)))
    arr = np.arange(150, dtype=np.float64).reshape(25, 6)
    with adc._x64(True):
        a, b = adc.eval_batched(fn, jnp.asarray(arr), batch=10)
    assert a.shape == (25, 6) and b.shape == (25, 6)
    np.testing.assert_allclose(a, arr * arr)
    np.testing.assert_allclose(b, arr + 1.0)


# ------------------------------------------------ tiny CNF of the control class

def _tiny_flow():
    import jax
    import jax.numpy as jnp
    from flow_ot_flow_matching_conditional import ConditionalPhaseSpaceFlow
    params = {"type": "MLP", "width": 16, "depth": 2}
    return ConditionalPhaseSpaceFlow(
        key=jax.random.key(0), data_mean=jnp.zeros(6), data_std=jnp.ones(6),
        spatial_vf_params=dict(params), conditional_vf_params=dict(params))


def test_tiny_cnf_grad_and_strict_ode_graft():
    import jax
    import jax.numpy as jnp
    flow = _tiny_flow()
    rng = np.random.default_rng(11)
    eta = rng.normal(size=(8, 6)) * 0.5
    with adc._x64(True):
        fn = adc._grad_lnf_fn(flow)
        lnf, grad = adc.eval_batched(fn, jnp.asarray(eta), batch=4)
    assert lnf.shape == (8,) and grad.shape == (8, 6)
    assert np.all(np.isfinite(grad))
    # strict graft: tolerances changed on BOTH subflows, original untouched
    strict = adc._with_ode_tolerance(flow, 1e-8, 1e-9)
    for f, expect in ((strict, (1e-8, 1e-9)), (flow, (1e-4, 1e-5))):
        for sub in (f.spatial_flow, f.conditional_velocity_flow):
            chain = sub.flow.bijection
            vf = chain[0] if hasattr(chain, "__getitem__") else chain.layers[0]
            ctrl = vf.stepsize_controller
            assert float(ctrl.rtol) == expect[0]
            assert float(ctrl.atol) == expect[1]


def test_tiny_cnf_split_identities_and_fd():
    import jax
    import jax.numpy as jnp
    flow = _tiny_flow()
    rng = np.random.default_rng(12)
    eta = rng.normal(size=(4, 6)) * 0.5
    strict = adc._with_ode_tolerance(flow, 1e-9, 1e-10)
    with adc._x64(True):
        fn = adc._grad_lnf_fn(strict)
        lnf, grad = adc.eval_batched(fn, jnp.asarray(eta), batch=4)
        vg_pos, vg_vel = adc._split_fns(strict)
        ln_nu, dnu = adc.eval_batched(vg_pos, jnp.asarray(eta[:, :3]))
        lnp, dlp = adc.eval_batched(vg_vel, jnp.asarray(eta))
    # lnF = ln n(q) + ln P(p|q) exactly (same code path, split evaluation)
    np.testing.assert_allclose(ln_nu + lnp, lnf, rtol=1e-12, atol=1e-12)
    # full gradient = position gradient + conditional gradient (Jacobian
    # of the condition normalization carried by autodiff)
    combined = adc.combine_split_grads(dnu, dlp)
    np.testing.assert_allclose(combined, grad, rtol=1e-10, atol=1e-10)
    # FD agrees with autodiff at adjacent steps on a small ladder
    with adc._x64(True):
        scalar = adc._lnf_fn_scalar(strict)
        steps = np.array([1e-1, 3e-2, 1e-2])
        err, fd = adc.fd_scan(scalar, eta, np.asarray(grad),
                              adc.component_floors(np.asarray(grad)), steps=steps)
    intervals, ok = adc.fd_stable_intervals(err, steps, level=5e-2)
    assert ok, f"FD not stable on the tiny CNF: err={err.tolist()}"


# ------------------------------------------------ frozen cache point designs

def test_point_order_hash_stable_and_sensitive():
    rng = np.random.default_rng(20)
    ids = np.arange(10, dtype=np.int64)
    eta = rng.normal(size=(10, 6))
    h1 = nsc.point_order_hash("heldout", ids, eta)
    assert h1 == nsc.point_order_hash("heldout", ids, eta)
    assert h1 != nsc.point_order_hash("velocity_probes", ids, eta)
    assert h1 != nsc.point_order_hash("heldout", ids[::-1].copy(), eta)       # reorder
    assert h1 != nsc.point_order_hash("heldout", ids, eta + 1e-12)            # values
    assert h1 != nsc.point_order_hash("heldout", ids, eta.astype(np.float32).astype(np.float64))


def test_velocity_probe_design_shapes_and_nesting():
    sigma = np.array([0.5, 0.6, 0.7])
    full = nsc.velocity_probes(sigma)
    assert full["eta"].shape == (7 * 16 * 96, 6)
    assert len(np.unique(full["point_id"])) == 7 * 16 * 96
    # shell radii = geometric mean of the frozen band edges, in kpc
    np.testing.assert_allclose(nsc.SHELL_R_KPC,
                               10.0 * np.sqrt(nsc.Q_BAND_EDGES[:-1] * nsc.Q_BAND_EDGES[1:]))
    # shared velocities: the same 96 vectors at every (shell, direction)
    vel = full["eta"][:, 3:].reshape(7, 16, 96, 3)
    for s in range(7):
        for d in range(16):
            np.testing.assert_array_equal(vel[s, d], vel[0, 0])
    # positions sit on the shell radius along the frozen directions
    q = full["eta"][:, :3].reshape(7, 16, 96, 3)
    np.testing.assert_allclose(np.linalg.norm(q[3, :, 0], axis=-1),
                               nsc.SHELL_R_KPC[3] / 10.0)
    # pilot subset = first shells/directions/velocities of the full design
    pilot = nsc.velocity_probes(sigma, shells=[3, 6], n_vel=8)
    assert pilot["eta"].shape == (2 * 16 * 8, 6)
    np.testing.assert_array_equal(pilot["eta"][:8, 3:], full["eta"][:8, 3:])
    np.testing.assert_array_equal(np.unique(pilot["eta"][:, 3:], axis=0),
                                  np.unique(full["eta"][:8, 3:], axis=0))
    # proposal log-density is the diag Gaussian with sigma
    lp = pilot["log_proposal"]
    manual = -0.5 * np.sum((pilot["eta"][:, 3:] / sigma) ** 2, axis=1) \
        - np.sum(np.log(sigma)) - 1.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(lp, manual)


def test_probe_directions_prefix_nested_into_grid():
    dirs_probe = nsc.sobol_directions(nsc.N_PROBE_DIRS, nsc.PROBE_SEED)
    grid = nsc.spatial_grid(radii=[10.0], n_dirs=32)
    dirs_grid = nsc.sobol_directions(32, nsc.PROBE_SEED)
    np.testing.assert_allclose(dirs_probe, dirs_grid[:nsc.N_PROBE_DIRS])
    assert grid["q"].shape == (32, 3)
    np.testing.assert_allclose(np.linalg.norm(grid["q"], axis=1), 1.0)


def test_spatial_grid_design():
    grid = nsc.spatial_grid()
    assert grid["q"].shape == (len(nsc.SPATIAL_RADII_KPC) * 2048, 3)
    np.testing.assert_allclose(nsc.SPATIAL_RADII_KPC,
                               np.concatenate([np.arange(2.0, 30.0, 3.0),
                                               np.arange(30.0, 70.1, 5.0)]))
    assert nsc.SPATIAL_RADII_KPC[0] == 2.0 and nsc.SPATIAL_RADII_KPC[-1] == 70.0
    assert len(nsc.SPATIAL_RADII_KPC) == 19


def test_heldout_split_rule_first_quarter(tmp_path):
    import h5py
    n = 1000
    rng = np.random.default_rng(13)
    p = tmp_path / "pop.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("eta", data=rng.normal(size=(n, 6)))
        f.create_dataset("particle_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("source_index", data=np.arange(n, dtype=np.int64))
        f.create_dataset("mass", data=np.ones(n))
        f.create_dataset("weights", data=np.ones(n))
    held = nsc.heldout_slice(str(p))
    assert len(held["eta"]) == 250                    # int(0.25 * 1000)
    np.testing.assert_array_equal(held["row_index"], np.arange(250))
    np.testing.assert_array_equal(held["particle_id"], np.arange(250))
    sub = nsc.heldout_slice(str(p), n_rows=50)
    assert len(sub["eta"]) == 50
    assert sub["row_index"].min() >= 0 and sub["row_index"].max() < 250
    r_sub = np.linalg.norm(sub["eta"][:, :3], axis=1)
    r_all = np.linalg.norm(held["eta"][:, :3], axis=1)
    assert r_sub.min() == pytest.approx(r_all.min())
    assert r_sub.max() == pytest.approx(r_all.max())


def test_population_velocity_std(tmp_path):
    import h5py
    rng = np.random.default_rng(14)
    vel = rng.normal(size=(2000, 3)) * np.array([1.0, 2.0, 3.0])
    eta = np.concatenate([rng.normal(size=(2000, 3)), vel], axis=1)
    p = tmp_path / "pop.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("eta", data=eta)
    np.testing.assert_allclose(nsc.population_velocity_std(str(p)),
                               np.std(vel, axis=0), rtol=1e-12)

def test_cache_points_only_smoke(tmp_path):
    """End-to-end points-only build against synthetic population + control
    layout: catches runtime name errors and contract breaks in build_cache
    without touching the real model (run ed679735 finding: a constant typo
    in the pilot branch only surfaced on the server)."""
    import h5py
    n = 400
    rng = np.random.default_rng(21)
    pop = tmp_path / "pop.h5"
    with h5py.File(pop, "w") as f:
        f.create_dataset("eta", data=rng.normal(size=(n, 6)))
        f.create_dataset("particle_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("source_index", data=np.arange(n, dtype=np.int64))
        f.create_dataset("mass", data=np.ones(n))
        f.create_dataset("weights", data=np.ones(n))
    control_repo = tmp_path / "control_repo"
    run_dir = control_repo / "runs" / "orx"
    (run_dir / "models").mkdir(parents=True)
    ckpt = run_dir / "models" / "model.eqx"
    ckpt.write_bytes(b"fake-checkpoint")
    (control_repo / "scripts").mkdir(parents=True)
    for rel in adc.MODEL_CODE_FILES:
        p = control_repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic control snapshot file")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "dpjax.nf-score-audit.t0-manifest.v1",
        "control": {"checkpoints_sha256": {"models/model.eqx": adc.sha256_file(ckpt)},
                    "source_commit": "synthetic", "training_run": "synthetic",
                    "single_flow_pair": True},
        "population": {"file": str(pop), "n": n},
    }))
    out = tmp_path / "out"
    nsc.build_cache(str(control_repo), str(run_dir), str(manifest),
                    str(pop), str(out), stage="pilot",
                    heldout_x64_rows=0, points_only=True)
    assert (out / "manifest.json").exists()
    m = json.loads((out / "manifest.json").read_text())
    assert m["schema"] == nsc.CACHE_SCHEMA
    assert set(m["point_order_sha256"]) == {"heldout", "velocity_probes", "spatial_grid"}
    assert (out / "points_heldout_pilot.h5").exists()
    assert (out / "points_velocity_probes_pilot.h5").exists()
    assert (out / "points_spatial_grid_pilot.h5").exists()
    with h5py.File(out / "points_velocity_probes_pilot.h5", "r") as f:
        assert f["eta"].shape == (2 * 16 * 8, 6)
        assert f.attrs["point_order_sha256"] == m["point_order_sha256"]["velocity_probes"]
