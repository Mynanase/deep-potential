#!/usr/bin/env python
"""Tests for the step-6 DF-constraint audit machinery.

These validate the AUDITOR (CBE residual algebra, SVD local-force
diagnostics, finite-difference scan, stratified row selection, floored
relative errors, x64 toggling, batching) on analytic examples.  They do not
touch the trained checkpoints; model-dependent plumbing is exercised by the
declared 256-point smoke run instead.

Run from the repo root:  python -m pytest tests/test_df_constraint_audit.py -q
CPU only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import audit_df_constraints as adc  # noqa: E402


# ------------------------------------------------------- analytic steady DF
# lnF(q, p) = -|p|^2 / 2 - phi_a(q),  phi_a(q) = 0.5 * ln(1 + q.q)
# d_q lnF = -grad phi_a,  d_p lnF = -p  ->  with grad phi = grad phi_a the
# steady-state CBE holds exactly in code units.

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
    assert out[1] == 0.1          # floored, not 1e-4/0 = inf
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


# ---------------------------------------------------------------- CBE residual

def test_cbe_residual_zero_for_steady_df():
    q, p = _make_points(64)
    dlnf_dq = -_grad_phi_a(q)
    dlnf_dp = -p
    term1, term2, r, rel = adc.cbe_residual(p, dlnf_dq, dlnf_dp,
                                            _grad_phi_a(q))
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    np.testing.assert_allclose(rel, 0.0, atol=1e-12)
    np.testing.assert_allclose(term1, term2, atol=1e-12)


def test_cbe_residual_detects_wrong_potential():
    q, p = _make_points(64)
    dlnf_dq = -_grad_phi_a(q)
    dlnf_dp = -p
    _t1, _t2, r, rel = adc.cbe_residual(p, dlnf_dq, dlnf_dp,
                                        2.0 * _grad_phi_a(q))
    # R = p.grad phi_a != 0 for generic p
    assert np.median(np.abs(r)) > 1e-2
    assert np.median(rel) > 1e-3


def test_cbe_residual_epsilon_kills_both_terms_zero():
    # p = 0 and dphi = 0: R = 0 and relR = 0 despite vanishing denominators
    q, _ = _make_points(4)
    p = np.zeros_like(q)
    _t1, _t2, r, rel = adc.cbe_residual(p, np.ones_like(q), np.ones_like(q),
                                        np.zeros_like(q))
    np.testing.assert_allclose(r, 0.0)
    np.testing.assert_allclose(rel, 0.0)


# ------------------------------------------------------- SVD local force

def test_local_force_svd_recovers_exact_force():
    rng = np.random.default_rng(2)
    q = rng.normal(size=3) * 0.5
    g_true = _grad_phi_a(q[None, :])[0]
    m = 64
    p = rng.normal(size=(m, 3))
    A = -p                       # d_p lnF
    b = p @ (-g_true)            # p . d_q lnF
    d = adc.local_force_svd(A, b)
    assert d["rank"] == 3
    assert d["resid_rel"] < 1e-10
    np.testing.assert_allclose(d["g_hat"], g_true, rtol=1e-10)


def test_local_force_svd_rank_deficient_suppresses_g():
    rng = np.random.default_rng(3)
    m = 32
    p = rng.normal(size=(m, 3))
    p[:, 2] = 0.0                # velocities confined to a plane
    g_true = np.array([0.3, -0.2, 0.7])
    d = adc.local_force_svd(-p, p @ (-g_true))
    assert d["rank"] == 2
    assert d["g_hat"] is None    # no stable force estimate emitted


def test_local_force_svd_flags_inconsistent_constraints():
    rng = np.random.default_rng(4)
    p = rng.normal(size=(128, 3))
    b = np.ones(128)             # orthogonal to anything A g can produce
    d = adc.local_force_svd(-p, b)
    assert d["rank"] == 3
    assert d["resid_rel"] > 0.9  # essentially all of b unexplained


# ---------------------------------------------------------------- FD machinery

def test_central_fd_grad_quadratic_exact():
    fn = lambda x: float(np.sum(x * x) + 3.0 * x[0])
    x = np.array([0.3, -1.2, 0.7, 0.0, 2.0, -0.5])
    g = adc.central_fd_grad(fn, x, 1e-3)
    exact = 2.0 * x + np.array([3.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(g, exact, rtol=0, atol=1e-6)


def test_fd_scan_finds_plateau_on_smooth_function():
    rng = np.random.default_rng(5)
    rows = rng.normal(size=(6, 6))
    fn = lambda x: float(np.sum(x * x * x) + np.sum(x))
    ad = 3.0 * rows ** 2 + 1.0
    floors = np.full(6, 1e-8)
    err, best_h = adc.fd_scan(fn, rows, ad, floors)
    plateau = np.nanmin(err, axis=0)
    assert np.all(plateau < 1e-6)
    assert best_h in adc.FD_STEPS
    # error decreases then hits the roundoff floor (no monotone blow-up)
    assert np.nanmax(err[0]) > np.nanmax(err[3])


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


def test_group_summary_keys():
    out = adc.group_summary({"spatial": np.array([1.0, 2.0]),
                             "velocity": np.array([3.0, 4.0, 5.0])})
    assert out["spatial_median"] == 1.5
    assert out["velocity_p99"] == pytest.approx(4.98)
    assert "velocity_max" in out
