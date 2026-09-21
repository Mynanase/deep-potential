#!/usr/bin/env python
"""Tests for the osc-pair density-oscillation penalty in get_phi_loss.

Validates the LOSS ALGEBRA on analytic potentials (constant-Laplacian
quadratic and radially varying quartic) and the 7-tuple batch unpacking.
CPU only; run from the repo root:  python -m pytest tests/test_osc_penalty.py -q
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import potential as pmod  # noqa: E402


class QuadPhi(eqx.Module):
    """phi(q) = c |q|^2: gradient 2cq, Laplacian 6c (constant)."""

    c: float

    def __call__(self, q):
        return self.c * jnp.sum(q ** 2)


class QuarticPhi(eqx.Module):
    """phi(q) = c sum q_i^4: Laplacian 12c |q|^2 (radially varying)."""

    c: float

    def __call__(self, q):
        return self.c * jnp.sum(q ** 4)


class ZeroFrameshift(eqx.Module):
    def __call__(self, q, p):
        return jnp.zeros(3), jnp.zeros(3)


def _inputs(seed=0, n=64, ng=32):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)
    q = jax.random.uniform(keys[0], (n, 3), minval=-2.0, maxval=2.0)
    p = jax.random.uniform(keys[1], (n, 3), minval=-1.0, maxval=1.0)
    dlnf_dq = jax.random.normal(keys[2], (n, 3))
    dlnf_dp = jax.random.normal(keys[3], (n, 3))
    w = 0.5 + jax.random.uniform(keys[4], (n,))
    q_grid = jax.random.uniform(jax.random.key(7), (ng, 3), minval=-6.0, maxval=6.0)
    q_pair = q_grid + 0.4
    return q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair


def _cbe(phi_c, q, p, dlnf_dq, dlnf_dp, alpha=1.0):
    dphi_dq = 2.0 * phi_c * q
    null_hyp = jnp.sum(p * dlnf_dq - dphi_dq * dlnf_dp, axis=1)
    return jnp.arcsinh(alpha * jnp.abs(null_hyp)) / alpha


def test_osc_pair_quadratic_zero_penalty():
    # Constant Laplacian -> the pair difference vanishes identically.
    q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair = _inputs()
    c, alpha, beta, lambda_, eta = -0.3, 1.0, 1.0, 1.0, 5.0
    loss, _ = pmod.get_phi_loss(
        QuadPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=alpha, beta=beta, lambda_=lambda_, l2_potential=0.0,
        weights=w, q_grid=q_grid, q_pair=q_pair, osc_weight=eta,
    )
    cbe = _cbe(c, q, p, dlnf_dq, dlnf_dp, alpha)
    lap = 6.0 * c
    pen = jnp.arcsinh(beta * jnp.maximum(-lap, 0.0)) / beta
    expected = jnp.log(jnp.sum(w * cbe) / jnp.sum(w)) + lambda_ * pen
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)


def test_osc_pair_quartic_analytic():
    q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair = _inputs()
    c, alpha, beta, lambda_, eta = 0.2, 1.0, 1.0, 1.0, 3.0
    loss, _ = pmod.get_phi_loss(
        QuarticPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=alpha, beta=beta, lambda_=lambda_, l2_potential=0.0,
        weights=w, q_grid=q_grid, q_pair=q_pair, osc_weight=eta,
    )
    cbe = _cbe(c, q, p, dlnf_dq, dlnf_dp, alpha)
    lap = 6.0 * c  # positive everywhere -> prior_neg inactive
    pen = jnp.arcsinh(beta * jnp.maximum(-lap, 0.0)) / beta
    lap_a = 12.0 * c * jnp.sum(q_grid ** 2, axis=1)
    lap_b = 12.0 * c * jnp.sum(q_pair ** 2, axis=1)
    osc = jnp.mean((lap_b - lap_a) ** 2)
    expected = (jnp.log(jnp.sum(w * cbe) / jnp.sum(w)) + lambda_ * pen
                + eta * osc)
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)


def test_osc_pair_requires_grid():
    # q_pair without q_grid must not crash the legacy path: it is ignored.
    q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair = _inputs()
    loss_no_pair, _ = pmod.get_phi_loss(
        QuadPhi(0.5), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        l2_potential=0.0, weights=w, q_grid=q_grid, osc_weight=2.0,
    )
    assert np.isfinite(float(loss_no_pair))


def test_unpack_seven_tuple():
    q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair = _inputs()
    got = pmod._unpack_phi_batch((q, p, dlnf_dq, dlnf_dp, w, q_grid, q_pair))
    np.testing.assert_array_equal(got[-1], q_pair)
    got6 = pmod._unpack_phi_batch((q, p, dlnf_dq, dlnf_dp, w, q_grid))
    assert got6[-1] is None
