#!/usr/bin/env python
"""Tests for the grid-decoupled negative-density prior in get_phi_loss.

These validate the LOSS ALGEBRA on analytic potentials (quadratic and
quartic), the uniform-ball grid sampler, and the 4/5/6-tuple batch
unpacking. They do not touch trained checkpoints.

Run from the repo root:  python -m pytest tests/test_grid_prior_loss.py -q
CPU only.
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


def _random_inputs(seed=0, n=64, ng=32):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 5)
    q = jax.random.uniform(keys[0], (n, 3), minval=-2.0, maxval=2.0)
    p = jax.random.uniform(keys[1], (n, 3), minval=-1.0, maxval=1.0)
    dlnf_dq = jax.random.normal(keys[2], (n, 3))
    dlnf_dp = jax.random.normal(keys[3], (n, 3))
    w = 0.5 + jax.random.uniform(keys[4], (n,))
    q_grid = jax.random.uniform(jax.random.key(7), (ng, 3), minval=-6.0, maxval=6.0)
    return q, p, dlnf_dq, dlnf_dp, w, q_grid


def _cbe(phi_c, q, p, dlnf_dq, dlnf_dp, alpha=1.0):
    dphi_dq = 2.0 * phi_c * q
    null_hyp = jnp.sum(p * dlnf_dq - dphi_dq * dlnf_dp, axis=1)
    return jnp.arcsinh(alpha * jnp.abs(null_hyp)) / alpha


def test_decoupled_loss_formula():
    q, p, dlnf_dq, dlnf_dp, w, q_grid = _random_inputs()
    c, alpha, beta, lambda_ = -0.3, 1.0, 1.0, 1.0
    loss, loss_noreg = pmod.get_phi_loss(
        QuadPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=alpha, beta=beta, lambda_=lambda_,
        l2_potential=0.0, weights=w, q_grid=q_grid,
    )
    cbe = _cbe(c, q, p, dlnf_dq, dlnf_dp, alpha)
    lap = 6.0 * c  # negative everywhere -> penalty active and constant
    pen = jnp.arcsinh(beta * jnp.maximum(-lap, 0.0)) / beta
    expected = jnp.log(jnp.sum(w * cbe) / jnp.sum(w)) + lambda_ * pen
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)
    np.testing.assert_allclose(float(loss_noreg), float(expected), rtol=1e-5)


def test_legacy_coupled_loss_formula():
    q, p, dlnf_dq, dlnf_dp, w, q_grid = _random_inputs()
    c, alpha, beta, lambda_ = -0.3, 1.0, 1.0, 1.0
    loss, _ = pmod.get_phi_loss(
        QuadPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=alpha, beta=beta, lambda_=lambda_,
        l2_potential=0.0, weights=w, q_grid=None,
    )
    cbe = _cbe(c, q, p, dlnf_dq, dlnf_dp, alpha)
    lap = 6.0 * c
    pen = jnp.arcsinh(beta * jnp.maximum(-lap, 0.0)) / beta
    expected = jnp.log(jnp.sum(w * (cbe + lambda_ * pen)) / jnp.sum(w))
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)
    # Placement matters: inside vs. outside the log gives different values.
    decoupled, _ = pmod.get_phi_loss(
        QuadPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=alpha, beta=beta, lambda_=lambda_,
        l2_potential=0.0, weights=w, q_grid=q_grid,
    )
    assert abs(float(decoupled) - float(loss)) > 1e-3


def test_penalty_evaluated_on_grid_not_samples():
    q, p, dlnf_dq, dlnf_dp, w, _ = _random_inputs()
    c, beta = -1.0, 1.0
    # Laplacian = 12c|q|^2 varies with radius, so WHERE the penalty is
    # evaluated changes its value.
    near = 0.1 * jnp.ones((16, 3)) / np.sqrt(3)
    far = 2.0 * jnp.ones((16, 3)) / np.sqrt(3)
    kw = dict(alpha=1.0, beta=beta, lambda_=1.0, l2_potential=0.0, weights=w)
    loss_near, _ = pmod.get_phi_loss(
        QuarticPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        q_grid=near, **kw)
    loss_far, _ = pmod.get_phi_loss(
        QuarticPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        q_grid=far, **kw)
    # Quartic gradient is 4c q^3.
    dphi = 4.0 * c * q ** 3
    null = jnp.sum(p * dlnf_dq - dphi * dlnf_dp, axis=1)
    cbe = jnp.arcsinh(jnp.abs(null))
    r_near2 = jnp.sum(near ** 2, axis=1)
    r_far2 = jnp.sum(far ** 2, axis=1)
    pen_near = jnp.arcsinh(jnp.maximum(12.0 * (-c) * r_near2, 0.0)).mean()
    pen_far = jnp.arcsinh(jnp.maximum(12.0 * (-c) * r_far2, 0.0)).mean()
    exp_near = jnp.log(jnp.sum(w * cbe) / jnp.sum(w)) + pen_near
    exp_far = jnp.log(jnp.sum(w * cbe) / jnp.sum(w)) + pen_far
    np.testing.assert_allclose(float(loss_near), float(exp_near), rtol=1e-5)
    np.testing.assert_allclose(float(loss_far), float(exp_far), rtol=1e-5)
    assert float(loss_far) > float(loss_near)


def test_unweighted_decoupled_loss():
    q, p, dlnf_dq, dlnf_dp, _, q_grid = _random_inputs()
    c = -0.2
    loss, _ = pmod.get_phi_loss(
        QuadPhi(c), ZeroFrameshift(), None, q, p, dlnf_dq, dlnf_dp,
        alpha=1.0, beta=1.0, lambda_=1.0, l2_potential=0.0,
        weights=None, q_grid=q_grid,
    )
    cbe = _cbe(c, q, p, dlnf_dq, dlnf_dp)
    pen = jnp.arcsinh(jnp.maximum(-6.0 * c, 0.0))
    expected = jnp.log(jnp.mean(cbe)) + jnp.mean(pen)
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)


def test_sample_uniform_ball_volume_weighting():
    n, radius = 200_000, 7.0
    pts = np.asarray(pmod.sample_uniform_ball(jax.random.key(3), n, radius))
    r = np.linalg.norm(pts, axis=1)
    assert r.max() <= radius + 1e-4
    # Volume CDF: E[r] = 3R/4, P(r < R/2) = 1/8.
    np.testing.assert_allclose(r.mean(), 0.75 * radius, atol=0.05)
    np.testing.assert_allclose((r < radius / 2).mean(), 0.125, atol=0.005)
    # Isotropy: mean direction ~ 0.
    assert np.abs(pts.mean(axis=0)).max() < 0.05


def test_sample_radius_balanced_ball_radius_weighting():
    n, radius = 200_000, 7.0
    pts = np.asarray(pmod.sample_radius_balanced_ball(jax.random.key(4), n, radius))
    r = np.linalg.norm(pts, axis=1)
    assert r.max() <= radius + 1e-4
    # Linear CDF: E[r] = R/2, P(r < R/2) = 1/2 -- equal points per kpc.
    np.testing.assert_allclose(r.mean(), 0.5 * radius, atol=0.05)
    np.testing.assert_allclose((r < radius / 2).mean(), 0.5, atol=0.005)
    # Inner share: P(r < 3) = 3/7 ~ 42.9% (volume sampler: (3/7)^3 ~ 7.9%).
    np.testing.assert_allclose((r < 3.0).mean(), 3.0 / 7.0, atol=0.005)
    # Isotropy: mean direction ~ 0.
    assert np.abs(pts.mean(axis=0)).max() < 0.05


def test_unpack_phi_batch_tuples():
    q, p, dq, dp = "q", "p", "dq", "dp"
    assert pmod._unpack_phi_batch((q, p, dq, dp)) == (q, p, dq, dp, None, None)
    assert pmod._unpack_phi_batch((q, p, dq, dp, "w")) == (q, p, dq, dp, "w", None)
    assert pmod._unpack_phi_batch((q, p, dq, dp, "w", "g")) == (q, p, dq, dp, "w", "g")


class ActPhi(eqx.Module):
    """phi(q) = c |q|^2 with a jitted callable attribute, mirroring the
    real network whose net.activation leaf breaks plain jax.jit calls."""

    c: float
    act: object

    def __call__(self, q):
        return self.c * jnp.sum(self.act(jnp.sum(q ** 2, keepdims=True)))


def test_vmapped_laplacian_call_pattern():
    """The run-script evidence block evaluates the Laplacian through a
    closure around the model + bare vmap. A module holding a jitted
    callable attribute cannot be passed as a plain jax.jit argument, so
    this locks in the pattern that works on the real checkpoints."""
    c = -0.4
    q = jax.random.uniform(jax.random.key(11), (50, 3), minval=-2.0, maxval=2.0)
    model = ActPhi(c=c, act=jax.jit(lambda x: x))

    def lap_batch(qb):
        def lap_one(x):
            return pmod.calc_phi_laplacian(model, x)
        return jax.vmap(lap_one)(qb)

    lap = np.asarray(lap_batch(q))
    np.testing.assert_allclose(lap, 6.0 * c, rtol=1e-5)
