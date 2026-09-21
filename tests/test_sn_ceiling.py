#!/usr/bin/env python
"""Tests for the spectral-norm ceiling on the potential MLP.

Validates the exact-SVD spectral norms and the ceiling projection
W <- W * min(1, sigma0/sigma(W)) on an eqx MLP. CPU only.
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


def _mlp(seed=0):
    key = jax.random.key(seed)
    return eqx.nn.MLP(3, 1, 8, 2, activation=jax.nn.tanh, key=key)


def test_spectral_norms_exact():
    net = _mlp()
    got = pmod.spectral_norms(net)
    want = [float(jnp.linalg.svdvals(l.weight)[-1])
            for l in pmod._linear_layers(net)]
    np.testing.assert_allclose(got, want, rtol=1e-10)
    assert all(g > 0 for g in got)


def test_ceiling_rescales_overshoot():
    net = _mlp()
    sigma0 = pmod.spectral_norms(net)
    blown = eqx.tree_at(
        lambda m: [l.weight for l in pmod._linear_layers(m)], net,
        [l.weight * 10.0 for l in pmod._linear_layers(net)])
    clipped_net, ratios, clipped = pmod.apply_spectral_ceiling(blown, sigma0)
    assert all(clipped), ratios
    sigma_after = pmod.spectral_norms(clipped_net)
    np.testing.assert_allclose(sigma_after, sigma0, rtol=1e-4)


def test_ceiling_passes_undershoot():
    net = _mlp()
    sigma0 = pmod.spectral_norms(net)
    shrunk = eqx.tree_at(
        lambda m: [l.weight for l in pmod._linear_layers(m)], net,
        [l.weight * 0.1 for l in pmod._linear_layers(net)])
    kept_net, ratios, clipped = pmod.apply_spectral_ceiling(shrunk, sigma0)
    assert not any(clipped)
    np.testing.assert_allclose(ratios, [0.1] * len(ratios), rtol=1e-4)
    for l0, l1 in zip(pmod._linear_layers(shrunk), pmod._linear_layers(kept_net)):
        np.testing.assert_allclose(np.asarray(l0.weight), np.asarray(l1.weight))

