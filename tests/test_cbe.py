from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dpjax.physics.cbe import (
    loss_cbe_mse,
    loss_cbe_robust,
    loss_negative_density,
)


def test_loss_cbe_mse_adds_independent_negative_density_penalty():
    residual = jnp.array([1.0, -2.0])
    laplacian = jnp.array([-3.0, 4.0])

    plain = loss_cbe_mse(residual)
    constrained = loss_cbe_mse(
        residual,
        laplacian,
        beta=2.0,
        lambda_mass=0.5,
    )

    expected_mass = np.mean(np.arcsinh([6.0, 0.0]))
    np.testing.assert_allclose(plain, 2.5)
    np.testing.assert_allclose(constrained, 2.5 + 0.5 * expected_mass)


def test_loss_cbe_mse_requires_laplacian_when_mass_penalty_enabled():
    with pytest.raises(ValueError, match="laplacian_phi_phys is required"):
        loss_cbe_mse(
            jnp.array([1.0]),
            lambda_mass=1.0,
        )


def test_negative_density_penalty_respects_weights_and_gradient_direction():
    laplacian = jnp.array([-2.0, 1.0])
    weights = jnp.array([0.25, 1.75])

    penalty = loss_negative_density(
        laplacian,
        beta=3.0,
        weights=weights,
    )
    expected = np.mean([0.25 * np.arcsinh(6.0), 0.0])
    np.testing.assert_allclose(penalty, expected)

    gradient = jax.grad(
        lambda values: loss_negative_density(values, beta=3.0)
    )(laplacian)
    assert float(gradient[0]) < 0.0
    assert float(gradient[1]) == 0.0


def test_robust_loss_keeps_previous_combined_formula():
    residual = jnp.array([1.0, -2.0])
    laplacian = jnp.array([-3.0, 4.0])
    weights = jnp.array([0.5, 1.5])
    alpha = 0.7
    beta = 2.0
    lambda_mass = 0.25

    actual = loss_cbe_robust(
        residual,
        laplacian,
        alpha=alpha,
        beta=beta,
        lambda_mass=lambda_mass,
        weights=weights,
    )
    expected_per_sample = (
        np.arcsinh(alpha * np.abs(np.asarray(residual)))
        + lambda_mass
        * np.arcsinh(beta * np.maximum(-np.asarray(laplacian), 0.0))
    )
    expected = np.mean(np.asarray(weights) * expected_per_sample)

    np.testing.assert_allclose(actual, expected)
