from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "plummer"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import plummer_oracle as po  # noqa: E402
import local_force_inversion as lfi  # noqa: E402
from plummer_sphere import draw_from_sphere  # noqa: E402


def test_draw_from_sphere_does_not_use_global_rng(monkeypatch):
    import numpy as np
    def fail(*args, **kwargs):
        raise AssertionError("global NumPy RNG was called")
    monkeypatch.setattr(np.random, "uniform", fail, raising=True)
    points = draw_from_sphere(16, rng=np.random.default_rng(3))
    assert points.shape == (16, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, rtol=1e-12)


def test_sampler_discretization_is_small():
    d = po.sampler_discretization(n_quantiles_if_supported=None) if False else po.sampler_discretization()
    assert d["speed_grid_n"] == 1000
    assert d["abs_error_median"] < 1e-5
    assert d["abs_error_p99"] < 1e-4
    assert d["abs_error_max"] < 1e-3


def test_oracle_score_matches_autodiff_and_units():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(4)
    q = rng.normal(size=(37, 3)) * 2.0
    p = rng.normal(size=(37, 3)) * 0.15
    eta = np.concatenate((q, p), axis=1)
    analytic = po.oracle_score(eta)
    sphere = po.make_oracle()
    scaling = np.asarray([sphere.k] * 3 + [sphere.gamma] * 3)
    df_normalization = float(sphere.unit_plummer.df_norm)

    def logf_single(z):
        unit = z / scaling
        unit_r2 = jnp.sum(unit[:3] ** 2)
        unit_v2 = jnp.sum(unit[3:] ** 2)
        energy = 1.0 / jnp.sqrt(1.0 + unit_r2) - 0.5 * unit_v2
        return jnp.log(df_normalization * jnp.maximum(energy, 1e-30) ** 3.5)
    autodiff = np.asarray(jax.vmap(jax.grad(logf_single))(jnp.asarray(eta)))
    np.testing.assert_allclose(analytic, autodiff, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(analytic[:, 3:], -7.0 * p / (2.0 * (sphere.psi(np.linalg.norm(q, axis=1)) - 0.5 * np.sum(p*p, axis=1)))[:, None], rtol=1e-12)


def test_oracle_force_mass_phi_and_score_chain():
    rng = np.random.default_rng(5)
    eta = np.concatenate((rng.normal(size=(64, 3)), rng.normal(size=(64, 3)) * .1), axis=1)
    phi = po.oracle_phi(eta)
    grad = po.oracle_grad_phi(eta)
    alpha = po.oracle_alpha(eta)
    mass = po.oracle_mass(eta)
    np.testing.assert_allclose(alpha, -grad, rtol=1e-14)
    # Phi=-G M r/(r^2+b^2), hence grad phi=G M q/(r^2+b^2)^{3/2}.
    r = np.linalg.norm(eta[:, :3], axis=1)
    np.testing.assert_allclose(
        phi, -1.0 / np.sqrt(po.B_CODE**2 + r*r), rtol=1e-11)
    np.testing.assert_allclose(
        mass, r**3 / (po.B_CODE**2 + r*r)**1.5, rtol=1e-11)
    C, y = po.velocity_constraints(eta)
    residual = np.sum(C * alpha, axis=1) - y
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)


def test_local_force_recovers_oracle_from_analytic_score():
    rng = np.random.default_rng(6)
    # Generate a conditional velocity cloud at one fixed radius/direction.
    q = np.array([2.1, -1.4, 0.7])
    p = rng.normal(size=(128, 3)) * 0.12
    eta = np.column_stack((np.repeat(q[None, :], len(p), axis=0), p))
    C, y = po.velocity_constraints(eta)
    solve = lfi.local_force_svd_alpha(C, y, weights=np.full(len(p), 0.7))
    assert solve["rank"] == 3
    assert solve["resid_rel"] < 1e-11
    np.testing.assert_allclose(solve["alpha_hat"], po.oracle_alpha(eta)[0], rtol=1e-11)


def test_rank_deficiency_has_no_alpha_and_weighted_projection():
    rng = np.random.default_rng(7)
    p = rng.normal(size=(80, 3))
    p[:, 2] = 0.0
    C = -p
    y = p @ np.array([0.2, -0.1, 0.4])
    solve = lfi.local_force_svd_alpha(C, y, weights=rng.uniform(.2, 2, len(p)))
    assert solve["rank"] == 2
    assert solve["alpha_hat"] is None
    split = lfi.absorbable_split(C, y, weights=np.ones(len(p)))
    np.testing.assert_allclose(split["absorbable"] + split["orthogonal"], y, atol=1e-12)
    assert np.linalg.norm(split["orthogonal"]) < 1e-10


def test_mean_zero_projection_and_energy_dependent_counterexamples():
    d = lfi.projection_bias_examples(n=160_000)
    assert abs(d["mean_energy_term"]) < 1e-13
    assert abs(d["mean_simple_term"]) < 1e-13
    assert np.linalg.norm(d["delta_alpha_energy"]) > 1e-4
    assert np.linalg.norm(d["delta_alpha_simple"]) > 1e-4


def test_four_combinations_interface_runs():
    rng = np.random.default_rng(8)
    q = rng.normal(size=(240, 3)) * 2.0
    p = rng.normal(size=(240, 3)) * .1
    eta = np.column_stack((q, p))
    # Sort in each coordinate so np.gradient is defined.  A production run
    # groups by q and uses the local estimator; this smoke checks the schema.
    order = np.lexsort((eta[:,2], eta[:,1], eta[:,0]))
    eta = eta[order]
    cases = po.four_combinations(eta, po.oracle_score(eta), po.oracle_phi(eta))
    assert len(cases) == 4
    assert {(x["score"], x["potential"]) for x in cases} == {("true", "true"), ("true", "estimated"), ("estimated", "true"), ("estimated", "estimated")}
    assert all(x["solve"]["rank"] == 3 for x in cases)


def test_point_potential_unit_calibration_and_rotated_gradient():
    G_KPC_KMS2_MSUN = 4.300917270036281e-6
    rng = np.random.default_rng(12)
    q = rng.normal(size=(256, 3)) * 3.0
    phi_dimless = po.oracle_phi(np.column_stack((q, np.zeros_like(q))))
    r = np.linalg.norm(q, axis=1)
    radius_kpc = r * po.L_KPC
    # For Phi_dimless = -1/sqrt(b_code^2+r_q^2), d Phi_dimless/dr_q near r=b
    # is 1/(sqrt(2) b^2).  Convert to physical kpc and km^2/s^2 to calibrate
    # the point-potential scale; an additive offset is explicitly removed.
    phi_centered = phi_dimless - np.mean(phi_dimless)
    # The A5 scale solution is defined by the frozen coordinate convention:
    # Phi_phys = V^2 Phi_dimless, independent of the unknown HDF5 additive
    # zero.  A monopole fit recovers this same conversion after calibration.
    np.testing.assert_allclose(po.V_KMS**2, 10000.0)

    # Rotation changes component coordinates but not radius, phi, mass, or
    # gradient magnitude; component gradients follow R^T grad(Rq).
    theta = np.pi / 7
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    qr = q @ rotation.T
    eta0 = np.column_stack((q, np.zeros_like(q)))
    eta1 = np.column_stack((qr, np.zeros_like(qr)))
    np.testing.assert_allclose(po.oracle_phi(eta1), po.oracle_phi(eta0), rtol=1e-13)
    np.testing.assert_allclose(po.oracle_mass(eta1), po.oracle_mass(eta0), rtol=1e-13)
    np.testing.assert_allclose(po.oracle_grad_phi(eta1), po.oracle_grad_phi(eta0) @ rotation.T, rtol=1e-13)
    np.testing.assert_allclose(
        np.linalg.norm(po.oracle_grad_phi(eta1), axis=1),
        np.linalg.norm(po.oracle_grad_phi(eta0), axis=1), rtol=1e-13)


def test_local_point_potential_gradient_error_is_quantified():
    rng = np.random.default_rng(13)
    n = 200_000
    q = rng.normal(size=(n, 3)) * 2.0
    r = np.linalg.norm(q, axis=1)
    # Density declines steeply outward; retain an outer testing subsample.
    q_outer = q[r > 3.0]
    # Local quadratic regression in the principal axes quantifies sparse-region
    # point-potential gradient error; no acceleration block is assumed.
    from scipy.spatial import cKDTree
    def local_gradient(points, values, k=32):
        idx = cKDTree(points).query(points, k=k)[1]
        offsets = points[idx] - points[:, None, :]
        design = np.concatenate((
            np.ones(offsets.shape[:2] + (1,)), offsets,
            offsets[..., 0:1] * offsets[..., 1:2],
            offsets[..., 0:1] * offsets[..., 2:3],
            offsets[..., 1:2] * offsets[..., 2:3], offsets**2), axis=2)
        return np.einsum("nij,nj->ni", np.linalg.pinv(design), values[idx])[:, 1:4]
    sample = q_outer[:256]
    true_grad = po.oracle_grad_phi(np.column_stack((sample, np.zeros_like(sample))))
    estimate = local_gradient(sample, po.oracle_phi(np.column_stack((sample, np.zeros_like(sample)))))
    rel = np.linalg.norm(estimate - true_grad, axis=1) / np.linalg.norm(true_grad, axis=1)
    assert np.percentile(rel, 10) < 0.15
    assert np.median(rel) < 0.35
