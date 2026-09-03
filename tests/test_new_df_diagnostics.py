from __future__ import annotations

import numpy as np

from experiments.diagnostics.evaluation import (
    radial_score_field_diagnostics,
    radial_speed_density_diagnostics,
)


def test_radial_score_projection_quantiles_support_mask_and_zero_vectors():
    rng = np.random.default_rng(8)
    eta = rng.normal(size=(2_000, 6))
    eta[0, :3] = 0.0
    eta[1, 3:] = 0.0
    radius = np.linalg.norm(eta[:, :3], axis=1)
    speed = np.linalg.norm(eta[:, 3:], axis=1)
    rhat = np.divide(
        eta[:, :3],
        radius[:, None],
        out=np.zeros((eta.shape[0], 3)),
        where=radius[:, None] > 0,
    )
    vhat = np.divide(
        eta[:, 3:],
        speed[:, None],
        out=np.zeros((eta.shape[0], 3)),
        where=speed[:, None] > 0,
    )
    scores = np.concatenate([2.0 * rhat, -3.0 * vhat], axis=1)[None, ...]
    weights = rng.uniform(0.1, 2.0, eta.shape[0])

    diagnostics = radial_score_field_diagnostics(
        eta,
        scores,
        weights,
        n_radius_bins=5,
        n_speed_bins=5,
        min_effective_count=5,
    )
    supported = diagnostics["score_field_effective_count"] >= 5
    np.testing.assert_allclose(
        diagnostics["score_field_r_median"][0][supported], 2.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        diagnostics["score_field_v_median"][0][supported], -3.0, atol=1.0e-12
    )
    assert diagnostics["score_slice_r_q16"].shape == (1, 3, 5)
    assert diagnostics["score_slice_r_q84"].shape == (1, 3, 5)

    masked = radial_score_field_diagnostics(
        eta,
        scores,
        weights,
        n_radius_bins=4,
        n_speed_bins=4,
        min_effective_count=1.0e9,
    )
    assert np.isnan(masked["score_field_r_median"]).all()

    zero_eta = np.zeros((16, 6))
    zero_field = radial_score_field_diagnostics(
        zero_eta,
        np.zeros((1, 16, 6)),
        np.ones(16),
        n_radius_bins=2,
        n_speed_bins=2,
        min_effective_count=1,
    )
    assert np.all(np.diff(zero_field["score_field_r_edges"]) > 0)
    assert np.all(np.diff(zero_field["score_field_v_edges"]) > 0)
    assert np.isnan(zero_field["score_field_r_median"]).all()


def test_weighted_resampling_is_consistent_on_shared_radial_speed_grid():
    rng = np.random.default_rng(91)
    reference = rng.normal(size=(5_000, 6))
    weights = np.exp(0.35 * reference[:, 0])
    indices = rng.choice(reference.shape[0], size=30_000, p=weights / weights.sum())
    model = reference[indices][None, ...]
    diagnostics = radial_speed_density_diagnostics(
        reference,
        model,
        weights,
        n_radius_bins=12,
        n_speed_bins=12,
    )
    ratio = diagnostics["radial_speed_log10_ratio"][0]
    finite = ratio[np.isfinite(ratio)]
    assert np.median(np.abs(finite)) < 0.12
