#!/usr/bin/env python
"""Tests for the radial conditional-velocity review machinery.

These validate the ESTIMATOR (mass weighting, Wasserstein-1, centred sigma,
weighted correlation, spherical basis degeneracy, the paired position bootstrap,
the truth half-split reference scale, the angular block bootstrap and the
reading/formatting helpers) on analytic and synthetic examples.  They never
touch a trained checkpoint and never need a GPU, which is what makes the review
auditable independently of the run that produced its inputs.

Run from the repo root:  python -m pytest tests/test_radial_cv_metrics.py -q
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import df_phase1_radial_cv_metrics as rcv  # noqa: E402


# ---------------------------------------------------------------- weighting
def test_weighted_moments_match_hand_computation():
    a = np.array([0.0, 1.0, 4.0])
    w = np.array([1.0, 2.0, 1.0])
    assert abs(rcv.wmean(a, w) - 1.5) < 1e-12
    # variance = (1*(0-1.5)^2 + 2*(1-1.5)^2 + 1*(4-1.5)^2) / 4 = 2.25
    assert abs(rcv.wstd(a, w) - 1.5) < 1e-12
    # N_eff of equal weights is the count, and of one heavy weight is ~1
    assert abs(rcv.n_eff(np.ones(7)) - 7.0) < 1e-12
    assert rcv.n_eff(np.array([1.0, 0.0, 0.0])) < 1.0 + 1e-12


def test_sigma_is_centred_and_not_an_rms_speed():
    a = np.array([10.0, 12.0])
    w = np.ones(2)
    assert abs(rcv.wstd(a, w) - 1.0) < 1e-12
    assert not abs(np.sqrt(np.mean(a ** 2)) - rcv.wstd(a, w)) < 1e-6


# --------------------------------------------------------------- distances
def test_w1_is_exact_for_translation_and_two_points():
    a = np.array([0.0, 1.0])
    assert abs(rcv.w1(a, np.ones(2), a + 3.0, np.ones(2)) - 3.0) < 1e-12
    # two unit masses shifted against each other: W1 = the gap between them
    assert abs(rcv.w1(np.array([0.0]), np.ones(1), np.array([0.0, 2.0]),
                      np.array([0.5, 0.5])) - 1.0) < 1e-12


def test_w1_is_called_on_the_uncut_samples():
    """The display window must never enter the distance.

    Two samples whose mass differs outside the window: the full-sample W1 sees
    that mass, a windowed one does not.  The review's numbers are the full ones.
    """
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.normal(0.0, 1.0, 3000), rng.normal(0.0, 1.0, 400)])
    b = np.concatenate([rng.normal(0.3, 1.3, 3000), rng.normal(-12.0, 1.0, 400)])
    w = np.ones(len(a))
    keep = (a > -4.0) & (a < 4.0) & (b > -4.0) & (b < 4.0)
    assert not keep.all()
    assert rcv.w1(a, w, b, w) > rcv.w1(a[keep], w[keep], b[keep], w[keep]) + 0.5


# ----------------------------------------------------------- correlations
def test_weighted_correlation_endpoints_and_degeneracy():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 3.0, 3.0, 1.0])
    assert abs(rcv.wcorr(x, 2 * x + 1, w) - 1.0) < 1e-12
    assert abs(rcv.wcorr(x, -x, w) + 1.0) < 1e-12
    assert np.isnan(rcv.wcorr(np.array([1.0, 1.0]), x[:2], np.ones(2)))
    # weights that zero out the spread make the correlation undefined too
    assert np.isnan(rcv.wcorr(x, -x, np.array([1.0, 0.0, 0.0, 0.0])))


# ------------------------------------------------------------ velocity basis
def test_spherical_basis_matches_hand_computed_values():
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vel = np.array([[2.0, 3.0, 5.0], [2.0, 3.0, 5.0], [2.0, 3.0, 5.0]])
    v, flags = rcv.sph_vel(pos, vel)
    assert np.allclose(v[0], [2.0, -5.0, 3.0])      # radial, theta, phi at +x
    assert np.allclose(v[1], [3.0, -5.0, -2.0])     # at +y
    # on the +z axis R = 0: v_r is well defined, v_theta and v_phi are not
    assert abs(v[2, 0] - 5.0) < 1e-12
    assert np.isnan(v[2, 1]) and np.isnan(v[2, 2])
    assert flags["n_degenerate_r"] == 0
    assert flags["n_degenerate_R"] == 1


def test_spherical_basis_on_an_oblique_position():
    pos = np.array([[3.0, 4.0, 0.0]])
    vel = np.array([[1.0, 2.0, 3.0]])
    v, _ = rcv.sph_vel(pos, vel)
    assert np.allclose(v[0], [11.0 / 5.0, -3.0, 2.0 / 5.0])


def test_degenerate_basis_is_flagged_not_silently_nan():
    pos = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    vel = np.ones((3, 3))
    v, flags = rcv.sph_vel(pos, vel)
    assert flags["n_degenerate_R"] == 3 and flags["n_degenerate_r"] == 1
    assert np.isnan(v[0, 1]) and np.isnan(v[0, 2])   # no v_theta / v_phi direction
    assert np.isfinite(v[0, 0])                      # v_r is still defined
    assert np.isnan(v[1]).all()                      # r = 0: nothing is defined
    assert flags["min_r"] == 0.0


# --------------------------------------------------------- block statistics
def _synthetic(n=4000, k=4, shift=(0.30, 0.0, 0.0), widen=(1.0, 1.5, 1.0), seed=5):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.05, 7.4, n)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pos = r[:, None] * u
    w = rng.uniform(0.5, 1.5, n)
    mu = 0.3 * np.sin(r)
    v_true = np.column_stack([mu + rng.normal(0, 1.2, n), rng.normal(0, 1.0, n),
                              rng.normal(0, 0.9, n)])
    gen = np.empty((n, k, 3))
    gen[:, :, 0] = widen[0] * rng.normal(0, 1.2, (n, k)) + shift[0] + mu[:, None]
    gen[:, :, 1] = widen[1] * rng.normal(0, 1.0, (n, k)) + shift[1]
    gen[:, :, 2] = widen[2] * rng.normal(0, 0.9, (n, k)) + shift[2]
    return pos, r, w, v_true, gen


def test_component_stats_reports_bias_sigma_and_w1_consistently():
    _pos, _r, w, v_true, gen = _synthetic(shift=(0.30, 0.0, 0.0), widen=(1.0, 1.5, 1.0))
    k = gen.shape[1]
    rec = rcv.component_stats(v_true, w, gen.reshape(-1, 3), np.repeat(w, k) / k)
    assert abs(rec["vr"]["dmean"] - 100 * 0.30) < 3.0
    assert abs(rec["vth"]["dsigma"] - 100 * 0.5) < 3.0
    assert abs(rec["vT"]["dsigma"]) < 2.0
    # W1 in km/s and the direct two-sample distance must agree
    direct = rcv.KMS * rcv.w1(v_true[:, 0], w, gen[:, :, 0].reshape(-1),
                              np.repeat(w, k) / k)
    assert abs(rec["vr"]["w1"] - direct) < 1e-9
    assert rec["vr"]["w1"] > rec["vT"]["w1"]


def test_paired_bootstrap_interval_covers_the_shift_and_matches_the_truth_draw():
    _pos, _r, w, v_true, gen = _synthetic(shift=(0.30, 0.0, 0.0))
    k = gen.shape[1]
    rng = np.random.default_rng(2)
    boot = rcv.bootstrap_bin(v_true, w, gen, k, rng, 60)
    rec = rcv.component_stats(v_true, w, gen.reshape(-1, 3), np.repeat(w, k) / k)
    lo, hi = boot["dmean|vr"]
    assert lo < rec["vr"]["dmean"] < hi
    wlo, whi = boot["w1|vr"]
    assert wlo <= rec["vr"]["w1"] <= whi
    # a component that is not biased has an interval that contains zero
    lo, hi = boot["dmean|vT"]
    assert lo < 0.0 < hi
    dlo, dhi = boot["drho|vr|vth"]
    assert dlo <= rec["vr|vth"]["drho"] <= dhi


def test_reference_scales_are_positive_and_shrink_with_more_positions():
    _p, _r, w, v_true, _g = _synthetic(n=3000)
    rng = np.random.default_rng(4)
    small = rcv.truth_halfsplit_bin(v_true, w, rng, 40)["vr"]["mean"]
    _p2, _r2, w2, v2, _g2 = _synthetic(n=12000, seed=9)
    big = rcv.truth_halfsplit_bin(v2, w2, rng, 40)["vr"]["mean"]
    assert 0 < big < small                       # a scale, smaller for more stars
    _p3, _r3, w3, v3, g3 = _synthetic(n=3000)
    assert rcv.gen_halfsplit_bin(g3, rng, 20)["vr"] > 0


def test_block_bootstrap_needs_at_least_two_cells():
    _p, _r, w, v_true, gen = _synthetic(n=1500)
    k = gen.shape[1]
    rng = np.random.default_rng(6)
    single = np.zeros(len(w), dtype=int)
    res, nb = rcv.block_bootstrap_bin(v_true, w, gen, k, single, rng, 10)
    assert nb == 1 and np.isnan(res["vr"][0])
    cells = rng.integers(0, 4, len(w))
    res, nb = rcv.block_bootstrap_bin(v_true, w, gen, k, cells, rng, 20)
    assert nb == 4 and np.isfinite(res["vr"][0]) and res["vr"][0] <= res["vr"][1]


def test_out_of_window_fraction_and_reading_helpers():
    v = np.array([-50.0, 0.0, 50.0, 400.0])
    w = np.ones(4)
    assert abs(rcv.frac_out(v, w, -100, 100) - 0.25) < 1e-12
    assert abs(rcv.frac_out(v, w, -500, 500)) < 1e-12
    assert rcv.fnum(1.23456, 2) == "1.23"
    assert rcv.fnum(float("nan")) == "undefined"
    assert rcv.fpm(1.0) == "1.0000"
    assert rcv.fpm(1.0, 0.5, 1.5) == "1.0000 [0.5000, 1.5000]"
    table = rcv.md_table(["a", "b"], [(1, 2), (3, 4)])
    assert table.splitlines()[0] == "| a | b |"
    assert table.splitlines()[2] == "| 1 | 2 |"
    assert len(table.splitlines()) == 4
