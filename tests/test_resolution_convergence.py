#!/usr/bin/env python
"""Analytic tests for the resolution-convergence machinery.

Validate the algebra, not checkpoints: the Hann/rfft periodogram recovers a
known wavelength, f_hi is exact on a two-component signal, the NGP histogram
places mass in the right cells with exact closure, log-lambda binning and
the pre-registered lambda_reliable rule behave as specified, and Sobol
directions are deterministic unit vectors.  CPU only, no data files.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import resolution_convergence as rc  # noqa: E402

R_MIN, R_MAX, N_RADIAL = 1.09, 70.0, 128


def _phase_varied_signal(lam, amp, n_dirs, seed):
    """Rows = rays of amp*sin(2*pi*r/lam + phi_j): survives the angular-mean
    removal that the pipeline applies per radius."""
    rng = np.random.default_rng(seed)
    r = np.linspace(R_MIN, R_MAX, N_RADIAL)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_dirs)
    return amp * np.sin(2.0 * np.pi * r[None, :] / lam + phase[:, None]), r


def test_periodogram_recovers_wavelength_and_f_hi_bracket():
    mat, r = _phase_varied_signal(6.0, 1.0, 64, 1)
    dr = float(r[1] - r[0])
    delta = mat - mat.mean(axis=0, keepdims=True)
    freq, power = rc.radial_spectrum(delta, dr)
    lam = 1.0 / freq
    assert abs(lam[int(np.argmax(power))] - 6.0) < 0.6
    assert rc.f_hi(freq, power, 4.0) < 0.05
    assert rc.f_hi(freq, power, 8.0) > 0.90


def test_f_hi_two_component_exact_fraction():
    # on-grid tones (lam = N*dr/m): the Hann main lobe spans +-1 bin, so the
    # slow tone sits at m=5 (bins 4-6 all >10 kpc) and the fast tone at m=23
    # (bins 22-24 all <4 kpc); the analytic f_hi is a1^2/(a1^2+a2^2)
    r = np.linspace(R_MIN, R_MAX, N_RADIAL)
    dr = float(r[1] - r[0])
    T = N_RADIAL * dr
    fast, r = _phase_varied_signal(T / 23.0, 1.0, 64, 2)
    slow, _ = _phase_varied_signal(T / 5.0, 3.0, 64, 3)
    delta = fast + slow
    delta = delta - delta.mean(axis=0, keepdims=True)
    freq, power = rc.radial_spectrum(delta, dr)
    assert abs(rc.f_hi(freq, power, 4.0) - 0.1) < 0.03
    assert abs(rc.f_hi(freq, power, 10.0) - 0.1) < 0.03
    assert rc.f_hi(freq, power, 2.5) < 0.02


def test_build_rho3d_placement_and_closure():
    n, ext = 8, 10.0
    xyz = np.array([[-9.9, -9.9, -9.9], [9.9, 9.9, 9.9], [0.11, 0.0, 0.0]])
    m = np.array([2.0, 3.0, 5.0])
    rho, edges, cell, closure, fill = rc.build_rho3d(xyz, m, n, ext)
    assert abs(cell - 2.5) < 1e-12
    assert abs(closure - 1.0) < 1e-12
    assert abs(rho[0, 0, 0] - 2.0 / cell ** 3) < 1e-6
    assert abs(rho[7, 7, 7] - 3.0 / cell ** 3) < 1e-6
    assert abs(rho[4, 4, 4] - 5.0 / cell ** 3) < 1e-6
    assert abs(fill - 3.0 / n ** 3) < 1e-12


def test_bin_median_and_reliable_rule():
    edges = np.array([1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])
    centers = np.sqrt(edges[:-1] * edges[1:])
    lam = np.array([1.6, 1.8, 2.2, 3.1, 3.5, 4.4, 6.0, 8.0])
    vals = np.array([0.5, 0.5, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05])
    b = rc.bin_median(lam, vals, edges)
    assert np.allclose(b[[0, 1, 3, 4, 5, 6]], [0.5, 0.2, 0.05, 0.05, 0.05, 0.05])
    assert np.isnan(b[2])
    converged = b < 0.1
    lam_rel, lam_emp = rc.reliable_bin_center(centers, converged, 3.125)
    assert abs(lam_emp - np.sqrt(3.0 * 4.0)) < 1e-12
    assert abs(lam_rel - np.sqrt(3.0 * 4.0)) < 1e-12
    lam_rel, _ = rc.reliable_bin_center(centers, converged, 5.0)
    assert abs(lam_rel - 5.0) < 1e-12
    lam_rel, lam_emp = rc.reliable_bin_center(centers, np.ones(len(centers), bool), 3.125)
    assert abs(lam_emp - centers[0]) < 1e-12
    assert abs(lam_rel - 3.125) < 1e-12
    lam_rel, lam_emp = rc.reliable_bin_center(centers, np.zeros(len(centers), bool), 3.125)
    assert np.isnan(lam_rel) and np.isnan(lam_emp)


def test_sobol_dirs_deterministic_unit_norm():
    a = rc.sobol_dirs(64, 20260921)
    b = rc.sobol_dirs(64, 20260921)
    c = rc.sobol_dirs(64, 20260922)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
