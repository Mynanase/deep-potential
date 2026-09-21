#!/usr/bin/env python
"""Tests for the cosine lambda-anneal schedule helper. CPU only."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import potential as pmod  # noqa: E402


def test_endpoints():
    T = 256
    np.testing.assert_allclose(
        pmod.cosine_anneal_value(0, T, 10.0, 1.0), 10.0, rtol=1e-12)
    np.testing.assert_allclose(
        pmod.cosine_anneal_value(T - 1, T, 10.0, 1.0), 1.0, rtol=1e-7)


def test_midpoint_mean():
    T = 256
    np.testing.assert_allclose(
        pmod.cosine_anneal_value(T // 2, T, 10.0, 1.0), 5.5, rtol=1e-3)


def test_monotone_decrease():
    vals = [pmod.cosine_anneal_value(t, 64, 10.0, 1.0) for t in range(64)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))

