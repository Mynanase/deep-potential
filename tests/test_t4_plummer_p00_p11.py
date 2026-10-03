from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import t4_plummer_p00_p11 as t4  # noqa: E402


def test_quota_total_matches_s1_contract():
    assert t4.QUOTAS.sum() == t4.N_CONSTRAINTS


def test_stratified_constraints_are_full_data_and_deterministic():
    eta = np.empty((40_000, 6), dtype=np.float32)
    # Spread the small test population across the seven frozen q bins.
    starts = np.array([0.11, 0.21, 1.1, 2.1, 3.1, 4.6, 6.1], dtype=np.float32)
    eta[:, 0] = np.tile(starts, len(eta) // len(starts) + 1)[:len(eta)]
    eta[:, 1:] = 0.0
    quota_backup = t4.QUOTAS.copy()
    try:
        t4.QUOTAS[:] = np.maximum(1, np.rint(t4.QUOTAS / 64).astype(int))
        t4.N_CONSTRAINTS_BACKUP = t4.N_CONSTRAINTS
        t4.N_CONSTRAINTS = int(t4.QUOTAS.sum())
        for i in np.flatnonzero(t4.QUOTAS > 71_428):
            excess = int(t4.QUOTAS[i] - 71_428)
            t4.QUOTAS[i] -= excess
            t4.QUOTAS[4] += excess
        rows, constrained = t4.stratified_constraints(eta)
        rows2, constrained2 = t4.stratified_constraints(eta.copy())
        np.testing.assert_array_equal(rows, rows2)
        np.testing.assert_array_equal(constrained, constrained2)
        assert np.all((np.linalg.norm(constrained[:, :3], axis=1) >= 0.1))
    finally:
        t4.QUOTAS[:] = quota_backup
        t4.N_CONSTRAINTS = t4.N_CONSTRAINTS_BACKUP
        del t4.N_CONSTRAINTS_BACKUP


def test_oracle_reference_conventions():
    eta = np.zeros((1, 6), dtype=np.float64); eta[0, 0] = t4.B_CODE
    alpha = t4.oracle_alpha(eta)[0]
    expected = t4.B_CODE / (2.0 * t4.B_CODE ** 2) ** 1.5
    assert alpha[0] < 0 and np.isclose(np.linalg.norm(alpha), expected, rtol=1e-12)
