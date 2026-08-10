from __future__ import annotations

import numpy as np
import pytest

from dpjax.normalization import Normalizer, fit_normalizer, validate_phase_space


def test_validate_phase_space_enforces_the_core_six_dimensional_contract():
    eta = np.arange(24, dtype=np.float64).reshape(4, 6)

    validated = validate_phase_space(eta)

    assert validated.dtype == np.float32
    np.testing.assert_array_equal(validated, eta)
    with pytest.raises(ValueError, match=r"shape \(N, 6\)"):
        validate_phase_space(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_phase_space(np.full((1, 6), np.nan, dtype=np.float32))


def test_normalizer_round_trip_and_physical_score_scaling():
    eta = np.arange(24, dtype=np.float32).reshape(4, 6)
    normalizer = fit_normalizer(eta)
    standardized = normalizer.transform(eta)

    np.testing.assert_allclose(normalizer.inverse(standardized), eta)
    score_std = np.ones((2, 6), dtype=np.float32)
    np.testing.assert_allclose(
        normalizer.transform_score(score_std),
        score_std / normalizer.std,
    )


def test_normalizer_rejects_invalid_model_state():
    with pytest.raises(ValueError, match="strictly positive"):
        Normalizer(
            mean=np.zeros(6, dtype=np.float32),
            std=np.zeros(6, dtype=np.float32),
        )
