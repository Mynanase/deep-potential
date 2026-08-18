from __future__ import annotations

import numpy as np
import pytest

from dpjax.physics.units import (
    density_from_laplacian,
    summarize_density_sign,
)
from experiments.workflows.evaluation.units import (
    G_KPC_KMS2_PER_MSUN,
    gravitational_constant_for_system,
)


def test_halo_density_uses_physical_gravitational_constant():
    assert gravitational_constant_for_system("halo") == pytest.approx(
        G_KPC_KMS2_PER_MSUN
    )
    assert gravitational_constant_for_system("plummer") == pytest.approx(1.0)
    laplacian = np.array([4.0 * np.pi * G_KPC_KMS2_PER_MSUN])
    density = density_from_laplacian(
        laplacian,
        gravitational_constant=G_KPC_KMS2_PER_MSUN,
    )
    np.testing.assert_allclose(density, [1.0])


def test_gravitational_constant_override_is_validated():
    assert gravitational_constant_for_system("halo", 2.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="finite and positive"):
        gravitational_constant_for_system("halo", 0.0)


def test_density_sign_summary_keeps_negative_fraction_visible():
    summary = summarize_density_sign(
        np.array([-2.0, 0.0, 1.0, np.nan])
    )

    assert summary["n_total"] == 4
    assert summary["n_finite"] == 3
    assert summary["finite_fraction"] == pytest.approx(0.75)
    assert summary["negative_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["nonpositive_fraction"] == pytest.approx(2.0 / 3.0)
