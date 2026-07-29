from __future__ import annotations

import numpy as np
import pytest

from dpjax.physics.units import (
    G_KPC_KMS2_PER_MSUN,
    density_from_laplacian,
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
