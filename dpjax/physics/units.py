"""Physical-unit helpers shared by potential evaluation and plotting."""

from __future__ import annotations

import numpy as np


# G in (km/s)^2 kpc / Msun.  This convention matches the canonical Auriga
# contract: positions in kpc and potentials in (km/s)^2.
G_KPC_KMS2_PER_MSUN = 4.30091727003628e-6


def gravitational_constant_for_system(
    system: str,
    override: float | None = None,
) -> float:
    """Return the Poisson-equation G for a named unit system.

    Plummer and generic experiments retain the historical dimensionless
    convention G=1.  Halo experiments use kpc, km/s, and solar masses.
    """
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("gravitational_constant must be finite and positive.")
        return value
    return G_KPC_KMS2_PER_MSUN if str(system).lower() == "halo" else 1.0


def density_from_laplacian(
    laplacian: np.ndarray,
    *,
    gravitational_constant: float,
) -> np.ndarray:
    """Convert ``nabla^2 Phi`` to total gravitating density."""
    gravitational_constant = float(gravitational_constant)
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive.")
    return np.asarray(laplacian) / (4.0 * np.pi * gravitational_constant)
