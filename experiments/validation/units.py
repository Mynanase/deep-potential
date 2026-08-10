"""Unit conventions selected by repository validation workflows."""

from __future__ import annotations

import numpy as np

G_KPC_KMS2_PER_MSUN = 4.30091727003628e-6


def gravitational_constant_for_system(
    system: str,
    override: float | None = None,
) -> float:
    """Resolve the experiment unit convention to a gravitational constant."""
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("gravitational_constant must be finite and positive.")
        return value
    return G_KPC_KMS2_PER_MSUN if str(system).lower() == "halo" else 1.0
