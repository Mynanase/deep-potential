"""Array-only physical-unit calculations."""

from __future__ import annotations

import numpy as np


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


def summarize_density_sign(density: np.ndarray) -> dict[str, float | int]:
    """Summarize sign failures in a density inferred from ``nabla^2 Phi``.

    A physical total gravitating density should be non-negative.  Reporting
    negative and non-positive fractions separately prevents logarithmic plots
    from silently turning Laplacian sign failures into apparently low-density
    pixels.
    """
    density = np.asarray(density, dtype=np.float64)
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise ValueError("density contains no finite values.")

    return {
        "n_total": int(density.size),
        "n_finite": int(finite.size),
        "finite_fraction": float(finite.size / density.size),
        "negative_fraction": float(np.mean(finite < 0.0)),
        "nonpositive_fraction": float(np.mean(finite <= 0.0)),
        "p01": float(np.percentile(finite, 1.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }
