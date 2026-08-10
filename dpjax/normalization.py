"""Six-dimensional phase-space validation and numerical normalization.

This module is deliberately array-only: it does not read files, select rows,
or know which survey or simulation produced the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASE_SPACE_DIM = 6


def validate_phase_space(
    eta: np.ndarray,
    *,
    name: str = "eta",
    allow_empty: bool = False,
) -> np.ndarray:
    """Return a finite float32 ``(N, 6)`` phase-space array."""
    values = np.asarray(eta, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != PHASE_SPACE_DIM:
        raise ValueError(
            f"Expected {name} shape (N, {PHASE_SPACE_DIM}), got {values.shape}."
        )
    if not allow_empty and values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return values


@dataclass(frozen=True)
class Normalizer:
    """Affine normalization stored as part of the learned model state."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        expected = (PHASE_SPACE_DIM,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"Expected mean/std shape {expected}, got {mean.shape}/{std.shape}."
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Normalizer mean/std must be finite.")
        if np.any(std <= 0.0):
            raise ValueError("Normalizer std must be strictly positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def transform(self, eta: np.ndarray) -> np.ndarray:
        values = validate_phase_space(eta)
        return (values - self.mean) / self.std

    def inverse(self, eta_std: np.ndarray) -> np.ndarray:
        values = validate_phase_space(eta_std, name="eta_std")
        return values * self.std + self.mean

    def transform_score(self, score_std: np.ndarray) -> np.ndarray:
        """Convert a score from standardized to physical coordinates."""
        values = validate_phase_space(score_std, name="score_std")
        return values / self.std


def fit_normalizer(
    eta: np.ndarray,
    *,
    eps: float = 1.0e-6,
    sample_weight: np.ndarray | None = None,
) -> Normalizer:
    """Fit affine normalization to phase-space rows."""
    values = validate_phase_space(eta)
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    if sample_weight is None:
        mean = np.mean(values, axis=0, dtype=np.float64)
        variance = np.var(values, axis=0, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (values.shape[0],):
            raise ValueError(
                f"Expected sample_weight shape ({values.shape[0]},), "
                f"got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("sample_weight must be finite and strictly positive.")
        weight_sum = float(np.sum(weights))
        mean = np.sum(values * weights[:, None], axis=0) / weight_sum
        variance = (
            np.sum(weights[:, None] * (values - mean[None, :]) ** 2, axis=0)
            / weight_sum
        )

    return Normalizer(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.maximum(
            np.sqrt(variance).astype(np.float32),
            np.float32(eps),
        ),
    )
