"""Affine phase-space normalization stored with the mock NF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_phase_space(eta: np.ndarray, *, name: str = "eta") -> np.ndarray:
    values = np.asarray(eta, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected {name} shape (N, 6), got {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return values


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("Normalizer mean/std must have shape (6,).")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Normalizer mean/std must be finite.")
        if np.any(std <= 0.0):
            raise ValueError("Normalizer std must be strictly positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def transform(self, eta: np.ndarray) -> np.ndarray:
        return (validate_phase_space(eta) - self.mean) / self.std

    def inverse(self, eta_std: np.ndarray) -> np.ndarray:
        return validate_phase_space(eta_std, name="eta_std") * self.std + self.mean


def fit_normalizer(eta: np.ndarray) -> Normalizer:
    """Fit an equal-weight affine normalizer in float64, then store f32."""
    values = validate_phase_space(eta)
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.maximum(np.std(values, axis=0, dtype=np.float64), 1e-6)
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))
