"""Pure NumPy utilities for sampling the analytic Plummer sphere.

The functions in this module are deterministic when supplied with a seeded
``numpy.random.Generator`` and have no dependency on JAX or TensorFlow.  This
keeps data generation fast, testable, and usable in CPU-only environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DF_NORMALIZATION = 24.0 * np.sqrt(2.0) / (7.0 * np.pi**3)
_SPEED_GRID = np.linspace(0.0, np.sqrt(2.0) - 1.0e-8, 1_000)
_SPEED_PDF = _SPEED_GRID**2 * (1.0 - _SPEED_GRID**2 / 2.0) ** (7.0 / 2.0)
_SPEED_CDF = np.concatenate(
    (
        np.array([0.0]),
        np.cumsum(
            0.5
            * (_SPEED_PDF[:-1] + _SPEED_PDF[1:])
            * np.diff(_SPEED_GRID)
        ),
    )
)
_SPEED_CDF /= _SPEED_CDF[-1]


def _as_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _sample_unit_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.column_stack(
        (
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        )
    )


def _validate_sample_args(n: int, max_dist: float | None) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if max_dist is not None and max_dist <= 0.0:
        raise ValueError(f"max_dist must be positive, got {max_dist}.")


@dataclass(frozen=True)
class PlummerSphere:
    """Analytic Plummer distribution and phase-space sampler."""

    @staticmethod
    def psi(r: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 + r**2)

    @classmethod
    def phi(cls, r: np.ndarray) -> np.ndarray:
        return -cls.psi(r)

    @staticmethod
    def rho(r: np.ndarray) -> np.ndarray:
        return 3.0 / (4.0 * np.pi) * (1.0 + r**2) ** (-5.0 / 2.0)

    @staticmethod
    def sample_radius(
        n: int,
        *,
        max_dist: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample radii, optionally from the distribution truncated at a radius."""
        _validate_sample_args(n, max_dist)
        rng = _as_rng(rng)

        if max_dist is None:
            cdf_max = 1.0
        else:
            cdf_max = (max_dist / np.sqrt(1.0 + max_dist**2)) ** 3

        u = rng.uniform(np.finfo(np.float64).tiny, cdf_max, size=n)
        return 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)

    def sample(
        self,
        n: int,
        *,
        max_dist: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return position and velocity arrays with shape ``(n, 3)``."""
        _validate_sample_args(n, max_dist)
        rng = _as_rng(rng)

        radius = self.sample_radius(n, max_dist=max_dist, rng=rng)
        position = radius[:, None] * _sample_unit_sphere(n, rng)

        scaled_speed = np.interp(
            rng.uniform(size=n),
            _SPEED_CDF,
            _SPEED_GRID,
        )
        speed = np.sqrt(self.psi(radius)) * scaled_speed
        velocity = speed[:, None] * _sample_unit_sphere(n, rng)
        return position, velocity

    def sample_df(
        self,
        n: int,
        *,
        max_dist: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward-compatible alias for :meth:`sample`."""
        return self.sample(n, max_dist=max_dist, rng=rng)

    def df(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        eta = np.concatenate((position, velocity), axis=1)
        return plummer_df(eta)


def sample_plummer(
    n: int,
    *,
    max_dist: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return exactly ``n`` Plummer samples as ``float32`` ``(n, 6)`` data."""
    position, velocity = PlummerSphere().sample(
        n,
        max_dist=max_dist,
        rng=rng,
    )
    return np.concatenate((position, velocity), axis=1).astype(np.float32)


def plummer_df(eta: np.ndarray) -> np.ndarray:
    """Evaluate the normalized Plummer distribution function."""
    eta = np.asarray(eta)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")

    radius_squared = np.sum(eta[:, :3] ** 2, axis=1)
    speed_squared = np.sum(eta[:, 3:] ** 2, axis=1)
    relative_energy = (1.0 + radius_squared) ** (-0.5) - 0.5 * speed_squared
    return _DF_NORMALIZATION * np.clip(relative_energy, 0.0, np.inf) ** (7.0 / 2.0)


def split_train_test(
    eta: np.ndarray,
    *,
    test_n: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and split ``eta`` into non-overlapping train and test arrays."""
    eta = np.asarray(eta)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"Expected eta shape (N, 6), got {eta.shape}.")
    if not 0 < test_n < len(eta):
        raise ValueError(
            f"test_n must be between 1 and {len(eta) - 1}, got {test_n}."
        )

    indices = _as_rng(rng).permutation(len(eta))
    return eta[indices[test_n:]], eta[indices[:test_n]]
