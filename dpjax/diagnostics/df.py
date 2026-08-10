"""Self-contained comparison of generated and reference phase-space rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpjax.normalization import PHASE_SPACE_DIM, validate_phase_space


@dataclass(frozen=True)
class DFDiagnostics:
    """Per-coordinate summaries and shared-bin marginal histograms."""

    reference_count: int
    generated_count: int
    reference_mean: np.ndarray
    generated_mean: np.ndarray
    reference_std: np.ndarray
    generated_std: np.ndarray
    bin_edges: np.ndarray
    reference_density: np.ndarray
    generated_density: np.ndarray


def compare_df_samples(
    reference_eta: np.ndarray,
    generated_eta: np.ndarray,
    *,
    bins: int = 64,
    reference_weight: np.ndarray | None = None,
) -> DFDiagnostics:
    """Compare two ``(N, 6)`` samples without file or dataset assumptions."""
    reference = validate_phase_space(reference_eta, name="reference_eta")
    generated = validate_phase_space(generated_eta, name="generated_eta")
    bins = int(bins)
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    weights = None
    if reference_weight is not None:
        weights = np.asarray(reference_weight, dtype=np.float64)
        if weights.shape != (reference.shape[0],):
            raise ValueError(
                f"Expected reference_weight shape ({reference.shape[0]},), "
                f"got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                "reference_weight must be finite and strictly positive."
            )

    edges = np.empty((PHASE_SPACE_DIM, bins + 1), dtype=np.float64)
    reference_density = np.empty((PHASE_SPACE_DIM, bins), dtype=np.float64)
    generated_density = np.empty((PHASE_SPACE_DIM, bins), dtype=np.float64)
    for dimension in range(PHASE_SPACE_DIM):
        combined = np.concatenate(
            [reference[:, dimension], generated[:, dimension]]
        )
        lower, upper = np.quantile(combined, [0.001, 0.999])
        if not upper > lower:
            lower -= 0.5
            upper += 0.5
        dimension_edges = np.linspace(lower, upper, bins + 1)
        edges[dimension] = dimension_edges
        reference_density[dimension] = np.histogram(
            reference[:, dimension],
            bins=dimension_edges,
            weights=weights,
            density=True,
        )[0]
        generated_density[dimension] = np.histogram(
            generated[:, dimension],
            bins=dimension_edges,
            density=True,
        )[0]

    if weights is None:
        reference_mean = np.mean(reference, axis=0)
        reference_variance = np.var(reference, axis=0)
    else:
        reference_mean = np.average(reference, axis=0, weights=weights)
        reference_variance = np.average(
            (reference - reference_mean) ** 2,
            axis=0,
            weights=weights,
        )

    return DFDiagnostics(
        reference_count=int(reference.shape[0]),
        generated_count=int(generated.shape[0]),
        reference_mean=np.asarray(reference_mean, dtype=np.float64),
        generated_mean=np.mean(generated, axis=0, dtype=np.float64),
        reference_std=np.sqrt(reference_variance),
        generated_std=np.std(generated, axis=0, dtype=np.float64),
        bin_edges=edges,
        reference_density=reference_density,
        generated_density=generated_density,
    )
