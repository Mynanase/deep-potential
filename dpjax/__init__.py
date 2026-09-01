"""Array-only JAX/Flax core for Deep Potential."""

from dpjax.normalization import Normalizer, fit_normalizer, validate_phase_space

__version__ = "0.1.0"

__all__ = [
    "Normalizer",
    "fit_normalizer",
    "validate_phase_space",
    "__version__",
]
