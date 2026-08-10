"""Optimizer selection for model recipes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optax


def build_optimizer(name: str, learning_rate: Any) -> optax.GradientTransformation:
    """Build a supported Optax optimizer from a model-config name."""
    builders: dict[str, Callable[[Any], optax.GradientTransformation]] = {
        "adam": optax.adam,
        "radam": optax.radam,
    }
    normalized = str(name).lower().strip()
    try:
        builder = builders[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer {name!r}; expected one of {sorted(builders)}."
        ) from exc
    return builder(learning_rate)
