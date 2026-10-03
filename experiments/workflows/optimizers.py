"""Optimizer selection used by the minimal T4a workflow."""

from __future__ import annotations

import optax


def build_optimizer(name: str, learning_rate) -> optax.GradientTransformation:
    builders = {"adam": optax.adam, "radam": optax.radam}
    key = str(name).lower().strip()
    try:
        return builders[key](learning_rate)
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer {name!r}; expected one of {sorted(builders)}.") from exc
