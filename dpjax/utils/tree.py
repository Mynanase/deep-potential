"""Small, reusable operations on JAX parameter pytrees."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def count_parameters(tree: Any) -> int:
    """Return the total number of scalar values in a parameter pytree."""
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))


def mean_square(tree: Any) -> jnp.ndarray:
    """Return the element-wise mean square across all leaves of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)

    square_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    element_count = sum(leaf.size for leaf in leaves)
    return square_sum / jnp.asarray(element_count, dtype=jnp.float32)
