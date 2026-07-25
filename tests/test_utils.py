from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dpjax.utils.tree import mean_square


def test_mean_square_across_pytree_leaves():
    tree = {
        "a": jnp.array([1.0, 2.0]),
        "b": {"c": jnp.array([3.0])},
    }

    np.testing.assert_allclose(mean_square(tree), (1.0 + 4.0 + 9.0) / 3.0)


def test_mean_square_empty_tree():
    np.testing.assert_allclose(mean_square({}), 0.0)
