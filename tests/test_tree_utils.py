from __future__ import annotations

import numpy as np

from dpjax.utils.tree import count_parameters


def test_count_parameters_counts_all_pytree_leaves():
    params = {
        "dense": {
            "kernel": np.zeros((3, 4)),
            "bias": np.zeros((4,)),
        },
        "scale": np.zeros((1,)),
    }

    assert count_parameters(params) == 17
