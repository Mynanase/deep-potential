from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "plummer"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import t4a_plummer_nf as t4a  # noqa: E402


def test_fixed_probe_hash_and_training_selection_are_deterministic():
    rng = np.random.default_rng(2)
    eta = rng.normal(size=(4096, 6)).astype(np.float32)
    first = t4a.select_fixed_probes(eta)
    second = t4a.select_fixed_probes(eta.copy())
    np.testing.assert_array_equal(first, second)


def test_normalizer_score_chain_rule_is_explicit():
    from dpjax.normalization import fit_normalizer
    rng = np.random.default_rng(3)
    eta = rng.normal(size=(128, 6)).astype(np.float32) * [2, 1, 0.7, .3, .4, .2]
    normalizer = fit_normalizer(eta)
    std_score = np.ones((2, 6), dtype=np.float32)
    np.testing.assert_allclose(std_score / normalizer.std, std_score * (1.0 / normalizer.std), rtol=1e-7)


def test_flow_initialization_and_zero_field_log_prob_gradient():
    import jax
    import jax.numpy as jnp
    from dpjax.flows.api import build_flow, init_flow, log_prob_apply

    config = {"type": "ffjord", "dim": 6, "ffjord": {"hidden_sizes": [8, 8], "n_blocks": 1, "solver": "tsit5", "rtol": 1e-4, "atol": 1e-5}}
    model = build_flow(config)
    params = init_flow(model, jax.random.key(0), config)
    row = jnp.zeros(6, dtype=jnp.float32)
    value = log_prob_apply(model, params, row[None])[0]
    gradient = jax.grad(lambda p: log_prob_apply(model, p, row[None])[0])(params)
    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(jax.tree_util.tree_leaves(gradient)[0])))
