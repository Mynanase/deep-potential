from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "scripts" / "auriga"))
import t4a_plummer_nf as t4a  # noqa: E402

def test_fixed_probe_rows_use_radius_stratification():
    rng = np.random.default_rng(2); eta = rng.normal(size=(4096, 6)).astype(np.float32)
    first = t4a.fixed_probe_rows(eta); second = t4a.fixed_probe_rows(eta.copy())
    np.testing.assert_array_equal(first, second); radius = np.linalg.norm(eta[:, :3], axis=1); selected = radius[first]
    assert selected[0] == radius.min() and selected[-1] == radius.max()

def test_analytic_log_prob_uses_full_plummer_units():
    rng = np.random.default_rng(3)
    eta = np.column_stack((rng.normal(size=(32, 3)) * 2.0, rng.normal(size=(32, 3)) * 0.1))
    log_prob = t4a.analytic_ln_prob(eta); assert log_prob.shape == (32,) and np.all(np.isfinite(log_prob))

def test_project_conditional_flow_split_log_prob_identity():
    import jax, jax.numpy as jnp
    from flow_ot_flow_matching_conditional import ConditionalPhaseSpaceFlow
    params = {"type": "MLP", "width": 8, "depth": 2}; flow = ConditionalPhaseSpaceFlow(key=jax.random.key(0), data_mean=jnp.zeros(6), data_std=jnp.ones(6), spatial_vf_params=params, conditional_vf_params=params)
    eta = jnp.asarray(np.random.default_rng(4).normal(size=(8, 6)) * 0.2); full = np.asarray(flow.log_prob(eta)); split = np.asarray(flow.log_prob_position(eta[:, :3]) + flow.log_prob_velocity_given_position(eta))
    np.testing.assert_allclose(full, split, rtol=2e-6, atol=2e-6)
