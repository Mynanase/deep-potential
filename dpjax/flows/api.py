"""Array-only flow factory and evaluation helpers for the mock NF."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


def build_flow(config: dict) -> nn.Module:
    """Build one configured flow backend."""
    from dpjax.flows.ffjord import MockFFJORD, MockFFJORDConfig

    ffjord = config.get("ffjord", {})
    return MockFFJORD(MockFFJORDConfig(**{
        "dim": int(config.get("dim", 6)),
        "hidden_sizes": tuple(ffjord.get("hidden_sizes", [64, 64, 64])),
        "n_blocks": int(ffjord.get("n_blocks", 2)),
        "solver": str(ffjord.get("solver", "tsit5")),
        "rtol": float(ffjord.get("rtol", 1e-4)),
        "atol": float(ffjord.get("atol", 1e-5)),
    }))


def init_flow(model: nn.Module, rng: jax.Array, config: dict) -> dict:
    """Initialize parameters without tracing an ODE solve."""
    from dpjax.flows.ffjord import MockFFJORD

    dim = int(config.get("dim", 6))
    return model.init(rng, jnp.zeros((1, dim), dtype=jnp.float32), method=MockFFJORD.init_only)["params"]


def _log_prob(model: nn.Module, params: dict, eta: jnp.ndarray):
    from dpjax.flows.ffjord import MockFFJORD

    return model.apply({"params": params}, eta, method=MockFFJORD.log_prob)


def log_prob_apply(model: nn.Module, params: dict, eta: jnp.ndarray) -> jnp.ndarray:
    """Return log probability in input eta coordinates."""
    return _log_prob(model, params, eta)


def score_apply(model: nn.Module, params: dict, eta: jnp.ndarray) -> jnp.ndarray:
    """Return grad_eta log probability."""
    def single(row):
        return _log_prob(model, params, row[None])[0]
    return jax.vmap(jax.grad(single))(eta)


def sample_apply(model: nn.Module, params: dict, rng: jax.Array, n: int):
    from dpjax.flows.ffjord import MockFFJORD

    return model.apply({"params": params}, rng, int(n), method=MockFFJORD.sample)
