from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class PotentialConfig:
    hidden_sizes: tuple[int, ...] = (512, 512, 512, 512)
    output_scale: float = 1.0


class PotentialMLP(nn.Module):
    cfg: PotentialConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 3)
        h = x
        for i, width in enumerate(self.cfg.hidden_sizes):
            h = nn.Dense(width, name=f"dense_{i}")(h)
            h = nn.tanh(h)
        out = nn.Dense(1, name="dense_out")(h)
        return jnp.squeeze(out, axis=-1) * self.cfg.output_scale


def phi_apply(model: PotentialMLP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return model.apply({"params": params}, x)


def grad_phi_apply(model: PotentialMLP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Compute ∇_x Φ(x) for x with shape (N, 3)."""

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, xi)

    return jax.vmap(jax.grad(phi_single))(x)


def laplacian_phi_apply(
    model: PotentialMLP,
    params: dict,
    x: jnp.ndarray,
    *,
    std_x: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute ∇²Φ(x) for x with shape (N, 3).

    If ``std_x`` is provided, the Laplacian is converted from standardized
    coordinates to physical coordinates using chain-rule scaling.
    """

    def phi_single(xi: jnp.ndarray) -> jnp.ndarray:
        return model.apply({"params": params}, xi)

    hess = jax.vmap(jax.hessian(phi_single))(x)
    diag = jnp.stack([hess[:, 0, 0], hess[:, 1, 1], hess[:, 2, 2]], axis=-1)

    if std_x is None:
        return jnp.sum(diag, axis=-1)

    std_x = jnp.asarray(std_x, dtype=x.dtype)
    return jnp.sum(diag / (std_x[None, :] ** 2), axis=-1)
