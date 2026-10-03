"""Minimal FFJORD model used by the T4a full-Plummer mock."""

from __future__ import annotations

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class MockFFJORDConfig:
    dim: int = 6
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    n_blocks: int = 3
    solver: str = "tsit5"
    rtol: float = 1e-4
    atol: float = 1e-5


class VelocityField(nn.Module):
    hidden_sizes: tuple[int, ...]
    dim: int

    @nn.compact
    def __call__(self, t, x):
        position = x if getattr(x, "ndim", None) == 1 else x[0]
        values = jnp.concatenate((jnp.atleast_1d(t.astype(position.dtype)), position))
        for width in self.hidden_sizes:
            values = nn.tanh(nn.Dense(width)(values))
        return nn.Dense(self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(values)


class MockFFJORD(nn.Module):
    cfg: MockFFJORDConfig

    def setup(self):
        self.vfs = [VelocityField(self.cfg.hidden_sizes, self.cfg.dim, name=f"vf_{i}") for i in range(self.cfg.n_blocks)]

    def init_only(self, x):
        z = x[0]
        t = jnp.asarray(0.0, dtype=z.dtype)
        for vf in self.vfs:
            z = vf(t, z)
        return z

    def log_prob(self, x):
        return jax.vmap(lambda row: _log_prob_single(self, row))(x)

    def sample(self, rng, n):
        return jax.random.normal(rng, (n, self.cfg.dim), dtype=jnp.float32)


def _solver(name):
    return {"tsit5": diffrax.Tsit5, "dopri5": diffrax.Dopri5}[name]()


def _log_prob_single(model, row):
    z = row
    delta = jnp.zeros((), dtype=row.dtype)
    for block in reversed(range(model.cfg.n_blocks)):
        vf = model.vfs[block]
        def rhs(t, y, args):
            f = vf(t, y)
            div = jnp.trace(jax.jacfwd(lambda x: vf(t, x))(y))
            return f, -div, jnp.zeros((), dtype=y.dtype)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), _solver(model.cfg.solver), 1.0, 0.0, -0.01,
            (z, delta, jnp.zeros((), dtype=z.dtype)),
            stepsize_controller=diffrax.PIDController(rtol=model.cfg.rtol, atol=model.cfg.atol),
            saveat=diffrax.SaveAt(t1=True),
        )
        z, delta, _ = sol.ys[0][0], sol.ys[1][0], sol.ys[2][0]
    base = -0.5 * (jnp.sum(z * z) + model.cfg.dim * jnp.log(2 * jnp.pi))
    return base - delta


def log_prob_single(model, row):
    return _log_prob_single(model, row)
