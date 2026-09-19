#!/usr/bin/env python
"""Shared helpers for particle-level total-matter truth diagnostics."""
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import G_KPC_KMS2_MSUN  # noqa: E402

TYPES = ("PartType0", "PartType1", "PartType4")


def load_particles(asset, r_max=75.0):
    """All particle types from the star-frame asset: (xyz kpc f8, mass Msun f8)."""
    xyzs, ms = [], []
    with h5py.File(asset, "r") as f:
        for ptype in TYPES:
            g = f[ptype]
            xyz = np.stack([g["x"][:], g["y"][:], g["z"][:]], axis=1).astype(np.float64)
            xyzs.append(xyz)
            ms.append(g["mass"][:].astype(np.float64))
    xyz = np.concatenate(xyzs)
    m = np.concatenate(ms)
    if r_max is not None:
        keep = np.linalg.norm(xyz, axis=1) <= r_max
        xyz, m = xyz[keep], m[keep]
    return xyz, m


def phi_direct(xyz, m, query, eps_kpc=1e-3, chunk_q=1024, chunk_p=2_000_000):
    """Direct-summation potential -G sum m_i/|x-q_i| [(km/s)^2] via GPU.

    fp32 pair kernels with fp64 accumulation: exact enough at the ~1e-3 level
    of the potential depth for figures, seconds on an A100.
    """
    import jax
    import jax.numpy as jnp
    P = jnp.asarray(xyz, dtype=jnp.float32)
    M = jnp.asarray(m, dtype=jnp.float32)
    out = np.empty(query.shape[0], dtype=np.float64)

    def block(qb):
        acc = jnp.zeros(qb.shape[0], dtype=jnp.float64)
        for i in range(0, P.shape[0], chunk_p):
            pb = P[i:i + chunk_p]
            mb = M[i:i + chunk_p]
            d2 = jnp.sum((qb[:, None, :] - pb[None, :, :]) ** 2, axis=2) + eps_kpc ** 2
            acc += jnp.sum(mb / jnp.sqrt(d2), axis=1).astype(jnp.float64)
        return -G_KPC_KMS2_MSUN * acc

    block_jit = jax.jit(block)
    for i in range(0, query.shape[0], chunk_q):
        qb = jnp.asarray(query[i:i + chunk_q], dtype=jnp.float32)
        out[i:i + chunk_q] = np.asarray(block_jit(qb).block_until_ready())
    return out

