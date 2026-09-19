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


def phi_direct(xyz, m, query, eps_kpc=1e-3, chunk_q=256, chunk_p=250_000):
    """Direct-summation potential -G sum m_i/|x-q_i| [(km/s)^2] via GPU.

    One jit-compiled kernel for fixed chunk shapes; the particle loop runs
    outside jit with fp64 numpy accumulation (avoids the pathological XLA
    input_reduce_fusion compile for large 2-D reductions).
    """
    import jax
    import jax.numpy as jnp
    P = jnp.asarray(xyz, dtype=jnp.float32)
    M = jnp.asarray(m, dtype=jnp.float32)
    out = np.empty(query.shape[0], dtype=np.float64)
    eps2 = float(eps_kpc) ** 2

    def kernel(qb, pb, mb):
        d2 = jnp.sum((qb[:, None, :] - pb[None, :, :]) ** 2, axis=2) + eps2
        return jnp.sum(mb / jnp.sqrt(d2), axis=1)

    kernel_jit = jax.jit(kernel)
    n_q = query.shape[0]
    pad = (-n_q) % chunk_q
    q_all = np.concatenate([query, np.repeat(query[:1], pad, axis=0)]) if pad else query
    parts = [(P[i:i + chunk_p], M[i:i + chunk_p])
             for i in range(0, P.shape[0], chunk_p)]
    for i in range(0, q_all.shape[0], chunk_q):
        qb = jnp.asarray(q_all[i:i + chunk_q], dtype=jnp.float32)
        acc = np.zeros(min(chunk_q, q_all.shape[0] - i), dtype=np.float64)
        for pb, mb in parts:
            acc += np.asarray(kernel_jit(qb, pb, mb).block_until_ready(),
                              dtype=np.float64)
        j0, j1 = i, min(i + chunk_q, n_q)
        out[j0:j1] = acc[:j1 - j0]
    return -G_KPC_KMS2_MSUN * out
