#!/usr/bin/env python
"""Local acceleration inversion and projection diagnostics (alpha convention)."""

from __future__ import annotations

import numpy as np


def acceleration_from_grad_phi(grad_phi: np.ndarray) -> np.ndarray:
    grad_phi = np.asarray(grad_phi, dtype=np.float64)
    if grad_phi.ndim != 2 or grad_phi.shape[1] != 3:
        raise ValueError(f"expected grad_phi with shape (N, 3), got {grad_phi.shape}")
    return -grad_phi


def weighted_mean(x: np.ndarray, weights: np.ndarray | None = None, axis: int | None = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if weights is None:
        return np.mean(x, axis=axis)
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if w.shape != x.shape and w.shape != np.asarray(x).shape:
        # np.average-style scalar weights are useful in tests; broadcasting is
        # intentionally avoided for scientific row weights.
        raise ValueError("weights and values must have the same shape")
    return np.sum(w * x, axis=axis) / np.maximum(np.sum(w, axis=axis), 1.0e-300)


def local_force_svd_alpha(C, y, weights=None, rcond=1e-6):
    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 3 or len(y) != len(C):
        raise ValueError("expected C shape (N, 3) and matching y")
    if weights is None:
        w = np.ones(len(y))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
    d = np.sqrt(w)[:, None] * C
    yw = np.sqrt(w) * y
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    rank = int(np.count_nonzero(s > rcond * (s[0] if len(s) and s[0] > 0 else 0.0)))
    alpha = np.zeros(C.shape[1]) if rank < C.shape[1] else vt.T @ ((u.T @ yw) / s)
    fitted = d @ alpha if rank == C.shape[1] else d @ np.linalg.lstsq(d, yw, rcond=rcond)[0]
    resid = fitted - yw
    return {
        "sigma": s,
        "rank": int(rank),
        "cond": float(s[0] / s[-1]) if len(s) and s[-1] > 0 else float("inf"),
        "resid_rel": float(np.linalg.norm(resid) / max(np.linalg.norm(yw), 1.0e-300)),
        "alpha_hat": alpha if rank == C.shape[1] else None,
        "y_norm": float(np.linalg.norm(yw)),
    }


def absorbable_split(C, e, weights=None, rcond=1e-6):
    C = np.asarray(C, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if C.ndim != 2 or C.shape[1] != 3 or len(e) != len(C):
        raise ValueError("expected C shape (N, 3) and matching e")
    w = np.ones(len(e)) if weights is None else np.asarray(weights, dtype=np.float64)
    d = np.sqrt(w)[:, None] * C
    ew = np.sqrt(w) * e
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    keep = s > rcond * (s[0] if len(s) and s[0] > 0 else 0.0)
    uk = u[:, keep]
    return {
        "absorbable": (uk @ (uk.T @ ew)) / np.sqrt(w),
        "orthogonal": e - (uk @ (uk.T @ ew)) / np.sqrt(w),
        "projection_rows": np.einsum("ij,ij->i", uk, uk),
    }


def projection_bias_examples(n=200_000, seed=6):
    """Analytic mean(E)=0 cases whose delta-alpha projection is nonzero."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    alpha = np.array([0.2, -0.1, 0.35])
    mean_zero_energy_dependent = p[:, 0] * (np.sum(p * p, axis=1) - 3.0)
    e_centered = mean_zero_energy_dependent - np.mean(mean_zero_energy_dependent)
    C = -p
    split_energy = absorbable_split(C, e_centered, weights=None)
    d_alpha = np.linalg.lstsq(C, split_energy["absorbable"], rcond=None)[0]
    simple = np.cos(p[:, 0] * 3.0)
    simple -= np.mean(simple)
    split_simple = absorbable_split(C, simple)
    d_alpha_simple = np.linalg.lstsq(C, split_simple["absorbable"], rcond=None)[0]
    return {
        "mean_energy_term": float(np.mean(e_centered)),
        "mean_simple_term": float(np.mean(simple)),
        "delta_alpha_energy": d_alpha,
        "delta_alpha_simple": d_alpha_simple,
        "projection_norm_energy": float(np.linalg.norm(split_energy["absorbable"])),
        "projection_norm_simple": float(np.linalg.norm(split_simple["absorbable"])),
    }
