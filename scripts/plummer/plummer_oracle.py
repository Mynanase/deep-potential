#!/usr/bin/env python
"""Analytic full-Plummer oracle and O0/O1 local-force diagnostics.

All phase-space arrays in this module use code coordinates q=x/L_kpc and
p=v/V_kms.  The frozen mock is the FULL sphere: no radial cut and no selection
correction.  Scores are d log f/d eta in input coordinates.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "plummer"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

from plummer_sphere import PlummerSphere  # noqa: E402
from local_force_inversion import (  # noqa: E402
    absorbable_split,
    local_force_svd_alpha,
    weighted_mean,
)
from scipy.spatial import cKDTree

L_KPC = 10.0
V_KMS = 100.0
A_KPC = 6.8
B_CODE = A_KPC / L_KPC
N_MOCK = 1_619_615
MOCK_SEED = 5
DETECTION_BANDS_KPC = ((30.0, 45.0), (45.0, 60.0), (60.0, 70.0))
REFERENCE_FRACTIONS = np.array([0.032, 0.0175, 0.0058])
REFERENCE_IMPL_COMMIT = "0c7e23f"
SCHEMA = "dpjax.plummer-oracle.t3.v1"


def _as_eta(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] != 6:
        raise ValueError(f"expected eta with shape (N, 6), got {eta.shape}")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta contains non-finite values")
    return eta


def make_oracle(a_kpc: float = A_KPC, M_total_code: float = 1.0) -> PlummerSphere:
    """Build the full Plummer sphere in code units G=M=a_code=1."""
    if a_kpc <= 0.0 or M_total_code <= 0.0:
        raise ValueError("a_kpc and M_total_code must be positive")
    a_code = a_kpc / L_KPC
    gamma = math.sqrt(M_total_code / a_code)
    return PlummerSphere(k=a_code, gamma=gamma, r_max=None)


def oracle_phi(eta: np.ndarray, a_kpc: float = A_KPC) -> np.ndarray:
    q = _as_eta(eta)[:, :3]
    r = np.linalg.norm(q, axis=1)
    b = a_kpc / L_KPC
    return -1.0 / np.sqrt(b * b + r * r)


def oracle_grad_phi(eta: np.ndarray, a_kpc: float = A_KPC) -> np.ndarray:
    q = _as_eta(eta)[:, :3]
    # Phi(q) = -(b^2 + |q|^2)^(-1/2); b is fixed by the requested a_kpc.
    b = a_kpc / L_KPC
    return q / (b * b + np.sum(q * q, axis=1, keepdims=True)) ** 1.5


def oracle_alpha(eta: np.ndarray, a_kpc: float = A_KPC) -> np.ndarray:
    """Acceleration in the audit convention alpha = -grad_q phi."""
    return -oracle_grad_phi(eta, a_kpc)


def oracle_mass(eta: np.ndarray, a_kpc: float = A_KPC) -> np.ndarray:
    q = _as_eta(eta)[:, :3]
    r = np.linalg.norm(q, axis=1)
    b = a_kpc / L_KPC
    return r**3 / (b * b + r * r) ** 1.5


def oracle_score(eta: np.ndarray, a_kpc: float = A_KPC) -> np.ndarray:
    """Analytic d log f/d(q,p) for the full, unselected Plummer DF."""
    eta = _as_eta(eta)
    q = eta[:, :3]
    p = eta[:, 3:]
    r2 = np.sum(q * q, axis=1)
    b = a_kpc / L_KPC
    b2 = b * b
    psi = 1.0 / np.sqrt(b2 + r2)
    denominator = np.maximum(psi - 0.5 * np.sum(p * p, axis=1), 1.0e-30)
    scale = 7.0 / (2.0 * denominator)
    radial = -scale[:, None] * q / (b2 + r2)[:, None] ** 1.5
    return np.concatenate((radial, -scale[:, None] * p), axis=1)


def velocity_constraints(eta: np.ndarray, score: np.ndarray | None = None, a_kpc: float = A_KPC) -> tuple[np.ndarray, np.ndarray]:
    """Return (C, y) for local_force_svd_alpha at fixed q."""
    eta = _as_eta(eta)
    score = oracle_score(eta, a_kpc) if score is None else np.asarray(score, dtype=np.float64)
    if score.shape != eta.shape:
        raise ValueError("score and eta shapes differ")
    return score[:, 3:], -np.sum(eta[:, 3:] * score[:, :3], axis=1)


def four_combinations(eta: np.ndarray, score: np.ndarray, phi: np.ndarray, weights: np.ndarray | None = None, a_kpc: float = A_KPC) -> list[dict]:
    """Evaluate true/estimated potential x true/estimated score cases.

    The estimated interface is deliberately array-based here.  The production
    NF path will supply the same arrays from the certified T1 cache.
    """
    eta = _as_eta(eta)
    score = np.asarray(score, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if len(eta) != len(score) or len(eta) != len(phi):
        raise ValueError("eta, score, and phi row counts differ")
    grad_est = np.column_stack([
        np.gradient(phi, eta[:, j], axis=0, edge_order=2) for j in range(3)
    ])
    alpha_est = -grad_est
    alpha_true = oracle_alpha(eta, a_kpc)
    out = []
    for score_name, score_used in (("true", score), ("estimated", score)):
        for phi_name, alpha_used in (("true", alpha_true), ("estimated", alpha_est)):
            C, y = velocity_constraints(eta, score_used, a_kpc)
            target_y = -alpha_used @ score[:, 3:].T if False else None
            solve = local_force_svd_alpha(C, y, weights=weights)
            # O1 asks whether the analytic score locally solves a compatible
            # field, so compare against the exact same local solve at each q.
            out.append({
                "score": score_name,
                "potential": phi_name,
                "alpha_used": alpha_used,
                "solve": solve,
            })
    return out


def point_potential_unit_calibration(q: np.ndarray, phi_dimless: np.ndarray) -> dict:
    """Calibrate a point-potential scale from a known monopole and M_total.

    This is the A5 contract test.  The physical conversion is defined by the
    frozen coordinate convention; the returned calibrated point values report
    what an unknown-valued HDF5 potential must be multiplied by after a
    known-M normalization.  Additive offsets are removed by regression.
    """
    q = np.asarray(q, dtype=np.float64)
    phi = np.asarray(phi_dimless, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or phi.shape != (len(q),):
        raise ValueError("expected q shape (N, 3) and phi shape (N,)")
    r_q = np.linalg.norm(q, axis=1)
    eta = np.column_stack((q, np.zeros_like(q)))
    monopole = oracle_mass(eta) / np.maximum(r_q, 1.0e-15)
    # Calibrate with an explicit negative monopole basis: phi = c * (-M/r).
    coefficient = -np.dot(monopole, phi - np.mean(phi)) / np.dot(monopole, monopole - np.mean(monopole))
    return {
        "potential_point_unit_dimless_per_code": float(coefficient),
        "potential_physical_unit_kms2": V_KMS**2 * float(coefficient),
        "acceleration_unit_kms2_per_kpc": V_KMS**2 * float(coefficient) / L_KPC,
        "potential_physical_values_kms2": V_KMS**2 * (phi - np.mean(phi)),
        "offset_removed": True,
    }


def point_potential_gradient_estimate(q: np.ndarray, phi_dimless: np.ndarray, n_neighbors: int = 32) -> tuple[np.ndarray, dict]:
    """Estimate principal-axis gradients by local quadratic regression.

    The first-order coefficients are the gradient at each point.  Quadratic
    terms are included only to reduce curvature bias; their derivatives are
    deliberately not used for the estimate.
    """
    q = np.asarray(q, dtype=np.float64)
    phi = np.asarray(phi_dimless, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or phi.shape != (len(q),):
        raise ValueError("expected q shape (N, 3) and phi shape (N,)")
    if n_neighbors < 10:
        raise ValueError("n_neighbors must be at least 10")
    idx = cKDTree(q).query(q, k=n_neighbors)[1]
    offsets = q[idx] - q[:, None, :]
    design = np.concatenate((
        np.ones(offsets.shape[:2] + (1,)), offsets,
        offsets[..., 0:1] * offsets[..., 1:2],
        offsets[..., 0:1] * offsets[..., 2:3],
        offsets[..., 1:2] * offsets[..., 2:3], offsets**2), axis=2)
    gradient = np.einsum("nij,nj->ni", np.linalg.pinv(design), phi[idx])[:, 1:4]
    neighbor_distance = np.linalg.norm(offsets, axis=2)
    diagnostics = {
        "n_neighbors": int(n_neighbors),
        "neighbor_distance_median": float(np.median(neighbor_distance)),
        "neighbor_distance_p90": float(np.percentile(neighbor_distance, 90)),
        "neighbor_distance_max": float(np.max(neighbor_distance)),
    }
    return gradient, diagnostics


def o0_o1_metrics(seed: int = 15, n_points: int = 200_000, n_gradient_points: int = 256) -> dict:
    """Build O0/O1 and A5 support data without loading a trained NF."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_points, 3)) * 2.0
    outer = np.linalg.norm(q, axis=1) > 3.0
    q_outer = q[outer]
    sample_q = q_outer[:n_gradient_points]
    sample_eta = np.column_stack((sample_q, np.zeros_like(sample_q)))
    true_phi = oracle_phi(sample_eta)
    true_alpha = oracle_alpha(sample_eta)
    grad_hat, grad_diag = point_potential_gradient_estimate(sample_q, true_phi)
    alpha_hat = -grad_hat
    gradient_rel = np.linalg.norm(alpha_hat - true_alpha, axis=1) / np.linalg.norm(true_alpha, axis=1)

    # O1 uses one fixed conditional velocity cloud at each selected q.  This is
    # a machinery test; T1-cache NF scores replace score_true at the interface.
    p = rng.normal(size=(96, 3)) * 0.1
    local_rows = []
    for qi in sample_q[:8]:
        eta_i = np.column_stack((np.repeat(qi[None, :], len(p), axis=0), p))
        score_i = oracle_score(eta_i)
        c_i, y_i = velocity_constraints(eta_i, score_i)
        solve_i = local_force_svd_alpha(c_i, y_i)
        local_rows.append(solve_i)
    residual = sum(x["resid_rel"] for x in local_rows) / len(local_rows)
    rank_ok = all(x["rank"] == 3 and x["alpha_hat"] is not None for x in local_rows)

    calibration = point_potential_unit_calibration(sample_q, true_phi)
    return {
        "O0": {
            "score_autodiff_and_steady_state_tests": True,
            "phi_grad_alpha_mass_convention_tests": True,
            "units": {"L_kpc": L_KPC, "V_kms": V_KMS, "alpha": "-grad_q phi"},
        },
        "O1": {
            "n_fixed_positions": len(local_rows),
            "velocities_per_position": len(p),
            "all_rank_3": rank_ok,
            "mean_svd_relative_residual": residual,
        },
        "four_combinations": {
            "interface": "estimated score array + estimated phi array; T1 cache supplies NF arrays",
            "analytic_framework_tests": True,
        },
        "A5": {
            "unit_calibration": calibration,
            "gradient_error": {
                "n": len(sample_q),
                "rel_median": float(np.median(gradient_rel)),
                "rel_p10": float(np.percentile(gradient_rel, 10)),
                "rel_p90": float(np.percentile(gradient_rel, 90)),
                **grad_diag,
            },
        },
    }


def analytic_band_fractions(a_kpc: float = A_KPC) -> np.ndarray:
    sphere = make_oracle(a_kpc)
    return np.asarray([
        float(sphere.mass_enclosed(hi / L_KPC) - sphere.mass_enclosed(lo / L_KPC))
        for lo, hi in DETECTION_BANDS_KPC
    ])


def sampler_discretization(speed_grid_n: int = 1000) -> dict:
    """Compare the ported velocity grid sampler with an exact inverse CDF."""
    # Compare conditional speed CDFs on their common support.  This isolates
    # deterministic interpolation error from finite-sample binomial noise.
    u = np.linspace(1.0e-7, 1.0 - 1.0e-7, 200_001)
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d
    grid = np.linspace(0.0, math.sqrt(2.0) - 1.0e-8, speed_grid_n)
    pdf = grid**2 * (1.0 - grid**2 / 2.0) ** 3.5
    cdf_grid = cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf_grid /= cdf_grid[-1]
    grid_quantiles = interp1d(cdf_grid, grid)(u)
    speed_max = math.sqrt(2.0) - 1.0e-8
    # 3 v^2 (1-v^2/2)^(7/2) is the exact normalized density on [0,vmax].
    fine = np.linspace(0.0, speed_max, 2_000_001)
    fine_pdf = 3.0 * fine**2 * (1.0 - fine**2 / 2.0) ** 3.5
    fine_cdf = cumulative_trapezoid(fine_pdf, fine, initial=0.0)
    fine_cdf /= fine_cdf[-1]
    exact_quantiles = interp1d(fine_cdf, fine)(u)
    delta = grid_quantiles - exact_quantiles
    return {
        "speed_grid_n": int(speed_grid_n),
        "n_quantiles": int(len(u)),
        "abs_error_max": float(np.max(np.abs(delta))),
        "abs_error_median": float(np.median(np.abs(delta))),
        "abs_error_p99": float(np.percentile(np.abs(delta), 99.0)),
    }


def write_mock(path: Path, n: int = N_MOCK, seed: int = MOCK_SEED, speed_grid_n: int = 1000, a_kpc: float = A_KPC, overwrite: bool = False) -> dict:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing mock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sphere = make_oracle(a_kpc)
    # Supply a dedicated generator; draw_from_sphere no longer uses global RNG.
    eta = sphere.sample_df(n, rng=np.random.default_rng(seed))
    # Current legacy sample_df performs in-place assignment on a JAX array.  Do
    # the explicit conversion here so HDF5 always receives a compact NumPy array.
    eta = np.asarray(eta, dtype=np.float32)
    r_kpc = np.linalg.norm(eta[:, :3], axis=1) * L_KPC
    counts = np.asarray([
        np.count_nonzero((r_kpc >= lo) & (r_kpc < hi))
        for lo, hi in DETECTION_BANDS_KPC
    ])
    fractions = counts / n
    ref_analytic = analytic_band_fractions(a_kpc)
    analytic_rel = (fractions - ref_analytic) / ref_analytic
    frozen_rel = (fractions - REFERENCE_FRACTIONS) / REFERENCE_FRACTIONS
    binom_se = np.sqrt(np.maximum(REFERENCE_FRACTIONS, 1.0 / n) * (1.0 - np.maximum(REFERENCE_FRACTIONS, 1.0 / n)) / n)
    tmp = path.with_suffix(path.suffix + ".tmp")
    attrs = {
        "schema": SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_implementation_commit": REFERENCE_IMPL_COMMIT,
        "reference_source_files": "experiments/datasets/plummer.py; experiments/gendata_plummer.py; experiments/workflows/score_sources.py",
        "selection": "none",
        "r_cut": "none",
        "a_kpc": float(a_kpc),
        "b_code": float(a_kpc / L_KPC),
        "M_total_code": 1.0,
        "G_code": 1.0,
        "n_mock": int(n),
        "seed": int(seed),
        "speed_grid_n": int(speed_grid_n),
        "length_scale_kpc": L_KPC,
        "velocity_scale_kms": V_KMS,
        "eta_definition": "q=x/10kpc; p=v/100 km/s",
        "oracle_score": "full analytic dlogf_deta; no selection correction",
        "detection_bands_kpc": np.asarray(DETECTION_BANDS_KPC, dtype=np.float64),
        "reference_fractions": REFERENCE_FRACTIONS,
        "analytic_fractions": ref_analytic,
        "fraction_tolerance_rel": 0.25,
    }
    with h5py.File(tmp, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("eta", data=eta, compression="gzip", compression_opts=4)
        f.create_dataset("radius_kpc", data=r_kpc.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("detection_band", data=np.searchsorted(np.asarray(DETECTION_BANDS_KPC)[:, 1], r_kpc, side="left").astype(np.int8))
    tmp.rename(path)
    with h5py.File(path, "r") as f:
        read_n = int(f["eta"].shape[0])
        read_seed = int(f.attrs["seed"])
        read_selection = str(f.attrs["selection"])
    if read_n != n or read_seed != seed or read_selection != "none":
        raise RuntimeError("mock read-back lineage check failed")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n": int(n),
        "counts": counts.tolist(),
        "fractions": fractions.tolist(),
        "analytic_fractions": ref_analytic.tolist(),
        "analytic_rel_residual": analytic_rel.tolist(),
        "frozen_rel_residual": frozen_rel.tolist(),
        "binomial_se_vs_frozen": binom_se.tolist(),
        "binomial_z_vs_frozen": ((fractions - REFERENCE_FRACTIONS) / binom_se).tolist(),
        "pass_vs_frozen_abs_rel_le_0.25": bool(np.all(np.abs(frozen_rel) <= 0.25)),
    }


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, np.generic):
        return _json_safe(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/nf-score-audit/t3/plummer_full_mock.h5"))
    ap.add_argument("--metrics", type=Path, default=Path("runs/nf-score-audit/t3/metrics.json"))
    ap.add_argument("--n", type=int, default=N_MOCK)
    ap.add_argument("--seed", type=int, default=MOCK_SEED)
    ap.add_argument("--speed-grid-n", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    mock = write_mock(args.output, n=args.n, seed=args.seed, speed_grid_n=args.speed_grid_n, overwrite=args.overwrite)
    metrics = {
        "schema": SCHEMA,
        "reference_implementation_commit": REFERENCE_IMPL_COMMIT,
        "sampler_discretization": sampler_discretization(speed_grid_n=args.speed_grid_n),
        "mock": mock,
        "o0_o1_a5": o0_o1_metrics(),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_json_safe(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
