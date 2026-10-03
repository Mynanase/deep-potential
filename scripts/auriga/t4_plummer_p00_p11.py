#!/usr/bin/env python
"""Minimal paired Plummer Phi experiment: analytic P00 versus NF P11 scores."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "plummer"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import audit_df_constraints as adc  # noqa: E402
import fit_all  # noqa: E402
from plummer_oracle import (  # noqa: E402
    A_KPC, B_CODE, DETECTION_BANDS_KPC, L_KPC, MOCK_SEED, N_MOCK, V_KMS,
    analytic_band_fractions, oracle_alpha, oracle_mass, oracle_phi, oracle_score,
)

SCHEMA = "dpjax.nf-score-audit.t4-p00-p11.v1"
FROZEN_MOCK_SHA256 = "7e01175d5e3fc85d7ceddcd8e825dd2298018a31096414174b7d406e073e503d"
N_CONSTRAINTS = 262145
PHI_EPOCHS = 256
PHI_BATCH_SIZE = 1024
Q_EDGES = np.array([0.1, 0.2, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0])
QUOTAS = np.array([31501, 104858, 45885, 18350, 13107, 37350, 11094], dtype=np.int64)
REPORT_BANDS_KPC = ((2.0, 10.0), (10.0, 30.0), (30.0, 50.0), (50.0, 70.0))
EVAL_RADII_KPC = np.concatenate((np.arange(2.0, 30.0, 3.0), np.arange(30.0, 70.1, 5.0)))
N_EVAL_DIRS = 2048
UNITS = {"L_kpc": L_KPC, "V_kms": V_KMS, "eta": "dimensionless [q,p]",
         "score": "grad_eta log f", "alpha": "-grad_q phi",
         "density": "laplacian * V^2 / (4 pi G L^2)"}
G_CODE = 4.30091e-6 * 1.0e12 / (V_KMS ** 2 / (L_KPC * 3.0856775814913673e17)) / 1000.0
PHI_OPTIONS = {
    "seed": 2, "potential_nn_opts": {"type": "MLP", "width": 1024, "depth": 3},
    "frameshift_opts": {"omega": 0, "r0": 0, "v0_x": 0, "v0_y": 0, "v0_z": 0,
                        "omega_trainable": False, "r0_trainable": False,
                        "v0_x_trainable": False, "v0_y_trainable": False,
                        "v0_z_trainable": False},
    "selection_function_opts": None, "n_epochs_noselfn": PHI_EPOCHS,
    "n_epochs_selfn": 0, "batch_size": PHI_BATCH_SIZE, "validation_frac": 0.25,
    "checkpoint_frequency_epochs": 25,
    "lr_opts": {"type": "warmup_cosine_decay", "init": 0.001, "final": 1e-5,
                "warmup_epochs": 1, "global_norm_clip": 1},
    "loss_opts": {"alpha": 1, "beta": 1, "lambda_": 1, "gamma": 0, "mu": 0,
                  "l2_potential": 0.01, "prior_grid_n": 4096,
                  "prior_grid_q_max": 7.0, "prior_grid_weighting": "radius"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_mock(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        eta = np.asarray(handle["eta"], dtype=np.float32)
        attrs = dict(handle.attrs)
    expected = {"schema": "dpjax.plummer-oracle.t3.v1", "selection": "none",
                "r_cut": "none", "a_kpc": A_KPC, "b_code": B_CODE,
                "n_mock": N_MOCK, "seed": MOCK_SEED,
                "length_scale_kpc": L_KPC, "velocity_scale_kms": V_KMS}
    for key, value in expected.items():
        actual = attrs.get(key)
        matches = str(actual) == value if isinstance(value, str) else np.asarray(actual) == value
        if not bool(np.all(matches)):
            raise RuntimeError(f"mock lineage mismatch {key}: {actual!r} != {value!r}")
    if sha256_file(path) != FROZEN_MOCK_SHA256 or eta.shape != (N_MOCK, 6) or not np.all(np.isfinite(eta)):
        raise RuntimeError("mock hash/shape/finite gate failed")
    radius_kpc = np.linalg.norm(eta[:, :3], axis=1) * L_KPC
    counts = np.asarray([np.count_nonzero((radius_kpc >= lo) & (radius_kpc < hi)) for lo, hi in DETECTION_BANDS_KPC])
    rel = (counts / N_MOCK - analytic_band_fractions(A_KPC)) / analytic_band_fractions(A_KPC)
    if not np.all(np.abs(rel) <= 0.006):
        raise RuntimeError(f"detection density gate failed: {rel.tolist()}")
    return {"path": str(path), "sha256": FROZEN_MOCK_SHA256,
            "eta_sha256": hashlib.sha256(np.ascontiguousarray(eta).tobytes()).hexdigest(),
            "n": N_MOCK, "selection": "none", "r_cut": "none",
            "detection_band_counts": counts.tolist(),
            "detection_band_relative_residual_vs_analytic": rel.tolist()}


def stratified_constraints(eta):
    radius_q = np.linalg.norm(eta[:, :3], axis=1)
    order = np.argsort(radius_q, kind="stable")
    rows = []
    for lo, hi, quota in zip(Q_EDGES[:-1], Q_EDGES[1:], QUOTAS):
        band = order[(radius_q[order] >= lo) & (radius_q[order] < hi)]
        if len(band) < quota:
            raise RuntimeError(f"constraint quota unavailable for q=[{lo},{hi})")
        rows.append(band[np.linspace(0, len(band) - 1, quota).round().astype(int)])
    rows = np.concatenate(rows)
    if len(rows) != N_CONSTRAINTS:
        raise RuntimeError("constraint row count mismatch")
    return rows, eta[rows].astype(np.float32)


def oracle_grad_phi(eta):
    q = np.asarray(eta[:, :3], dtype=np.float64)
    return q / (B_CODE ** 2 + np.sum(q * q, axis=1, keepdims=True)) ** 1.5


def oracle_laplacian(eta):
    q = np.asarray(eta[:, :3], dtype=np.float64)
    return 3.0 * B_CODE ** 2 / (B_CODE ** 2 + np.sum(q * q, axis=1)) ** 2.5


def load_flow(flow_dir: Path):
    flow, _ = fit_all.load_flow(flow_dir, checkpoint_index=21, load_history=False)
    spatial_ref, _ = fit_all.ConditionalPhaseSpaceFlow.load(
        flow_dir, load_index=10, load_prefix="flow_pos_only", load_history=False)
    integrity = adc.flow_pair_integrity(flow, spatial_ref)
    if not integrity["bitwise_identical"]:
        raise RuntimeError(f"flow-pair integrity failed: {integrity}")
    return flow, integrity


def nf_scores(flow, eta):
    with adc._x64(False):
        fn = adc._grad_lnf_fn(flow)
        lnf, score = adc.eval_batched(fn, jnp.asarray(eta, dtype=jnp.float32), batch=256)
    return lnf.astype(np.float32), score.astype(np.float32)


def write_constraints(path: Path, eta, analytic_score, nf_score, flow_info, mock_info):
    order_hash = point_order_hash("t4_paired_constraints", np.arange(len(eta), dtype=np.int64), eta)
    with h5py.File(path.with_suffix(".tmp"), "w") as handle:
        for key, value in {"schema": SCHEMA, "point_order_sha256": order_hash,
                           "selection": "none", "r_cut": "none",
                           "units": json.dumps(UNITS), "mock_sha256": mock_info["sha256"],
                           "flow": json.dumps(flow_info)}.items():
            handle.attrs[key] = value
        handle.create_dataset("source_row", data=np.arange(len(eta), dtype=np.int64), compression="gzip")
        handle.create_dataset("eta", data=eta, compression="gzip")
        handle.create_dataset("score_p00", data=analytic_score.astype(np.float32), compression="gzip")
        handle.create_dataset("score_p11", data=nf_score, compression="gzip")
        handle.create_dataset("score_delta_p11_minus_p00", data=(nf_score - analytic_score).astype(np.float32), compression="gzip")
    path.with_suffix(".tmp").rename(path)
    return {"path": str(path), "sha256": sha256_file(path), "point_order_sha256": order_hash}


def point_order_hash(name, ids, eta):
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(np.asarray(ids, dtype=np.int64).tobytes())
    digest.update(np.asarray(eta, dtype=np.float64).tobytes())
    return digest.hexdigest()


def train_arm(arm, constraint_path, out_dir):
    with h5py.File(constraint_path, "r") as handle:
        eta = np.asarray(handle["eta"], dtype=np.float32)
        score = np.asarray(handle[f"score_{arm}"], dtype=np.float32)
    n_val = int(0.25 * len(eta))
    val_rows, train_rows = np.arange(n_val, dtype=np.int64), np.arange(n_val, len(eta), dtype=np.int64)
    df_data = {"eta": eta, "dlnf_deta": score, "dlnp_deta": score,
               "importance_weights": np.ones(len(eta), dtype=np.float32)}
    manifest = {"arm": arm, "rows": len(eta), "train_rows": len(train_rows),
                "val_rows": len(val_rows), "split_rule": "first int(0.25*n) rows validation",
                "phi_options": PHI_OPTIONS, "units": UNITS}
    model, history = fit_all.train_potential(df_data, out_dir, **PHI_OPTIONS)
    checkpoint = latest_phi_checkpoint(out_dir)
    manifest.update({"checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
                     "history": compact_history(history)})
    return model, manifest


def latest_phi_checkpoint(out_dir):
    return max(out_dir.glob("potential-[0-9]*_model.eqx"), key=lambda p: int(re.search(r"-(\d+)_", p.name).group(1)))


def compact_history(history):
    return {key: {"first": values[0], "middle": values[len(values)//2], "final": values[-1], "n": len(values)}
            for key, values in history.items() if isinstance(values, list) and values}


def sphere_grid():
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, scramble=True, seed=17)
    uv = sampler.random(N_EVAL_DIRS)
    mu = 2.0 * uv[:, 0] - 1.0
    azimuth = 2.0 * np.pi * uv[:, 1]
    sine = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
    dirs = np.column_stack((sine * np.cos(azimuth), sine * np.sin(azimuth), mu))
    q = (EVAL_RADII_KPC[:, None] / L_KPC) * dirs[None, :]
    return dirs, q.reshape(-1, 3)


def phi_products(phi_model, q):
    from potential import calc_phi_derivatives, calc_phi_laplacian
    vf = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))
    lf = jax.vmap(calc_phi_laplacian, in_axes=(None, 0))
    with adc._x64(True):
        phi, grad, lap = [], [], []
        for start in range(0, len(q), 2048):
            chunk = jnp.asarray(q[start:start+2048], dtype=jnp.float64)
            g, l = vf(phi_model, chunk)
            grad.append(np.asarray(g)); lap.append(np.asarray(l))
            phi.append(np.asarray(jax.vmap(phi_model)(chunk)))
    return np.concatenate(phi), np.concatenate(grad), np.concatenate(lap)


def evaluate_arm(arm, model, out_dir, constraint_path):
    dirs, q_grid = sphere_grid()
    phi, grad, lap = phi_products(model.phi_model, q_grid)
    eta_grid = np.concatenate((q_grid, np.zeros_like(q_grid)), axis=1)
    true_phi, true_grad = oracle_phi(eta_grid), oracle_grad_phi(eta_grid)
    true_alpha, true_mass = oracle_alpha(eta_grid), oracle_mass(eta_grid)
    alpha = -grad
    force_rel = np.linalg.norm(alpha - true_alpha, axis=1) / np.linalg.norm(true_alpha, axis=1)
    phi_abs = np.abs(phi - true_phi)
    radius = np.repeat(EVAL_RADII_KPC, N_EVAL_DIRS)
    true_lap = oracle_laplacian(eta_grid)
    lap_abs = np.abs(lap - true_lap)
    metrics = {"n_grid": int(len(q_grid)), "n_dirs": N_EVAL_DIRS,
               "force_rel_median": float(np.median(force_rel)),
               "force_rel_p99": float(np.percentile(force_rel, 99)),
               "phi_abs_median": float(np.median(phi_abs)),
               "phi_abs_p99": float(np.percentile(phi_abs, 99)),
               "laplacian_abs_median": float(np.median(lap_abs)),
               "laplacian_abs_p99": float(np.percentile(lap_abs, 99)),
               "negative_lap_fraction": float(np.mean(lap < 0)),
               "negative_lap_abs_mean": float(np.mean(np.maximum(-lap, 0)))}
    band_rows = []
    for lo, hi in REPORT_BANDS_KPC:
        mask = (radius >= lo) & (radius < hi)
        band_rows.append({"band_kpc": [lo, hi], "n": int(mask.sum()),
                          "force_rel_median": float(np.median(force_rel[mask])),
                          "force_rel_p99": float(np.percentile(force_rel[mask], 99)),
                          "negative_lap_fraction": float(np.mean(lap[mask] < 0)),
                          "negative_lap_abs_mean": float(np.mean(np.maximum(-lap[mask], 0)))})
    metrics["bands"] = band_rows
    arrays_path = out_dir / "evaluation_arrays.h5"
    with h5py.File(arrays_path.with_suffix(".tmp"), "w") as handle:
        for key, value in {"schema": SCHEMA, "arm": arm, "units": json.dumps(UNITS)}.items(): handle.attrs[key] = value
        for key, value in {"q": q_grid, "directions": dirs, "radius_kpc": radius,
                           "phi": phi, "grad_phi": grad, "acceleration": alpha,
                           "laplacian": lap, "true_phi": true_phi,
                           "true_grad_phi": true_grad, "true_alpha": true_alpha,
                           "true_mass": true_mass, "force_rel": force_rel}.items():
            handle.create_dataset(key, data=value, compression="gzip")
    arrays_path.with_suffix(".tmp").rename(arrays_path)
    return metrics, arrays_path


def density_from_lap(lap):
    return lap * V_KMS ** 2 / (4.0 * np.pi * G_CODE * L_KPC ** 2)


def paired_metrics(p00, p11):
    return {"force_rel_median_delta_p11_minus_p00": p11["force_rel_median"] - p00["force_rel_median"],
            "force_rel_p99_delta_p11_minus_p00": p11["force_rel_p99"] - p00["force_rel_p99"],
            "negative_lap_fraction_delta_p11_minus_p00": p11["negative_lap_fraction"] - p00["negative_lap_fraction"],
            "negative_lap_abs_mean_ratio_p11_over_p00": p11["negative_lap_abs_mean"] / max(p00["negative_lap_abs_mean"], 1e-30),
            "bands": [{"band_kpc": b0["band_kpc"],
                       "force_rel_median_delta": b1["force_rel_median"] - b0["force_rel_median"],
                       "force_rel_p99_delta": b1["force_rel_p99"] - b0["force_rel_p99"],
                       "negative_lap_fraction_delta": b1["negative_lap_fraction"] - b0["negative_lap_fraction"]}
                      for b0, b1 in zip(p00["bands"], p11["bands"])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time(); args.out_dir.mkdir(parents=True, exist_ok=True)
    mock_info = verify_mock(args.mock)
    source_rows, eta = stratified_constraints(load_mock_eta(args.mock))
    flow, integrity = load_flow(args.flow_dir)
    flow_info = {"checkpoint_index": 21, "spatial_checkpoint_index": 10, "integrity": integrity,
                 "flow_sha256": sha256_file(args.flow_dir / "flow-21_model.eqx"),
                 "spatial_sha256": sha256_file(args.flow_dir / "flow_pos_only-10_model.eqx")}
    nf_score = nf_scores(flow, eta)[1]
    analytic_score = oracle_score(eta.astype(np.float64)).astype(np.float32)
    constraints_path = args.out_dir / "paired_constraints.h5"
    constraint_info = write_constraints(constraints_path, eta, analytic_score, nf_score, flow_info, mock_info)
    manifest = {"schema": SCHEMA, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "mock": mock_info, "flow": flow_info, "constraints": constraint_info,
                "constraint_design": {"source_rows": source_rows.tolist(), "q_edges": Q_EDGES.tolist(), "quotas": QUOTAS.tolist()},
                "arms": {}, "units": UNITS}
    for arm in ("P00", "P11"):
        model, arm_manifest = train_arm(arm, constraints_path, args.out_dir / arm)
        metrics, arrays = evaluate_arm(arm, model, args.out_dir / arm)
        arm_manifest["evaluation"] = metrics; arm_manifest["evaluation_arrays"] = {"path": str(arrays), "sha256": sha256_file(arrays)}
        manifest["arms"][arm] = arm_manifest
        print(f"[t4.{arm}] " + json.dumps(metrics, sort_keys=True), flush=True)
    manifest["paired"] = paired_metrics(manifest["arms"]["P00"]["evaluation"], manifest["arms"]["P11"]["evaluation"])
    manifest["elapsed_s"] = time.time() - started
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("[t4.summary] " + json.dumps({"P00": manifest["arms"]["P00"]["evaluation"],
                                         "P11": manifest["arms"]["P11"]["evaluation"],
                                         "delta": manifest["paired"]}, indent=2), flush=True)
    return 0


def load_mock_eta(path):
    with h5py.File(path, "r") as handle: return np.asarray(handle["eta"], dtype=np.float32)


if __name__ == "__main__": raise SystemExit(main())
