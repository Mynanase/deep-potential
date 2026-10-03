#!/usr/bin/env python
"""Train and qualify the project conditional NF on the frozen Plummer mock."""
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
import fit_all  # noqa: E402
import audit_df_constraints as adc  # noqa: E402
from flow_ot_flow_matching_conditional import ConditionalPhaseSpaceFlow  # noqa: E402
from plummer_oracle import (  # noqa: E402
    A_KPC, B_CODE, DETECTION_BANDS_KPC, L_KPC, MOCK_SEED, N_MOCK, V_KMS,
    analytic_band_fractions, oracle_score,
)

SCHEMA = "dpjax.nf-score-audit.t4a-conditional-nf.v1"
FROZEN_MOCK_SHA256 = "7e01175d5e3fc85d7ceddcd8e825dd2298018a31096414174b7d406e073e503d"
UNITS = {
    "eta": "dimensionless [q, p]", "q": "x / 10 kpc", "p": "v / 100 km/s",
    "score": "grad_eta log f(q,p)", "log_prob": "log f(q,p) in input eta coordinates",
    "flow_factorization": "log f(q,p)=log n(q)+log P(p|q)",
    "selection": "none", "r_cut": "none",
}
MODEL_CODE_FILES = [
    "scripts/fit_all.py", "scripts/flow_matching_conditional.py",
    "scripts/flow_ot_flow_matching_conditional.py", "scripts/flow_vector_fields.py",
    "scripts/utils.py", "scripts/auriga/t4a_plummer_nf.py",
]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def verify_mock(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys()); eta = np.asarray(handle["eta"], dtype=np.float32); attrs = dict(handle.attrs)
    if not {"eta", "radius_kpc", "detection_band"}.issubset(keys): raise RuntimeError("frozen mock datasets are incomplete")
    expected = {"schema": "dpjax.plummer-oracle.t3.v1", "reference_implementation_commit": "0c7e23f",
        "selection": "none", "r_cut": "none", "a_kpc": A_KPC, "b_code": B_CODE,
        "n_mock": N_MOCK, "seed": MOCK_SEED, "length_scale_kpc": L_KPC, "velocity_scale_kms": V_KMS}
    for key, value in expected.items():
        actual = attrs.get(key); matches = str(actual) == value if isinstance(value, str) else np.asarray(actual) == value
        if not bool(np.all(matches)): raise RuntimeError(f"mock lineage mismatch: {key}={actual!r}, expected {value!r}")
    file_hash = sha256_file(path)
    if file_hash != FROZEN_MOCK_SHA256: raise RuntimeError(f"mock SHA256 mismatch: {file_hash} != {FROZEN_MOCK_SHA256}")
    if eta.shape != (N_MOCK, 6) or not np.all(np.isfinite(eta)): raise RuntimeError("mock eta shape/finite gate failed")
    radius_kpc = np.linalg.norm(eta[:, :3], axis=1) * L_KPC
    counts = np.asarray([np.count_nonzero((radius_kpc >= lo) & (radius_kpc < hi)) for lo, hi in DETECTION_BANDS_KPC])
    fractions = counts / N_MOCK; analytic = analytic_band_fractions(A_KPC); analytic_rel = (fractions - analytic) / analytic
    if not np.all(np.abs(analytic_rel) <= 0.006): raise RuntimeError(f"mock density gate failed: {analytic_rel.tolist()}")
    return {"path": str(path), "sha256": file_hash, "eta_sha256": hashlib.sha256(np.ascontiguousarray(eta).tobytes()).hexdigest(),
        "n": N_MOCK, "selection": "none", "r_cut": "none", "detection_band_counts": counts.tolist(),
        "detection_band_fractions": fractions.tolist(), "analytic_detection_band_fractions": analytic.tolist(),
        "relative_residual_vs_analytic": analytic_rel.tolist()}

def load_training_options() -> dict:
    options = json.loads((REPO / "scripts/auriga/options.json").read_text())["df"]
    expected = {"training_method": "FlowMatching", "seed": 0, "validation_frac": 0.25,
        "time_scheduler_type": "uniform", "checkpoint_frequency_epochs": 25}
    for key, value in expected.items():
        if options.get(key) != value: raise RuntimeError(f"refusing non-control DF option {key}={options.get(key)!r}")
    for key in ("spatial_flow_opts", "conditional_velocity_flow_opts"):
        if options[key]["n_epochs"] != 256 or options[key]["batch_size"] != 4096: raise RuntimeError(f"refusing non-control epochs/batch size in {key}")
        if options[key]["vector_field_opts"] != {"type": "MLP", "width": 1024, "depth": 3}: raise RuntimeError(f"refusing non-w1024 vector field in {key}")
    return {key: value for key, value in options.items() if key != "benchmarking_r0"}

def train_conditional_nf(mock_path: Path, flow_dir: Path) -> dict:
    if (flow_dir / "metadata.json").exists() and max(flow_dir.glob("flow-[0-9]*_loss.json"), key=lambda p: p.stat().st_mtime).exists():
        model, history = fit_all.load_flow(flow_dir, checkpoint_index=-1, load_history=True)
        return {"model": model, "history": history, "history_path": None,
                "elapsed_s": 0.0, "options": load_training_options(), "reused_existing_training": True}
    with h5py.File(mock_path, "r") as handle: eta = np.asarray(handle["eta"], dtype=np.float32)
    data = {"eta": eta, "weights": np.ones(len(eta), dtype=np.float32)}; options = load_training_options(); started = time.time()
    model, history = fit_all.train_flow_conditional(data, str(flow_dir), **options)
    history_path = max(flow_dir.glob("flow-*_loss.json"), key=lambda p: p.stat().st_mtime)
    return {"model": model, "history": history, "history_path": history_path, "elapsed_s": time.time() - started, "options": options}

def checkpoint_index(path: Path) -> int:
    match = re.search(r"-(\d+)_model.eqx$", path.name)
    if match is None: raise RuntimeError(f"cannot parse checkpoint index from {path}")
    return int(match.group(1))

def load_and_bind_model(flow_dir: Path):
    model, history = fit_all.load_flow(flow_dir, checkpoint_index=-1, load_history=True)
    spatial_path = max(flow_dir.glob("flow_pos_only-*_model.eqx"), key=checkpoint_index)
    combined_path = max(flow_dir.glob("flow-[0-9]*_model.eqx"), key=checkpoint_index)
    spatial_ref, spatial_history = ConditionalPhaseSpaceFlow.load(flow_dir, load_index=checkpoint_index(spatial_path), load_prefix="flow_pos_only", load_history=True)
    integrity = adc.flow_pair_integrity(model, spatial_ref)
    if not integrity["bitwise_identical"]: raise RuntimeError(f"flow-pair integrity failed: {integrity}")
    return model, history, {"spatial": spatial_path, "combined": combined_path, "spatial_history": spatial_history, "integrity": integrity}

def fixed_probe_rows(eta): return adc.stratified_rows_by_radius(eta, 4096)

def evaluate_flow(flow, rows, strict=False, batch=256):
    source = adc._with_ode_tolerance(flow, 1e-7, 1e-8) if strict else flow
    with adc._x64(strict):
        fn = adc._grad_lnf_fn(source); lnf, score = adc.eval_batched(fn, jnp.asarray(rows), batch=batch)
    return lnf, score

def finite_difference_scores(flow_strict, rows, step=1e-4):
    output = np.empty((len(rows), 6), dtype=np.float64); scalar = adc._lnf_fn_scalar(flow_strict)
    with adc._x64(True):
        for i, row in enumerate(rows):
            output[i] = adc.central_fd_grad(lambda x: float(np.asarray(scalar(jnp.asarray(x)))), np.asarray(row, dtype=np.float64), step)
    return output

def relative_error(estimate, reference, floor): return np.abs(estimate - reference) / np.maximum(np.abs(reference), floor)

def analytic_ln_prob(eta):
    normalization = np.log(24.0 * np.sqrt(2.0) / (7.0 * np.pi ** 3)) - 3.0 * np.log(B_CODE) - np.log(np.sqrt(1.0 / B_CODE))
    relative_energy = 1.0 / np.sqrt(B_CODE ** 2 + np.sum(eta[:, :3] ** 2, axis=1)) - 0.5 * np.sum(eta[:, 3:] ** 2, axis=1)
    return normalization + 3.5 * np.log(np.maximum(relative_energy, 1e-30))

def compact_history(history):
    return {key: {"first": values[0], "middle": values[len(values)//2], "final": values[-1], "n": len(values)} for key, values in history.items() if isinstance(values, list) and values}

def qualify(mock_info, flow_dir: Path, cache_dir: Path) -> dict:
    flow, history, checkpoints = load_and_bind_model(flow_dir)
    with h5py.File(mock_info["path"], "r") as handle: eta = np.asarray(handle["eta"], dtype=np.float32)
    rows_idx = fixed_probe_rows(eta); rows = eta[rows_idx].astype(np.float64)
    lnf32, score32 = evaluate_flow(flow, rows, strict=False); lnf64, score64 = evaluate_flow(flow, rows, strict=True)
    flow_strict = adc._with_ode_tolerance(flow, 1e-7, 1e-8); fd = finite_difference_scores(flow_strict, rows[:8])
    vg_pos, vg_vel = adc._split_fns(flow)
    with adc._x64(False):
        ln_pos, grad_pos = adc.eval_batched(vg_pos, jnp.asarray(rows[:, :3], dtype=jnp.float32), batch=256)
        ln_vel, grad_vel = adc.eval_batched(vg_vel, jnp.asarray(rows, dtype=jnp.float32), batch=256)
    split_score = adc.combine_split_grads(grad_pos, grad_vel)
    split_identity = relative_error(split_score, score32, np.percentile(np.abs(score32), 50, axis=0))
    split_log_identity = np.abs(ln_pos + ln_vel - lnf32); analytic_score = oracle_score(rows); analytic_log_prob = analytic_ln_prob(rows)
    score_floor = np.percentile(np.abs(analytic_score), 50, axis=0)
    precision_rel = relative_error(score32, score64, np.percentile(np.abs(score64), 50, axis=0))
    fit_rel = relative_error(score64, analytic_score, score_floor); fd_rel = relative_error(fd, score64[:8], score_floor)
    log_prob_error = np.abs(lnf64 - analytic_log_prob); point_hash = adc.point_order_hash("t4a_fixed_probes", rows_idx, rows)
    cache_dir.mkdir(parents=True, exist_ok=True); points_path = cache_dir / "points_fixed_probes.h5"; arrays_path = cache_dir / "arrays_fixed_probes.h5"
    write_h5(points_path, {"source_row": rows_idx, "eta": rows}, {"schema": SCHEMA, "point_set": "fixed_probes", "point_order_sha256": point_hash, "units": json.dumps(UNITS), "mock_sha256": mock_info["sha256"]})
    write_h5(arrays_path, {"eta": rows, "lnf_f32": lnf32, "score_f32": score32, "lnf_x64strict": lnf64,
        "score_x64strict": score64, "analytic_ln_prob": analytic_log_prob, "analytic_score": analytic_score},
        {"schema": SCHEMA, "point_set": "fixed_probes", "point_order_sha256": point_hash,
         "precision": "f32 default ODE and f64 strict ODE", "ode": json.dumps({"default": {"solver": "Tsit5", "rtol": 1e-4, "atol": 1e-5}, "strict": {"solver": "Tsit5", "rtol": 1e-7, "atol": 1e-8}}), "units": json.dumps(UNITS)})
    checkpoint_hashes = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in checkpoints.items() if isinstance(path, Path)}
    code_hashes = {rel: sha256_file(REPO / rel) for rel in MODEL_CODE_FILES}
    gates = {"mock_lineage": True, "full_data": True, "density": True,
        "finite": bool(np.all(np.isfinite(score32)) and np.all(np.isfinite(score64))), "flow_pair_bitwise": bool(checkpoints["integrity"]["bitwise_identical"]),
        "precision_score_median_rel": float(np.median(precision_rel)), "precision_score_p99_rel": float(np.percentile(precision_rel, 99)),
        "fd_score_median_rel": float(np.median(fd_rel)), "fd_score_p99_rel": float(np.percentile(fd_rel, 99)),
        "split_score_median_rel": float(np.median(split_identity)), "split_score_max_abs": float(np.max(np.abs(split_score - score32))),
        "split_log_prob_max_abs": float(np.max(split_log_identity)), "score_fit_median_rel": float(np.median(fit_rel)),
        "score_fit_p99_rel": float(np.percentile(fit_rel, 99)), "log_prob_fit_median_abs": float(np.median(log_prob_error)),
        "log_prob_fit_p99_abs": float(np.percentile(log_prob_error, 99))}
    manifest = {"schema": SCHEMA, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "mock": mock_info,
        "model": {"kind": "project ConditionalPhaseSpaceFlow (spatial + conditional velocity)", "factorization": "log n(q) + log P(p|q)",
                  "training_options": load_training_options(), "checkpoint_sha256": checkpoint_hashes,
                  "model_code_sha256": code_hashes, "flow_pair_integrity": checkpoints["integrity"], "final_history": compact_history(history)},
        "points": {"path": str(points_path), "sha256": sha256_file(points_path), "point_order_sha256": point_hash},
        "arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path), "point_order_sha256": point_hash},
        "numerics": {"ode": {"default": {"solver": "Tsit5", "rtol": 1e-4, "atol": 1e-5}, "strict": {"solver": "Tsit5", "rtol": 1e-7, "atol": 1e-8}}, "trace": "exact production path; no Hutchinson estimator"},
        "gates": gates, "units": UNITS}
    gates["all_ok"] = bool(all(gates[key] for key in ("mock_lineage", "full_data", "density", "finite", "flow_pair_bitwise"))
        and gates["precision_score_median_rel"] < 0.01 and gates["precision_score_p99_rel"] < 0.05
        and gates["fd_score_median_rel"] < 0.05 and gates["fd_score_p99_rel"] < 0.20
        and gates["split_score_median_rel"] < 1e-6 and gates["split_score_max_abs"] < 1e-4
        and gates["split_log_prob_max_abs"] < 1e-4 and gates["score_fit_median_rel"] < 0.25 and gates["score_fit_p99_rel"] < 1.0)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest

def write_h5(path: Path, arrays: dict, attrs: dict) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        for key, value in attrs.items(): handle.attrs[key] = value
        for key, value in arrays.items(): handle.create_dataset(key, data=value, compression="gzip", compression_opts=4)
    tmp.rename(path); return sha256_file(path)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(); started = time.time(); mock_info = verify_mock(args.mock)
    training = train_conditional_nf(args.mock, args.run_dir); manifest = qualify(mock_info, args.run_dir, args.cache_dir)
    manifest["training_elapsed_s"] = training["elapsed_s"]; manifest["elapsed_s"] = time.time() - started
    (args.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("[t4a.summary] " + json.dumps({"mock_sha256": mock_info["sha256"], "checkpoints": manifest["model"]["checkpoint_sha256"], "gates": manifest["gates"], "elapsed_s": manifest["elapsed_s"]}, indent=2))
    return 0 if manifest["gates"]["all_ok"] else 2

if __name__ == "__main__": raise SystemExit(main())
