#!/usr/bin/env python
"""Train and qualify a full-data Plummer mock FFJORD for the score audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO = Path(__file__).resolve()
while REPO.name and not (REPO / "scripts" / "auriga").is_dir() and REPO.parent != REPO:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "plummer"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

from dpjax.flows.api import build_flow, init_flow, log_prob_apply, score_apply  # noqa: E402
from dpjax.flows.ffjord import MockFFJORD, MockFFJORDConfig  # noqa: E402
from dpjax.normalization import Normalizer, fit_normalizer  # noqa: E402
from experiments.workflows.optimizers import build_optimizer  # noqa: E402
from experiments.workflows.tree import count_parameters  # noqa: E402
from plummer_oracle import (  # noqa: E402
    A_KPC, DETECTION_BANDS_KPC, L_KPC, MOCK_SEED, N_MOCK, V_KMS,
    B_CODE, analytic_band_fractions, make_oracle, oracle_score, write_mock,
)

SCHEMA = "dpjax.nf-score-audit.t4a-mock-nf.v1"
FLOW_CONFIG = {
    "type": "ffjord",
    "dim": 6,
    "ffjord": {
        "hidden_sizes": [256, 256, 256],
        "n_blocks": 3,
        "solver": "tsit5",
        "rtol": 1e-4,
        "atol": 1e-5,
        "trace": "exact",
    },
}
FLOW_SEED = 4
SPLIT_SEED = 6
DETECTION_TOLERANCE = 0.25
UNITS = {
    "eta": "dimensionless [q, p]",
    "q": "x / 10 kpc",
    "p": "v / 100 km/s",
    "score": "grad_eta log f",
    "normalizer": "affine training standardization; scores converted to input coordinates",
}


def _log_prob_single(self, row):
    from dpjax.flows.ffjord import log_prob_single
    return log_prob_single(self, row)


def _strict_flow(model, params):
    """Return a model clone evaluated with strict ODE tolerances."""
    from dataclasses import replace
    strict_model = MockFFJORD(replace(model.cfg, rtol=1e-7, atol=1e-8))
    with jax.enable_x64():
        strict_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
    return strict_model, strict_params


def sha256_file(path: Path) -> str:
    hh = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hh.update(block)
    return hh.hexdigest()


def verify_mock(path: Path) -> dict:
    """Verify immutable full-data mock lineage and density gates."""
    with h5py.File(path, "r") as handle:
        required = {"eta", "radius_kpc", "detection_band"}
        if not required.issubset(handle.keys()):
            raise RuntimeError(f"mock datasets missing: {sorted(required - set(handle.keys()))}")
        eta = np.asarray(handle["eta"], dtype=np.float32)
        attrs = dict(handle.attrs)
    expected = {
        "schema": "dpjax.plummer-oracle.t3.v1",
        "reference_implementation_commit": "0c7e23f",
        "selection": "none",
        "r_cut": "none",
        "a_kpc": A_KPC,
        "n_mock": N_MOCK,
        "seed": MOCK_SEED,
        "length_scale_kpc": L_KPC,
        "velocity_scale_kms": V_KMS,
    }
    for key, value in expected.items():
        actual = attrs.get(key)
        if isinstance(value, str):
            matches = str(actual) == value
        else:
            matches = np.asarray(actual) == value
        if not bool(np.all(matches)):
            raise RuntimeError(f"mock lineage mismatch for {key}: {actual!r} != {value!r}")
    if eta.shape != (N_MOCK, 6) or not np.all(np.isfinite(eta)):
        raise RuntimeError("mock eta shape/finite check failed")
    radius_kpc = np.linalg.norm(eta[:, :3], axis=1) * L_KPC
    fractions = []
    counts = []
    for lo, hi in DETECTION_BANDS_KPC:
        count = int(np.count_nonzero((radius_kpc >= lo) & (radius_kpc < hi)))
        counts.append(count)
        fractions.append(count / N_MOCK)
    analytic = analytic_band_fractions(A_KPC)
    rel_analytic = (np.asarray(fractions) - analytic) / analytic
    if not np.all(np.abs(rel_analytic) <= 0.006):
        raise RuntimeError(f"mock-vs-analytic density gate failed: {rel_analytic.tolist()}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "eta_sha256": hashlib.sha256(np.ascontiguousarray(eta).tobytes()).hexdigest(),
        "n": int(len(eta)),
        "selection": "none",
        "r_cut": "none",
        "detection_band_counts": counts,
        "detection_band_fractions": fractions,
        "analytic_detection_band_fractions": analytic.tolist(),
        "relative_residual_vs_analytic": rel_analytic.tolist(),
    }


def train(eta, normalizer, run_dir: Path, epochs: int, batch_size: int,
          max_batches_per_epoch: int | None, seed: int = FLOW_SEED):
    """Train the full population with a deterministic random split."""
    model = build_flow(FLOW_CONFIG)
    params = init_flow(model, jax.random.key(seed), FLOW_CONFIG)
    eta_std = normalizer.transform(eta)
    n_val = int(len(eta) * 0.1)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(eta))
    val_rows, train_rows = order[:n_val], order[n_val:]
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        build_optimizer("radam", optax.warmup_cosine_decay_schedule(
            0.0, 3e-3, max(1, int(0.1 * epochs * (max_batches_per_epoch or 1))),
            max(2, epochs * (max_batches_per_epoch or 1)), 1e-4)),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def eval_batch(p, rows):
        return jnp.mean(-log_prob_apply(model, p, rows))

    @jax.jit
    def train_step(p, state, rows_std, rows_weight):
        def loss(q):
            log_prob = log_prob_apply(model, q, rows_std)
            return jnp.mean(-log_prob)
        loss_value, grads = jax.value_and_grad(loss)(p)
        updates, state2 = optimizer.update(grads, state, p)
        p2 = optax.apply_updates(p, updates)
        return p2, state2, loss_value

    metrics_path = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="") as handle:
        handle.write("step,epoch,loss,val_loss\n")
        rng = np.random.default_rng(seed)
        step = 0
        for epoch in range(epochs):
            shuffled = train_rows[rng.permutation(len(train_rows))]
            batch_starts = range(0, len(shuffled) - batch_size + 1, batch_size)
            if max_batches_per_epoch is not None:
                batch_starts = list(batch_starts)[:max_batches_per_epoch]
            for start in batch_starts:
                rows = eta_std[shuffled[start:start + batch_size]]
                params, opt_state, loss_value = train_step(params, opt_state, rows, None)
                if step % 50 == 0:
                    val_sample = eta_std[val_rows[:2048]]
                    val_loss = float(jax.device_get(eval_batch(params, val_sample)))
                    loss_scalar = float(jax.device_get(loss_value))
                    handle.write(f"{step},{epoch},{loss_scalar:.9g},{val_loss:.9g}\n")
                    handle.flush()
        print(f"[t4a.train] step={step} epoch={epoch} loss={loss_scalar:.6g} val_loss={val_loss:.6g}", flush=True)
                step += 1
    checkpoint = run_dir / "final.mpck"
    save(checkpoint, {"params": params, "step": step, "schema": SCHEMA})
    return {
        "model": model,
        "params": params,
        "normalizer": normalizer,
        "checkpoint": checkpoint,
        "parameter_count": count_parameters(params),
        "final_step": step,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "split_seed": SPLIT_SEED,
        "flow_seed": seed,
    }


def finite_difference_score(model, params, rows, step=1e-3):
    """Central FD check of the cached score in input eta coordinates."""
    rows = np.asarray(rows, dtype=np.float64)
    out = np.empty_like(rows)
    for i, row in enumerate(rows):
        for component in range(6):
            plus, minus = row.copy(), row.copy()
            plus[component] += step
            minus[component] -= step
            lp_plus = float(jax.device_get(log_prob_apply(model, params, jnp.asarray(plus[None], dtype=params_dtype(params)))[0]))
            lp_minus = float(jax.device_get(log_prob_apply(model, params, jnp.asarray(minus[None], dtype=params_dtype(params)))[0]))
            out[i, component] = (lp_plus - lp_minus) / (2 * step)
    return out


def params_dtype(params):
    leaves = jax.tree_util.tree_leaves(params)
    return leaves[0].dtype if leaves else jnp.float32


def analytic_log_prob(eta):
    import math
    sphere = make_oracle()
    unit_log_norm = math.log(float(sphere.unit_plummer.df_norm))
    log_rescale = math.log(float(sphere.df_rescale))
    q, p = eta[:, :3], eta[:, 3:]
    relative_energy = 1.0 / np.sqrt(B_CODE**2 + np.sum(q * q, axis=1)) - 0.5 * np.sum(p * p, axis=1)
    return unit_log_norm + log_rescale + 3.5 * np.log(np.maximum(relative_energy, 1e-30))


def qualification(model, params, normalizer, eta, mock_info, training_info, cache_dir: Path):
    """Evaluate fixed points and write a T1-compatible hash-bound cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = select_fixed_probes(eta)
    rows = eta[probe]
    rows_std = normalizer.transform(rows)
    log_prob_f32, score_std_f32 = evaluate(model, params, rows_std, np.float32)
    with jax.enable_x64():
        strict_model, strict_params = _strict_flow(model, params)
        log_prob_f64, score_std_f64 = evaluate(strict_model, strict_params, rows_std, np.float64)
    analytic = oracle_score(rows.astype(np.float64))
    score_f32 = score_std_f32 / normalizer.std[None, :]
    score_f64 = score_std_f64 / normalizer.std[None, :]
    precision_scale = np.percentile(np.abs(score_f64), 50, axis=0)
    precision_rel = np.abs(score_f32 - score_f64) / np.maximum(np.abs(score_f64), precision_scale)
    fit_scale = np.percentile(np.abs(analytic), 50, axis=0)
    fit_rel = np.abs(score_f64 - analytic) / np.maximum(np.abs(analytic), fit_scale)
    analytic_lp = analytic_log_prob(rows.astype(np.float64))
    log_prob_abs_error = np.abs(log_prob_f64 - analytic_lp)
    fd = finite_difference_score(strict_model, strict_params, rows[:8], step=1e-4)
    fd_rel = np.abs(fd - score_f64[:8]) / np.maximum(np.abs(score_f64[:8]), fit_scale)

    point_hash = hashlib.sha256()
    point_hash.update(b"t4a_fixed_probes")
    point_hash.update(np.asarray(probe, dtype=np.int64).tobytes())
    point_hash.update(np.asarray(rows, dtype=np.float64).tobytes())
    point_hash = point_hash.hexdigest()
    points_path = cache_dir / "points_fixed_probes.h5"
    write_h5(points_path, {"source_row": probe, "eta": rows}, {
        "schema": SCHEMA, "point_order_sha256": point_hash, "units": json.dumps(UNITS)})
    arrays_path = cache_dir / "arrays_fixed_probes.h5"
    write_h5(arrays_path, {
        "eta": rows, "lnf_f32": log_prob_f32, "score_f32": score_f32,
        "lnf_x64strict": log_prob_f64, "score_x64strict": score_f64,
        "analytic_score": analytic,
        "analytic_log_prob": analytic_lp,
    }, {
        "schema": SCHEMA, "point_order_sha256": point_hash,
        "precision": "f32 default ODE and f64 strict ODE",
        "ode": json.dumps({"default": {"solver": "Tsit5", "rtol": 1e-4, "atol": 1e-5}, "strict": {"solver": "Tsit5", "rtol": 1e-7, "atol": 1e-8}}),
        "units": json.dumps(UNITS),
    })
    checkpoint = training_info["checkpoint"]
    manifest = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mock": mock_info,
        "flow": {
            "architecture": FLOW_CONFIG,
            "seed": FLOW_SEED,
            "split_seed": SPLIT_SEED,
            "train_rows": training_info["train_rows"],
            "val_rows": training_info["val_rows"],
            "parameter_count": training_info["parameter_count"],
            "final_step": training_info["final_step"],
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "normalizer_mean": normalizer.mean.tolist(),
            "normalizer_std": normalizer.std.tolist(),
        },
        "points": {"path": str(points_path), "sha256": sha256_file(points_path), "point_order_sha256": point_hash},
        "arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path), "point_order_sha256": point_hash},
        "gates": {
            "mock_lineage": True,
            "full_data": bool(mock_info["n"] == N_MOCK and mock_info["selection"] == "none"),
            "density": True,
            "finite": bool(np.all(np.isfinite(score_f32)) and np.all(np.isfinite(score_f64))),
            "precision_score_median_rel": float(np.median(precision_rel)),
            "precision_score_p99_rel": float(np.percentile(precision_rel, 99)),
            "fd_score_median_rel": float(np.median(fd_rel)),
            "fd_score_p99_rel": float(np.percentile(fd_rel, 99)),
            "score_fit_median_rel": float(np.median(fit_rel)),
            "score_fit_p99_rel": float(np.percentile(fit_rel, 99)),
            "log_prob_fit_median_abs": float(np.median(log_prob_abs_error)),
            "log_prob_fit_p99_abs": float(np.percentile(log_prob_abs_error, 99)),
        },
        "units": UNITS,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def select_fixed_probes(eta):
    radius = np.linalg.norm(eta[:, :3], axis=1)
    order = np.argsort(radius, kind="stable")
    return order[np.linspace(0, len(order) - 1, 4096).round().astype(int)]


def evaluate(model, params, rows_std, dtype):
    rows = jnp.asarray(rows_std, dtype=dtype)
    log_prob, score = [], []
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        lp = log_prob_apply(model, params, batch)
        def single(row):
            return log_prob_apply(model, params, row[None])[0]
        score.append(jax.vmap(jax.grad(single))(batch))
        log_prob.append(lp)
    return (np.concatenate([np.asarray(x) for x in log_prob]),
            np.concatenate([np.asarray(x) for x in score]))


def write_h5(path: Path, arrays: dict, attrs: dict) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        for key, value in arrays.items():
            handle.create_dataset(key, data=value, compression="gzip", compression_opts=4)
    tmp.rename(path)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    args = parser.parse_args()
    started = time.time()
    if not args.mock.exists():
        write_mock(args.mock)
    mock_info = verify_mock(args.mock)
    with h5py.File(args.mock, "r") as handle:
        eta = np.asarray(handle["eta"], dtype=np.float32)
    normalizer = fit_normalizer(eta)
    training = train(eta, normalizer, args.run_dir, args.epochs, args.batch_size, args.max_batches_per_epoch)
    manifest = qualification(training["model"], training["params"], normalizer, eta, mock_info, training, args.cache_dir)
    gates = manifest["gates"]
    gates["all_ok"] = bool(
        gates["mock_lineage"] and gates["full_data"] and gates["density"]
        and gates["finite"] and gates["precision_score_median_rel"] < 0.05
        and gates["precision_score_p99_rel"] < 0.20 and gates["fd_score_median_rel"] < 0.05
        and gates["fd_score_p99_rel"] < 0.20 and gates["score_fit_median_rel"] < 0.25
        and gates["score_fit_p99_rel"] < 1.0
    )
    manifest["elapsed_s"] = time.time() - started
    (args.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("[t4a.summary] " + json.dumps({
        "mock_sha256": mock_info["sha256"], "checkpoint_sha256": manifest["flow"]["checkpoint"]["sha256"],
        "final_step": training["final_step"], "gates": gates}, indent=2))
    if not gates["all_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
