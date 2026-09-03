"""Preflight and common-target evaluation for the Halo12 clump pair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from experiments.datasets.phase_space import phase_space_sha256
from experiments.workflows.config import RunSpec, load_run_spec
from experiments.workflows.evaluation.df import (
    DEFAULT_RADIAL_EDGES,
    evaluate_df_diagnostics,
)


DEFAULT_RAW_CONFIG = Path("configs/runs/halo12_raw_no_clip_v1.yaml")
DEFAULT_CLEAN_CONFIG = Path(
    "configs/runs/halo12_clean_outer_clump_v1.yaml"
)
DEFAULT_MASK = Path(
    "runs/halo12/outer-clump-removal/outer_clump_mask.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/halo12/clump-pair-comparison/results/data"
)


def validate_source_partition(
    raw_source_index: np.ndarray,
    clean_source_index: np.ndarray,
    removed_source_index: np.ndarray,
) -> np.ndarray:
    """Validate and return raw row positions retained by the clean dataset."""
    raw = np.asarray(raw_source_index, dtype=np.int64)
    clean = np.asarray(clean_source_index, dtype=np.int64)
    removed = np.asarray(removed_source_index, dtype=np.int64)
    for name, values in (
        ("raw_source_index", raw),
        ("clean_source_index", clean),
        ("removed_source_index", removed),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} contains duplicate values.")

    if np.intersect1d(clean, removed).size:
        raise ValueError("Clean and removed source indices overlap.")
    keep = ~np.isin(raw, removed, assume_unique=True)
    expected_clean = raw[keep]
    if not np.array_equal(clean, expected_clean):
        raise ValueError(
            "Clean source_index is not the ordered raw complement of the mask."
        )
    return np.flatnonzero(keep).astype(np.int64, copy=False)


def _resolved_pair_configs(
    raw_spec: RunSpec,
    clean_spec: RunSpec,
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for stage in ("df", "phi"):
        raw_config = raw_spec.resolve_stage_config(stage)
        clean_config = clean_spec.resolve_stage_config(stage)
        if raw_config != clean_config:
            raise ValueError(
                f"Raw and clean resolved {stage.upper()} configs differ."
            )
        resolved[stage] = raw_config
    clip_sigma = float(resolved["df"]["data"].get("clip_sigma", 0.0))
    if clip_sigma != 0.0:
        raise ValueError(
            "The causal pair must use clip_sigma=0 so the explicit clump mask "
            "is the only row-selection difference."
        )
    if raw_spec.output_dir == clean_spec.output_dir:
        raise ValueError("Raw and clean runs must use different output dirs.")
    if raw_spec.data_path == clean_spec.data_path:
        raise ValueError("Raw and clean runs must use different data paths.")
    return resolved


def _validate_aligned_hdf5(
    raw_path: Path,
    clean_path: Path,
    mask_path: Path,
    *,
    chunk_size: int = 131_072,
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    missing = [
        path for path in (raw_path, clean_path, mask_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Halo12 pair artifact(s): "
            + ", ".join(str(path) for path in missing)
        )

    with np.load(mask_path, allow_pickle=False) as artifact:
        mask_source_size = int(artifact["source_size"])
        mask_source_sha256 = str(artifact["source_sha256"])
        removed_source_index = np.asarray(
            artifact["removed_source_index"],
            dtype=np.int64,
        )

    with h5py.File(raw_path, "r") as raw, h5py.File(clean_path, "r") as clean:
        for handle, label in ((raw, "raw"), (clean, "clean")):
            for dataset in ("eta", "source_index", "tracer_weight"):
                if dataset not in handle:
                    raise KeyError(f"Missing {label} dataset {dataset!r}.")
        raw_source_index = np.asarray(raw["source_index"], dtype=np.int64)
        clean_source_index = np.asarray(clean["source_index"], dtype=np.int64)
        if mask_source_size != raw_source_index.size:
            raise ValueError(
                "Mask source_size does not match the raw canonical dataset."
            )
        keep_rows = validate_source_partition(
            raw_source_index,
            clean_source_index,
            removed_source_index,
        )

        raw_eta = np.asarray(raw["eta"], dtype=np.float32)
        if phase_space_sha256(raw_eta) != mask_source_sha256:
            raise ValueError("Mask source SHA-256 does not match raw eta.")
        for start in range(0, clean_source_index.size, chunk_size):
            stop = min(start + chunk_size, clean_source_index.size)
            clean_eta = np.asarray(clean["eta"][start:stop])
            expected_eta = raw_eta[keep_rows[start:stop]]
            if not np.array_equal(clean_eta, expected_eta):
                raise ValueError(
                    "Clean eta rows do not match raw eta through source_index."
                )

        weight_summary = {}
        for handle, label in ((raw, "raw"), (clean, "clean")):
            weight = np.asarray(handle["tracer_weight"], dtype=np.float64)
            if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
                raise ValueError(f"{label} tracer weights must be finite/positive.")
            weight_summary[label] = {
                "mean": float(weight.mean()),
                "min": float(weight.min()),
                "max": float(weight.max()),
            }

    return {
        "raw_rows": int(raw_source_index.size),
        "clean_rows": int(clean_source_index.size),
        "removed_rows": int(removed_source_index.size),
        "removed_fraction": float(
            removed_source_index.size / raw_source_index.size
        ),
        "mask_source_sha256": mask_source_sha256,
        "tracer_weight": weight_summary,
    }


def preflight_pair(
    raw_config: str | Path = DEFAULT_RAW_CONFIG,
    clean_config: str | Path = DEFAULT_CLEAN_CONFIG,
    mask_path: str | Path = DEFAULT_MASK,
) -> tuple[RunSpec, RunSpec, dict[str, Any]]:
    """Prove that data row identity is the only paired-run difference."""
    raw_spec = load_run_spec(raw_config)
    clean_spec = load_run_spec(clean_config)
    resolved = _resolved_pair_configs(raw_spec, clean_spec)
    data_summary = _validate_aligned_hdf5(
        raw_spec.data_path,
        clean_spec.data_path,
        Path(mask_path),
    )
    summary = {
        "raw_config": str(Path(raw_config)),
        "clean_config": str(Path(clean_config)),
        "raw_data": str(raw_spec.data_path),
        "clean_data": str(clean_spec.data_path),
        "raw_output_dir": str(raw_spec.output_dir),
        "clean_output_dir": str(clean_spec.output_dir),
        "resolved_df_equal": True,
        "resolved_phi_equal": True,
        "clip_sigma": float(resolved["df"]["data"]["clip_sigma"]),
        **data_summary,
    }
    return raw_spec, clean_spec, summary


def _evaluation_config(spec: RunSpec) -> Mapping[str, Any]:
    config = spec.evaluation.get("df", {})
    if not isinstance(config, Mapping):
        raise TypeError("evaluation.df must be a mapping.")
    return config


def evaluate_pair_on_clean_target(
    raw_spec: RunSpec,
    clean_spec: RunSpec,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate both trained DFs on identical clean data and score points."""
    for run_dir, label in (
        (raw_spec.df_dir, "raw DF"),
        (clean_spec.df_dir, "clean DF"),
    ):
        if not (run_dir / "ckpt").exists():
            raise FileNotFoundError(f"Missing trained {label}: {run_dir}")

    raw_config = dict(_evaluation_config(raw_spec))
    clean_config = dict(_evaluation_config(clean_spec))
    if raw_config != clean_config:
        raise ValueError("Raw and clean DF evaluation configs differ.")
    config = clean_config
    theta_edges = np.arccos(
        np.linspace(1.0, -1.0, int(config.get("n_theta_bins", 6)) + 1)
    )
    phi_edges = np.linspace(
        -np.pi,
        np.pi,
        int(config.get("n_phi_bins", 8)) + 1,
    )
    return evaluate_df_diagnostics(
        clean_spec.data_path,
        [raw_spec.df_dir, clean_spec.df_dir],
        output_dir,
        n_samples_per_model=int(config.get("n_samples_per_model", 262_144)),
        n_score_points=int(config.get("n_score_points", 32_768)),
        score_batch_size=int(config.get("score_batch_size", 1_024)),
        seed=int(config.get("seed", 42)),
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        n_velocity_bins=int(config.get("n_velocity_bins", 64)),
        radial_edges=np.asarray(
            config.get("radial_edges", DEFAULT_RADIAL_EDGES),
            dtype=np.float64,
        ),
        score_field_r_bins=int(config.get("score_field_r_bins", 32)),
        score_field_v_bins=int(config.get("score_field_v_bins", 32)),
        score_field_min_effective_count=float(
            config.get("score_field_min_effective_count", 20)
        ),
        radial_speed_r_bins=int(config.get("radial_speed_r_bins", 64)),
        radial_speed_v_bins=int(config.get("radial_speed_v_bins", 64)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Halo12 raw/clean causal pair and optionally evaluate "
            "both trained DFs on the same clean target rows."
        )
    )
    parser.add_argument("--raw-config", type=Path, default=DEFAULT_RAW_CONFIG)
    parser.add_argument(
        "--clean-config",
        type=Path,
        default=DEFAULT_CLEAN_CONFIG,
    )
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluate-df", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_spec, clean_spec, summary = preflight_pair(
        args.raw_config,
        args.clean_config,
        args.mask,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / "pair_preflight.json"
    preflight_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote paired preflight to {preflight_path}")
    if args.evaluate_df:
        evaluate_pair_on_clean_target(raw_spec, clean_spec, args.output_dir)
        print(f"Wrote common-target DF evaluation to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
