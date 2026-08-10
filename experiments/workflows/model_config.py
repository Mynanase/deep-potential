"""Load and validate reusable model-recipe YAML files.

Model files describe numerical architecture and optimization only. Dataset,
filesystem, logging, checkpoint cadence, GPU layout, evaluation, and plotting
belong to the run configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from experiments.paths import resolve_path

MODEL_SCHEMA = "dpjax.model.v1"
MODEL_KINDS = frozenset({"df", "phi"})
_FORBIDDEN_ROOTS = frozenset(
    {
        "data",
        "evaluation",
        "execution",
        "logging",
        "output_dir",
        "plots",
        "seed",
        "validation",
    }
)
_RUNTIME_TRAIN_KEYS = frozenset(
    {"ckpt_every", "log_every", "max_to_keep", "multi_gpu"}
)


def merge_config(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*."""
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_model_config(
    config: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Validate the boundary between a model recipe and run operations."""
    result = dict(config)
    if result.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"model config schema must be {MODEL_SCHEMA!r}.")

    kind = result.get("kind")
    if kind not in MODEL_KINDS:
        raise ValueError("model config kind must be 'df' or 'phi'.")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(
            f"Expected a {expected_kind!r} model config, got kind={kind!r}."
        )

    misplaced = sorted(_FORBIDDEN_ROOTS.intersection(result))
    if misplaced:
        raise ValueError(
            "Run-level field(s) found in model config: " + ", ".join(misplaced)
        )

    required_model_key = "flow" if kind == "df" else "potential"
    other_model_key = "potential" if kind == "df" else "flow"
    if not isinstance(result.get(required_model_key), Mapping):
        raise TypeError(
            f"{kind} model config must define a {required_model_key!r} mapping."
        )
    if other_model_key in result:
        raise ValueError(
            f"{kind} model config must not define {other_model_key!r}."
        )

    train = result.get("train")
    if not isinstance(train, Mapping):
        raise TypeError("model config must define a 'train' mapping.")
    misplaced_train = sorted(_RUNTIME_TRAIN_KEYS.intersection(train))
    if misplaced_train:
        raise ValueError(
            "Run-time train field(s) found in model config: "
            + ", ".join(misplaced_train)
        )

    optimizer = str(train.get("optimizer", "")).lower()
    if optimizer not in {"adam", "radam"}:
        raise ValueError("train.optimizer must be 'adam' or 'radam'.")
    return result


def load_model_config(
    path: str | Path,
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Load one model recipe without adding run-level values."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Model config not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("model config must be a mapping.")
    return validate_model_config(raw, expected_kind=expected_kind)
