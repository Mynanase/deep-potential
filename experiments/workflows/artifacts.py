"""Load trained model artifacts from experiment run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from flax import linen as nn

from dpjax.flows.api import build_flow
from dpjax.models.potential import PotentialConfig, PotentialMLP
from dpjax.normalization import Normalizer
from experiments.datasets.phase_space import (
    CoordinateTransform,
    load_run_preprocessing,
)
from experiments.workflows.checkpoints import create_manager, restore_latest


def load_df(
    run_dir: str | Path,
) -> tuple[
    nn.Module,
    dict,
    Normalizer,
    dict[str, Any],
    CoordinateTransform | None,
]:
    """Restore a trained DF and its operational preprocessing artifacts."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = build_flow(config.get("flow", {}))
    normalizer, coordinate_transform = load_run_preprocessing(run_dir)
    restored = restore_latest(create_manager(run_dir / "ckpt"))
    return (
        model,
        restored["params"],
        normalizer,
        config,
        coordinate_transform,
    )


def load_phi(
    run_dir: str | Path,
) -> tuple[PotentialMLP, dict, dict[str, Any]]:
    """Restore a trained potential model from an experiment run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    potential_config = config.get("potential", {})
    model = PotentialMLP(
        PotentialConfig(
            hidden_sizes=tuple(
                int(width)
                for width in potential_config.get(
                    "hidden_sizes",
                    [512, 512, 512, 512],
                )
            ),
            output_scale=float(potential_config.get("output_scale", 1.0)),
        )
    )
    restored = restore_latest(create_manager(run_dir / "ckpt"))
    return model, restored["params"], config
