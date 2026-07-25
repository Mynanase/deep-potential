"""Shared argument and configuration helpers for experiment entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dpjax.config import load_config, merge_config


def add_config_override_argument(
    parser: argparse.ArgumentParser,
    *,
    example: str,
) -> None:
    parser.add_argument(
        "--override",
        default=None,
        help=f"JSON object merged into the YAML config, e.g. {example!r}.",
    )


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logger",
        default="csv",
        help="Logger backend: csv, wandb, tensorboard, wandb+tb.",
    )
    parser.add_argument(
        "--project",
        default="dp-plummer",
        help="W&B project name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B / experiment run name.",
    )


def load_experiment_config(
    path: str | Path,
    override_json: str | None = None,
) -> dict[str, Any]:
    """Load YAML and recursively apply an optional JSON object override."""
    config = load_config(path)
    if override_json is None:
        return config

    overrides = json.loads(override_json)
    if not isinstance(overrides, dict):
        raise ValueError("--override must decode to a JSON object.")
    return merge_config(config, overrides)
