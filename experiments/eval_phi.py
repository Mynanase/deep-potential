"""Evaluate only the trained Phi stage for one experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.run_eval import run_stage


def run(config_path: str | Path) -> None:
    """Write Phi evaluation artifacts declared by one run config."""
    run_stage(config_path, "phi")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the Phi stage declared by an experiment YAML config."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
