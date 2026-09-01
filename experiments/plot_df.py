"""Render the complete DF training and evaluation figure set."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.run_plot import run_stage


def run(config_path: str | Path) -> dict[str, tuple[Path, ...]]:
    """Write every required DF figure for one experiment."""
    return run_stage(config_path, "df")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write DF training and evaluation figures from saved artifacts."
    )
    parser.add_argument("config", help="Path to configs/runs/<name>.yaml")
    args = parser.parse_args()
    outputs = run(args.config)
    for name, paths in outputs.items():
        print(f"{name}: {', '.join(str(path) for path in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
