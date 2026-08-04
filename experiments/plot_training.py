from __future__ import annotations

import argparse
from pathlib import Path

from dpjax.diagnostics.training import load_metrics, plot_training_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training curves from metrics.csv.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(metrics_path)
    figures = plot_training_metrics(data, dpi=args.dpi)
    import matplotlib.pyplot as plt

    for name, figure in figures.items():
        figure.savefig(out_dir / f"{name}.png")
        plt.close(figure)

    print(f"Wrote training plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
