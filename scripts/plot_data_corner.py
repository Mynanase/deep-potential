#!/usr/bin/env python
"""Generate a 6x6 corner plot of the training data (x, y, z, vx, vy, vz)."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def load_data(path: str | Path, dataset: str = "eta") -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset], dtype=np.float64)


def corner_plot(
    data: np.ndarray,
    labels: list[str],
    out_path: str | Path,
    *,
    bins_1d: int = 80,
    bins_2d: int = 120,
    figsize: float = 14.0,
    dpi: int = 150,
) -> None:
    ndim = data.shape[1]
    fig, axes = plt.subplots(
        ndim, ndim, figsize=(figsize, figsize), dpi=dpi,
    )

    # Precompute ranges (1-99 percentile with 20% padding)
    ranges = []
    for i in range(ndim):
        lo, hi = np.percentile(data[:, i], [0.5, 99.5])
        pad = 0.15 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]

            if j > i:
                # Upper triangle: hide
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(
                    data[:, i],
                    bins=bins_1d,
                    range=ranges[i],
                    density=True,
                    color="steelblue",
                    alpha=0.85,
                    edgecolor="none",
                )
                mu = np.mean(data[:, i])
                sigma = np.std(data[:, i])
                ax.axvline(mu, color="crimson", lw=1, ls="--", alpha=0.8)
                ax.set_title(
                    f"$\\mu={mu:.2f},\\ \\sigma={sigma:.2f}$",
                    fontsize=8,
                    pad=3,
                )
                ax.set_yticks([])
            else:
                # Lower triangle: 2D histogram (density)
                ax.hist2d(
                    data[:, j],
                    data[:, i],
                    bins=bins_2d,
                    range=[ranges[j], ranges[i]],
                    norm=LogNorm(),
                    cmap="inferno",
                    rasterized=True,
                )

            # Axis limits
            ax.set_xlim(ranges[j])
            if i != j:
                ax.set_ylim(ranges[i])

            # Labels
            if i == ndim - 1:
                ax.set_xlabel(labels[j], fontsize=11)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=11)
            elif j != 0:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Training data corner plot  (n = {len(data):,})",
        fontsize=14,
        y=0.92,
    )
    fig.subplots_adjust(
        hspace=0.06, wspace=0.06,
        left=0.06, right=0.97, bottom=0.06, top=0.90,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved corner plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Corner plot of training data")
    parser.add_argument("--data", type=str, default="data/halo_12_train.h5")
    parser.add_argument("--dataset", type=str, default="eta")
    parser.add_argument("--out", type=str, default="runs/halo_12/data_corner.png")
    parser.add_argument(
        "--n-samples", type=int, default=50000,
        help="Subsample size for plotting (0 = use all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_data(args.data, args.dataset)
    print(f"Loaded {data.shape[0]:,} rows, {data.shape[1]} dims from {args.data}")

    if args.n_samples > 0 and args.n_samples < data.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(data.shape[0], size=args.n_samples, replace=False)
        data = data[idx]
        print(f"Subsampled to {len(data):,} rows")

    labels = [
        r"$x$ [kpc]", r"$y$ [kpc]", r"$z$ [kpc]",
        r"$v_x$ [km/s]", r"$v_y$ [km/s]", r"$v_z$ [km/s]",
    ]

    corner_plot(data, labels, args.out)


if __name__ == "__main__":
    main()
