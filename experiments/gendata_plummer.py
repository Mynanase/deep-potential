"""Generate Plummer-sphere phase-space data with optional train/test split.

Usage
-----
    python -m experiments.gendata_plummer \
        --total-n 524288 --test-frac 0.1 --max-dist 10.0 \
        --train-out data/plummer_train.h5 \
        --test-out data/plummer_test.h5

This is a pure-CPU script (no JAX/GPU required).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.datasets.phase_space import save_eta_h5
from experiments.datasets.plummer import sample_plummer, split_train_test

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Plummer-sphere phase-space data (train + optional test split)."
    )
    parser.add_argument("--total-n", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Fraction of data for test set (0 = no split). Ignored if --test-n is set.")
    parser.add_argument("--test-n", type=int, default=None, help="Exact number of test samples (overrides --test-frac).")
    parser.add_argument("--max-dist", type=float, default=10.0, help="Maximum radial distance for sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--train-out", type=str, required=True, help="Output path for training data (.h5).")
    parser.add_argument("--test-out", type=str, default=None, help="Output path for test data (.h5). Required if --test-frac > 0.")
    args = parser.parse_args()

    if args.total_n <= 0:
        parser.error("--total-n must be positive")
    if args.max_dist <= 0.0:
        parser.error("--max-dist must be positive")
    if not 0.0 <= args.test_frac < 1.0:
        parser.error("--test-frac must satisfy 0 <= value < 1")
    if args.test_n is not None and not 0 < args.test_n < args.total_n:
        parser.error("--test-n must satisfy 0 < value < --total-n")

    has_split = args.test_n is not None or args.test_frac > 0
    if has_split and args.total_n < 2:
        parser.error("--total-n must be at least 2 when creating a test split")
    if has_split and args.test_out is None:
        parser.error("--test-out is required when --test-frac > 0 or --test-n is set")

    rng = np.random.default_rng(args.seed)
    print(
        f"Generating {args.total_n} Plummer samples "
        f"(max_dist={args.max_dist}, seed={args.seed}) ..."
    )
    eta_all = sample_plummer(
        args.total_n,
        max_dist=args.max_dist,
        rng=rng,
    )
    print(f"  Final dataset: {eta_all.shape[0]} samples, shape={eta_all.shape}")

    # Split
    if has_split:
        if args.test_n is not None:
            n_test = args.test_n
        else:
            n_test = round(eta_all.shape[0] * args.test_frac)
            n_test = min(max(n_test, 1), eta_all.shape[0] - 1)

        eta_train, eta_test = split_train_test(eta_all, test_n=n_test, rng=rng)
        train_out = Path(args.train_out)
        test_out = Path(args.test_out)

        save_eta_h5(eta_train, train_out)
        save_eta_h5(eta_test, test_out)

        print(f"  Train: {eta_train.shape[0]} samples -> {train_out}")
        print(f"  Test:  {eta_test.shape[0]} samples -> {test_out}")
    else:
        train_out = Path(args.train_out)
        save_eta_h5(eta_all, train_out)
        print(f"  All: {eta_all.shape[0]} samples -> {train_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
