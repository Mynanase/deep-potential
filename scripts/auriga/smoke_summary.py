#!/usr/bin/env python
"""Print the compact evidence block for the Halo12 smoke baseline run."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
from pathlib import Path

import h5py
import numpy as np


def final_metrics(path):
    with open(path) as f:
        history = json.load(f)
    loss_keys = sorted(k for k in history if k.startswith(("train", "val")))
    out = {"epochs": max((len(history[k]) for k in loss_keys if k.startswith("train")), default=0)}
    for key in loss_keys:
        if key in history and len(history[key]):
            out[key] = float(history[key][-1])
            out[key + "_min"] = float(np.min(history[key]))
    return out


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir

    print("=" * 72)
    print("AURIGA HALO12 SMOKE BASELINE SUMMARY")
    print("=" * 72)

    with h5py.File(args.input, "r") as f:
        attrs = dict(f.attrs)
        n_particles = f["eta"].shape[0]
    length_scale = float(attrs["length_scale_kpc"])
    velocity_scale = float(attrs["velocity_scale_kms"])
    print(f"input: {args.input}")
    print(f"  particles={n_particles}  weighting={attrs['weighting']}  shuffle_seed={attrs['shuffle_seed']}")
    print(f"  L={length_scale} kpc  V={velocity_scale} km/s  DF r<{attrs['train_r_max_kpc']} kpc"
          f"  Phi {attrs['phi_r_min_kpc']}-{attrs['phi_r_max_kpc']} kpc")

    print("losses (final epoch, min over epochs):")
    for subdir in ("models/df/flow", "models/Phi"):
        model_dir = run / subdir
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*_loss.json")):
            m = final_metrics(path)
            line = f"  {subdir}/{path.name}: epochs={m['epochs']}"
            for key in sorted(k for k in m
                              if k.startswith(("train", "val")) and not k.endswith("_min")):
                if key in m:
                    line += f"  {key}={m[key]:.6g} (min {m[key + '_min']:.6g})"
            print(line)

    grads = run / "data/df_gradients.h5"
    if grads.is_file():
        with h5py.File(grads, "r") as f:
            eta = f["eta"][:]
            score = f["df_deta"][:] if "df_deta" in f else None
        radius = np.linalg.norm(eta[:, :3], axis=1) * length_scale
        line = f"  n={len(eta)}  r=[{radius.min():.2f}, {radius.max():.2f}] kpc"
        if score is not None:
            line += f"  finite_score_fraction={np.isfinite(score).mean():.4f}"
        print("df samples:")
        print(line)

    npz = run / "plots/potential_axes.npz"
    if npz.is_file():
        data = np.load(npz)
        density = data["density_msun_per_kpc3"]
        print("axis profiles:")
        print(f"  r=[{data['r_kpc'].min():.2f}, {data['r_kpc'].max():.2f}] kpc"
              f"  negative_density_fraction={np.mean(density < 0):.3f}")

    print("outputs:")
    for path in sorted(run.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(run)}  {path.stat().st_size} bytes")
    print("=" * 72)


if __name__ == "__main__":
    main()
