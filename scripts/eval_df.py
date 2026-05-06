"""
Evaluate trained DF model on halo_12 data.
Loads checkpoint, computes samples, compares distributions.
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpjax.flows.api import build_flow, init_flow, sample_apply
from dpjax.data import load_eta_h5
import flax
import yaml


def load_checkpoint(ckpt_dir: str):
    params = flax.serialization.msgpack_restore(
        open(os.path.join(ckpt_dir, "params.msgpack"), "rb").read()
    )
    meta = yaml.safe_load(open(os.path.join(ckpt_dir, "meta.yaml")))
    return params, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run dir (e.g. runs/halo_12/df_realnvp)")
    parser.add_argument("--data", default="data/halo_12_train.h5")
    parser.add_argument("--key", default="eta")
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    data = load_eta_h5(args.data, dataset=args.key)
    data = np.asarray(data, dtype=np.float32)
    print(f"Data shape: {data.shape}")

    # Load model + params + normalizer + config + optional coord_transform via unified loader
    from dpjax.flows.api import load_df
    model, params, normalizer, cfg, coord_transform = load_df(args.run_dir)
    flow_cfg = cfg.get("flow", {})
    print(f"Flow type: {flow_cfg.get('type', 'realnvp')}")
    if coord_transform is not None:
        print(f"Coordinate transform: {coord_transform.type!r} on dims {coord_transform.dims.tolist()}")

    rng = jax.random.PRNGKey(42)

    # Normalize data (and optionally apply same forward transform if present)
    data_norm = normalizer.transform(data)
    if coord_transform is not None:
        data_norm = coord_transform.transform(data_norm)

    # Sample from model
    print(f"Sampling {args.n_samples} points...")
    rng, rng_sample = jax.random.split(rng)
    from dpjax.flows.api import sample_apply
    samples_norm = sample_apply(model, params, rng_sample, args.n_samples, flow_cfg)
    samples = normalizer.inverse(np.asarray(samples_norm))
    if coord_transform is not None:
        samples = coord_transform.inverse(samples)

    # Compare distributions
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.hist(data[:, i], bins=100, density=True, alpha=0.5, label="data")
        ax.hist(samples[:, i], bins=100, density=True, alpha=0.5, label="model")
        ax.set_xlabel(labels[i])
        ax.set_ylabel("density")
        ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "marginals.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved marginals to {out_path}")

    # Compare radial distribution
    r_data = np.linalg.norm(data[:, :3], axis=1)
    r_sample = np.linalg.norm(samples[:, :3], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(r_data.max(), r_sample.max()), 100)
    ax.hist(r_data, bins=bins, density=True, alpha=0.5, label="data")
    ax.hist(r_sample, bins=bins, density=True, alpha=0.5, label="model")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "radial.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved radial to {out_path}")

    # Compare velocity magnitude
    v_data = np.linalg.norm(data[:, 3:], axis=1)
    v_sample = np.linalg.norm(samples[:, 3:], axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(v_data.max(), v_sample.max()), 100)
    ax.hist(v_data, bins=bins, density=True, alpha=0.5, label="data")
    ax.hist(v_sample, bins=bins, density=True, alpha=0.5, label="model")
    ax.set_xlabel("|v| [km/s]")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "velocity.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved velocity to {out_path}")

    # Print statistics
    print("\n=== Data vs Model ===")
    for i, name in enumerate(labels):
        print(f"{name}: data_mean={data[:,i].mean():.3f}, model_mean={samples[:,i].mean():.3f}, "
              f"data_std={data[:,i].std():.3f}, model_std={samples[:,i].std():.3f}")
    print(f"r:  data_mean={r_data.mean():.3f}, model_mean={r_sample.mean():.3f}")
    print(f"|v|: data_mean={v_data.mean():.3f}, model_mean={v_sample.mean():.3f}")


if __name__ == "__main__":
    main()
