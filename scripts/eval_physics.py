"""Evaluate v21: density profile rho(r) and other physical diagnostics.

Compares data vs model:
  - rho(r): spherically averaged density profile
  - rho(r) in log scale
  - velocity dispersion profiles sigma_v(r)
  - anisotropy beta(r) = 1 - sigma_t^2 / (2 * sigma_r^2)
"""
import argparse
import os

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dpjax.flows.api import sample_apply
from experiments.datasets.phase_space import inverse_preprocess_eta, load_eta_h5
from experiments.workflows.artifacts import load_df


def spherical_coords(pos, vel):
    """Convert to spherical coordinates for radial profiles."""
    r = np.linalg.norm(pos, axis=1)
    # Radial velocity: v_r = (v . r_hat)
    r_hat = pos / r[:, None]
    v_r = np.sum(vel * r_hat, axis=1)
    # Tangential velocity squared: v_t^2 = |v|^2 - v_r^2
    v_t2 = np.sum(vel**2, axis=1) - v_r**2
    return r, v_r, v_t2


def radial_profile(r, values, r_bins):
    """Compute mean of values in radial bins."""
    idx = np.digitize(r, r_bins) - 1
    valid = (idx >= 0) & (idx < len(r_bins) - 1)
    counts = np.bincount(idx[valid], minlength=len(r_bins)-1)[:len(r_bins)-1]
    sums = np.bincount(idx[valid], weights=values[valid], minlength=len(r_bins)-1)[:len(r_bins)-1]
    means = np.where(counts > 50, sums / counts, np.nan)
    return means, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="data/halo_12_train.h5")
    parser.add_argument("--n-samples", type=int, default=500000)
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "eval_physics")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, params, normalizer, cfg, coord_transform = load_df(args.run_dir)
    flow_cfg = cfg.get("flow", {})
    print(f"Flow: {flow_cfg.get('type')}, dim={flow_cfg.get('dim')}")
    if coord_transform is not None:
        print(f"Coord transform: {coord_transform.type!r} on dims {coord_transform.dims.tolist()}")

    # Load raw data
    raw_data = load_eta_h5(args.data)
    print(f"Raw data shape: {raw_data.shape}")

    # Sample from model
    rng = jax.random.PRNGKey(42)
    rng, rng_s = jax.random.split(rng)
    print(f"Sampling {args.n_samples} points...")
    samples_norm = sample_apply(model, params, rng_s, args.n_samples, flow_cfg)
    samples_norm = np.asarray(samples_norm)

    # Inverse transform: normalizer → coord_transform
    samples = inverse_preprocess_eta(
        samples_norm,
        normalizer,
        coord_transform,
    )

    pos_data = raw_data[:, :3]
    vel_data = raw_data[:, 3:]
    pos_model = samples[:, :3]
    vel_model = samples[:, 3:]

    r_data, vr_data, vt2_data = spherical_coords(pos_data, vel_data)
    r_model, vr_model, vt2_model = spherical_coords(pos_model, vel_model)

    print(f"\nData: N={len(r_data)}, r_range=[{r_data.min():.1f}, {r_data.max():.1f}] kpc")
    print(f"Model: N={len(r_model)}, r_range=[{r_model.min():.1f}, {r_model.max():.1f}] kpc")

    # ---- Radial bins ----
    r_bins = np.linspace(0, 70, 71)  # 1 kpc bins
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    # ---- Density profile rho(r) ----
    # Shell volume: 4/3 * pi * (r2^3 - r1^3)
    shell_vol = (4.0/3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    data_counts, _ = np.histogram(r_data, bins=r_bins)
    model_counts, _ = np.histogram(r_model, bins=r_bins)
    # Normalize by total N and shell volume
    rho_data = data_counts / (len(r_data) * shell_vol)
    rho_model = model_counts / (len(r_model) * shell_vol)

    # ---- Velocity dispersion profiles ----
    sigma_vr_data, _ = radial_profile(r_data, vr_data**2, r_bins)
    sigma_vr_model, _ = radial_profile(r_model, vr_model**2, r_bins)
    sigma_vt_data, _ = radial_profile(r_data, vt2_data, r_bins)
    sigma_vt_model, _ = radial_profile(r_model, vt2_model, r_bins)

    # Take sqrt for dispersions
    sigma_vr_data = np.sqrt(np.abs(sigma_vr_data))
    sigma_vr_model = np.sqrt(np.abs(sigma_vr_model))
    sigma_vt_data = np.sqrt(np.abs(sigma_vt_data))
    sigma_vt_model = np.sqrt(np.abs(sigma_vt_model))

    # Total velocity dispersion
    sigma_v_data = np.sqrt(sigma_vr_data**2 + sigma_vt_data**2)
    sigma_v_model = np.sqrt(np.abs(sigma_vr_model**2 + sigma_vt_model**2))

    # ---- Anisotropy profile beta(r) = 1 - sigma_t^2 / (2 sigma_r^2) ----
    beta_data = 1.0 - sigma_vt_data**2 / (2.0 * np.maximum(sigma_vr_data**2, 1e-10))
    beta_model = 1.0 - sigma_vt_model**2 / (2.0 * np.maximum(sigma_vr_model**2, 1e-10))

    # ---- Print summary ----
    print("\n" + "="*70)
    print("Physical profile comparison")
    print("="*70)
    print(f"{'r_mid':>6}  {'rho_data':>12}  {'rho_model':>12}  {'ratio':>8}  {'sigma_v_data':>12}  {'sigma_v_model':>12}")
    for i in range(0, len(r_mid), 5):
        if not np.isnan(rho_data[i]) and not np.isnan(rho_model[i]) and rho_data[i] > 0:
            ratio = rho_model[i] / rho_data[i]
            print(f"{r_mid[i]:6.1f}  {rho_data[i]:12.4e}  {rho_model[i]:12.4e}  {ratio:8.2f}  {sigma_v_data[i]:12.1f}  {sigma_v_model[i]:12.1f}")

    # ---- Plot: density profile ----
    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    valid = rho_data > 0
    axes[0].plot(r_mid[valid], rho_data[valid], 'b-', lw=2, label='data')
    axes[0].plot(r_mid[valid & (rho_model > 0)], rho_model[valid & (rho_model > 0)], 'r--', lw=2, label='model')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\rho(r)$ [arb. units]')
    axes[0].set_title('Density profile')
    axes[0].legend()

    # Log scale
    axes[1].plot(r_mid[valid], rho_data[valid], 'b-', lw=2, label='data')
    axes[1].plot(r_mid[valid & (rho_model > 0)], rho_model[valid & (rho_model > 0)], 'r--', lw=2, label='model')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'$\rho(r)$ [arb. units]')
    axes[1].set_yscale('log')
    axes[1].set_title('Density profile (log)')
    axes[1].legend()

    plt.suptitle(f'{os.path.basename(args.run_dir)}: Density profile')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "density_profile.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    # ---- Plot: density ratio ----
    _fig, ax = plt.subplots(figsize=(10, 5))
    ratio = np.where(rho_data > 0, rho_model / rho_data, np.nan)
    valid = (rho_data > 0) & (data_counts > 50) & (model_counts > 50)
    ax.plot(r_mid[valid], ratio[valid], 'ko-', ms=3, lw=1.5)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('model / data')
    ax.set_title('Density ratio (model / data)')
    ax.set_ylim(0, 2.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "density_ratio.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # ---- Plot: velocity dispersion profiles ----
    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    valid_d = np.isfinite(sigma_vr_data) & (data_counts > 50)
    valid_m = np.isfinite(sigma_vr_model) & (model_counts > 50)

    axes[0].plot(r_mid[valid_d], sigma_vr_data[valid_d], 'b-', lw=2, label='data')
    axes[0].plot(r_mid[valid_m], sigma_vr_model[valid_m], 'r--', lw=2, label='model')
    axes[0].set_xlabel('r [kpc]')
    axes[0].set_ylabel(r'$\sigma_{v_r}$ [km/s]')
    axes[0].set_title('Radial velocity dispersion')
    axes[0].legend()

    axes[1].plot(r_mid[valid_d], sigma_vt_data[valid_d], 'b-', lw=2, label='data')
    axes[1].plot(r_mid[valid_m], sigma_vt_model[valid_m], 'r--', lw=2, label='model')
    axes[1].set_xlabel('r [kpc]')
    axes[1].set_ylabel(r'$\sigma_{v_t}$ [km/s]')
    axes[1].set_title('Tangential velocity dispersion')
    axes[1].legend()

    axes[2].plot(r_mid[valid_d], sigma_v_data[valid_d], 'b-', lw=2, label='data')
    axes[2].plot(r_mid[valid_m], sigma_v_model[valid_m], 'r--', lw=2, label='model')
    axes[2].set_xlabel('r [kpc]')
    axes[2].set_ylabel(r'$\sigma_v$ [km/s]')
    axes[2].set_title('Total velocity dispersion')
    axes[2].legend()

    plt.suptitle(f'{os.path.basename(args.run_dir)}: Velocity dispersion profiles')
    plt.tight_layout()
    out_path = os.path.join(out_dir, "velocity_dispersion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # ---- Plot: anisotropy profile ----
    _fig, ax = plt.subplots(figsize=(10, 5))
    valid_d = np.isfinite(beta_data) & (data_counts > 100)
    valid_m = np.isfinite(beta_model) & (model_counts > 100)
    ax.plot(r_mid[valid_d], beta_data[valid_d], 'b-', lw=2, label='data')
    ax.plot(r_mid[valid_m], beta_model[valid_m], 'r--', lw=2, label='model')
    ax.axhline(0.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\beta = 1 - \sigma_t^2 / (2\sigma_r^2)$')
    ax.set_title('Velocity anisotropy profile')
    ax.set_ylim(-0.5, 1.0)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "anisotropy.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # ---- Cumulative mass profile ----
    # M(<r) proportional to cumulative count * particle mass
    cum_data = np.cumsum(data_counts).astype(float) / len(r_data)
    cum_model = np.cumsum(model_counts).astype(float) / len(r_model)

    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_mid, cum_data, 'b-', lw=2, label='data')
    ax.plot(r_mid, cum_model, 'r--', lw=2, label='model')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('Fraction of particles within r')
    ax.set_title('Cumulative radial distribution')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cumulative_radial.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
