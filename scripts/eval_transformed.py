"""Evaluate FFJORD runs in transformed space (after power+zscore, before inverse).

This bypasses the inverse coord_transform to isolate:
  A) How well the flow fits the transformed distribution?
  B) Are the distortions coming from the flow or the inverse transform?

Pipeline: raw → coord_transform(power) → clip → normalizer(zscore) → flow
We compare flow samples and data in the same standardized model space.

When --mass-npz is provided (an npz with 'mass_for_a' field aligned 1-to-1 to
--data rows), additionally produces mass-weighted variants of marginals, KS,
QQ plots to address the question: "is the std/kurtosis offset partly an
artifact of equal-weight histogramming when f(η) is the mass density?"
"""
import argparse
import os

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from dpjax.flows.api import sample_apply
from experiments.datasets.phase_space import load_eta_h5
from experiments.workflows.artifacts import load_df


def _stats_1d(x, w=None):
    """Return (mean, std, kurt) with optional mass-weighting."""
    if w is None:
        m = x.mean()
        s = x.std()
        k = sp_stats.kurtosis(x, fisher=True)
    else:
        w = w / w.sum()
        m = (w * x).sum()
        var = (w * (x - m) ** 2).sum()
        s = np.sqrt(var)
        k = (w * (x - m) ** 4).sum() / (var * var) - 3.0
    return float(m), float(s), float(k)


def _quantile_weighted(x, w, n_out):
    """Mass-weighted empirical quantiles of x."""
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws) / ws.sum()
    # Avoid CDF endpoints 0/1 by inserting the linspace(0,1) directly
    q_targets = np.linspace(0.0, 1.0, n_out)
    return np.interp(q_targets, cdf, xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="data/halo_12_train.h5")
    parser.add_argument("--key", default="eta")
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--mass-npz", default=None,
                        help="Optional: npz with 'mass_for_a' field, "
                             "aligned 1-to-1 to --data rows. Enables "
                             "mass-weighted marginals, KS and QQ plots.")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_dir, "eval_transformed")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, params, normalizer, cfg, coord_transform = load_df(args.run_dir)
    flow_cfg = cfg.get("flow", {})
    print(f"Flow: {flow_cfg.get('type')}, dim={flow_cfg.get('dim')}")
    if coord_transform is not None:
        print(f"Coord transform: {coord_transform.type!r} on dims {coord_transform.dims.tolist()}")

    # Load raw data and apply SAME pipeline as training
    raw_data = load_eta_h5(args.data, dataset="eta")
    raw_data = np.asarray(raw_data, dtype=np.float32)

    # Load mass weights (aligned to raw_data ordering)
    mass_raw = None
    if args.mass_npz is not None:
        with np.load(args.mass_npz) as d:
            mass_raw = np.asarray(d["mass_for_a"], dtype=np.float64)
        if mass_raw.shape[0] != raw_data.shape[0]:
            raise ValueError(
                f"mass N ({mass_raw.shape[0]}) != eta N ({raw_data.shape[0]}); "
                f"did you pass the wrong alignment npz?"
            )
        print(f"Loaded mass: N={mass_raw.shape[0]}  "
              f"mean={mass_raw.mean():.4e}  std={mass_raw.std():.4e}  "
              f"max/min={mass_raw.max()/mass_raw.min():.1f}x")

    # Forward: coord_transform → clip → normalizer
    data_fwd = raw_data.copy()
    if coord_transform is not None:
        data_fwd = coord_transform.transform(data_fwd)

    # Apply clip (same sigma as training); mass follows along
    clip_sigma = cfg.get("data", {}).get("clip_sigma", 0.0)
    n_before = data_fwd.shape[0]
    if clip_sigma > 0:
        clip_mean = np.mean(data_fwd, axis=0)
        clip_std = np.std(data_fwd, axis=0)
        clip_std = np.maximum(clip_std, 1e-6)
        mask = np.all(np.abs(data_fwd - clip_mean) < clip_sigma * clip_std, axis=1)
        data_fwd = data_fwd[mask]
        if mass_raw is not None:
            mass_raw = mass_raw[mask]
        n_after = data_fwd.shape[0]
        print(f"Clipped {n_before - n_after}/{n_before} ({100*(n_before-n_after)/n_before:.2f}%)")

    data_norm = normalizer.transform(data_fwd)
    print(f"Transformed data shape: {data_norm.shape}")

    # Sample from flow (IN NORMALIZED SPACE)
    rng = jax.random.PRNGKey(42)
    rng, rng_sample = jax.random.split(rng)
    print(f"Sampling {args.n_samples} points...")
    samples_norm = np.asarray(sample_apply(model, params, rng_sample, args.n_samples, flow_cfg))

    # Compare both arrays in the exact standardized space seen by the flow.
    data_power = data_norm
    samples_power = samples_norm
    # Flow samples correspond to equal-weight draws from p_theta by definition.
    # (Indeed the DF is a mass density, but flow samples should be interpreted
    # by mass weight of each sample cell — which here is unknown a priori, so
    # we keep equal-weight samples, exactly as the model is trained!)

    raw_labels = ["x", "y", "z", "vx", "vy", "vz"]

    # ---- Statistics table (equal-weight + mass-weighted) ----
    print("\n" + "=" * 90)
    print("Statistics in POWER(α) + z-score space")
    print(f"  alpha: {coord_transform.params.get('alpha', 'n/a') if coord_transform is not None else 'n/a'}")
    print("=" * 90)
    if mass_raw is not None:
        print(f"{'Dim':>4}  {'d_mean_eq':>10} {'d_mean_w':>10} {'d_std_eq':>10} {'d_std_w':>10} "
              f"{'m_std_eq':>10} {'d_kurt_eq':>10} {'d_kurt_w':>10} {'m_kurt_eq':>10}")
    else:
        print(f"{'Dim':>4}  {'data_mean':>12}  {'model_mean':>12}  {'data_std':>12}  {'model_std':>12}  {'data_kurt':>12}  {'model_kurt':>12}")
    for i in range(6):
        d = data_power[:, i]
        s = samples_power[:, i]
        d_mean_eq, d_std_eq, d_kurt_eq = _stats_1d(d)
        s_mean_eq, s_std_eq, s_kurt_eq = _stats_1d(s)
        if mass_raw is not None:
            d_mean_w, d_std_w, d_kurt_w = _stats_1d(d, mass_raw)
            print(f"{raw_labels[i]:>4}  {d_mean_eq:>10.3f} {d_mean_w:>10.3f} {d_std_eq:>10.3f} {d_std_w:>10.3f} "
                  f"{s_std_eq:>10.3f} {d_kurt_eq:>10.2f} {d_kurt_w:>10.2f} {s_kurt_eq:>10.2f}")
        else:
            print(f"{raw_labels[i]:>4}  {d_mean_eq:>12.4f}  {s_mean_eq:>12.4f}  {d_std_eq:>12.4f}  {s_std_eq:>12.4f}  {d_kurt_eq:>12.2f}  {s_kurt_eq:>12.2f}")

    # ---- KS test in transformed space ----
    print("\n" + "=" * 70)
    print("1D KS test in transformed space (equal-weight)")
    print("=" * 70)
    for i in range(6):
        ks_stat, p_val = sp_stats.ks_2samp(data_power[:, i], samples_power[:, i])
        print(f"  {raw_labels[i]:>4}: KS={ks_stat:.4f}, p={p_val:.2e}  {'✓' if p_val > 0.01 else '✗'}")

    # Mass-weighted KS test via resampling
    # (CDF-based weighted KS is non-trivial; easier to draw a mass-weighted
    #  resample of the same size and compare to model samples.)
    if mass_raw is not None:
        print("\n" + "=" * 70)
        print("1D KS in transformed space (data mass-weighted resample)")
        print("=" * 70)
        rng_ks = np.random.default_rng(123)
        n_res = int(args.n_samples)  # match number of model samples
        for i in range(6):
            w = mass_raw / mass_raw.sum()
            idx = rng_ks.choice(data_power.shape[0], size=n_res, replace=True, p=w)
            d_resamp = data_power[idx, i]
            ks_stat, p_val = sp_stats.ks_2samp(d_resamp, samples_power[:, i])
            print(f"  {raw_labels[i]:>4}: KS={ks_stat:.4f}, p={p_val:.2e}  {'✓' if p_val > 0.01 else '✗'}")

    # Helper used for marginal histograms
    def _draw_marginal_panel(axes, data, samples, weights_data=None,
                             data_color="steelblue", data_label="data",
                             log_scale=False):
        for i, ax in enumerate(axes):
            if weights_data is not None:
                # density + weights makes pdf under mass-weighted measure
                w = weights_data / weights_data.sum()
                ax.hist(data[:, i], bins=200, density=True, weights=w, alpha=0.6,
                        label=data_label, color=data_color)
            else:
                ax.hist(data[:, i], bins=200, density=True, alpha=0.5,
                        label=data_label, color=data_color)
            ax.hist(samples[:, i], bins=200, density=True, alpha=0.5,
                    label="model", color="coral")
            ax.set_xlabel(f"{raw_labels[i]} (transformed)")
            ax.set_ylabel("density")
            ax.legend(loc='upper right', fontsize=8)
            if log_scale:
                ax.set_yscale("log")

    # ---- Marginal histograms (equal-weight) ----
    _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _draw_marginal_panel(axes.flatten(), data_power, samples_power,
                         weights_data=None, log_scale=True)
    plt.suptitle(f"{os.path.basename(args.run_dir)}: Transformed space (equal-weight, log)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "marginals_transformed_log.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")

    _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _draw_marginal_panel(axes.flatten(), data_power, samples_power,
                         weights_data=None, log_scale=False)
    plt.suptitle(f"{os.path.basename(args.run_dir)}: Transformed space (equal-weight, linear)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "marginals_transformed_linear.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # ---- Marginal histograms (mass-weighted data) ----
    if mass_raw is not None:
        _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        _draw_marginal_panel(axes.flatten(), data_power, samples_power,
                             weights_data=mass_raw, log_scale=True,
                             data_color="seagreen", data_label="data (mass-weighted)")
        plt.suptitle(f"{os.path.basename(args.run_dir)}: Transformed space (mass-weighted data, log)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "marginals_transformed_weighted_log.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

        _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        _draw_marginal_panel(axes.flatten(), data_power, samples_power,
                             weights_data=mass_raw, log_scale=False,
                             data_color="seagreen", data_label="data (mass-weighted)")
        plt.suptitle(f"{os.path.basename(args.run_dir)}: Transformed space (mass-weighted data, linear)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "marginals_transformed_weighted_linear.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # ---- QQ plots (transformed space) ----
    # Equal-weight QQ: sort both, take equal-quantile indices.
    _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        d_sorted = np.sort(data_power[:, i])
        s_sorted = np.sort(samples_power[:, i])
        n_min = min(len(d_sorted), len(s_sorted))
        d_sub = d_sorted[np.linspace(0, len(d_sorted)-1, n_min).astype(int)]
        s_sub = s_sorted[np.linspace(0, len(s_sorted)-1, n_min).astype(int)]
        ax.scatter(d_sub[::max(1, n_min//2000)], s_sub[::max(1, n_min//2000)],
                   s=1, alpha=0.3, color="coral")
        lims = [min(d_sub.min(), s_sub.min()), max(d_sub.max(), s_sub.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel(f"data {raw_labels[i]} (transformed)")
        ax.set_ylabel(f"model {raw_labels[i]} (transformed)")
        ax.set_title(f"QQ: {raw_labels[i]} (equal-weight)")
        ax.set_aspect('equal')
    plt.suptitle(f"{os.path.basename(args.run_dir)}: QQ plots (equal-weight)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "qq_transformed.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # Mass-weighted QQ: data side uses mass CDF to get quantiles.
    if mass_raw is not None:
        _fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            s_sorted = np.sort(samples_power[:, i])
            n_min = min(data_power.shape[0], len(s_sorted))
            d_sub = _quantile_weighted(data_power[:, i], mass_raw, n_min)
            s_sub = s_sorted[np.linspace(0, len(s_sorted)-1, n_min).astype(int)]
            ax.scatter(d_sub[::max(1, n_min//2000)], s_sub[::max(1, n_min//2000)],
                       s=1, alpha=0.3, color="coral")
            lims = [min(d_sub.min(), s_sub.min()), max(d_sub.max(), s_sub.max())]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlabel(f"data {raw_labels[i]} (mass-weighted quantile)")
            ax.set_ylabel(f"model {raw_labels[i]} (transformed)")
            ax.set_title(f"QQ: {raw_labels[i]} (mass-weighted data)")
            ax.set_aspect('equal')
        plt.suptitle(f"{os.path.basename(args.run_dir)}: QQ plots (data side mass-weighted)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "qq_transformed_weighted.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # ---- z-dimension focus in transformed space ----
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    i = 2  # z dimension
    axes[0].hist(data_power[:, i], bins=300, density=True, alpha=0.5, label="data", color="steelblue")
    axes[0].hist(samples_power[:, i], bins=300, density=True, alpha=0.5, label="model", color="coral")
    axes[0].set_xlabel("z (transformed)")
    axes[0].set_ylabel("density")
    axes[0].set_title("z (transformed) - linear")
    axes[0].legend()
    axes[1].hist(data_power[:, i], bins=300, density=True, alpha=0.5, label="data", color="steelblue")
    axes[1].hist(samples_power[:, i], bins=300, density=True, alpha=0.5, label="model", color="coral")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("z (transformed)")
    axes[1].set_ylabel("density")
    axes[1].set_title("z (transformed) - log")
    axes[1].legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "z_focus_transformed.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # Mass-weighted z focus
    if mass_raw is not None:
        _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        i = 2
        w_n = mass_raw / mass_raw.sum()
        axes[0].hist(data_power[:, i], bins=300, density=True, weights=w_n,
                     alpha=0.5, label="data (mass-weighted)", color="seagreen")
        axes[0].hist(samples_power[:, i], bins=300, density=True, alpha=0.5, label="model", color="coral")
        axes[0].set_xlabel("z (transformed)")
        axes[0].set_ylabel("density")
        axes[0].set_title("z (transformed) - linear (mass-weighted data)")
        axes[0].legend()
        axes[1].hist(data_power[:, i], bins=300, density=True, weights=w_n,
                     alpha=0.5, label="data (mass-weighted)", color="seagreen")
        axes[1].hist(samples_power[:, i], bins=300, density=True, alpha=0.5, label="model", color="coral")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("z (transformed)")
        axes[1].set_ylabel("density")
        axes[1].set_title("z (transformed) - log (mass-weighted data)")
        axes[1].legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, "z_focus_transformed_weighted.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
