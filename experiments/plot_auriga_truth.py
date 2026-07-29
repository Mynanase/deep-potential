"""Plot Phi potential evaluation against Auriga simulator truth.

Reads the ``auriga_truth_metrics.json`` + ``auriga_truth_predictions.npz``
produced by :mod:`experiments.eval_auriga_truth` and renders:

* ``potential_scatter.png``       – aligned Φ_model vs Φ_truth scatter
* ``potential_radial_profile.png`` – radial median profile with error bands
* ``potential_residual_vs_r.png``  – residual |Φ_model − Φ_truth| vs r
* ``potential_error_hist.png``     – signed error distribution

Acceleration truth is optional: when present, an additional
``acceleration_radial_profile.png`` and ``acceleration_scatter.png`` are
rendered; otherwise the script silently skips them.

Examples
--------
    python -m experiments.plot_auriga_truth \
        --eval-dir runs/halo_12/phi_static_v1_seed43/truth_evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _radial_bin_median(r: np.ndarray, values: np.ndarray, edges: np.ndarray):
    """Per-bin median of ``values`` grouped by radial ``r``."""
    idx = np.clip(np.digitize(r, edges) - 1, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    med = np.full(n_bins, np.nan)
    p25 = np.full(n_bins, np.nan)
    p75 = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        med[b] = np.median(values[sel])
        p25[b], p75[b] = np.percentile(values[sel], [25, 75])
    return med, p25, p75


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Auriga simulator-truth evaluation results.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory containing auriga_truth_metrics.json and "
        "auriga_truth_predictions.npz.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = Path(args.eval_dir)
    metrics_path = eval_dir / "auriga_truth_metrics.json"
    pred_path = eval_dir / "auriga_truth_predictions.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}")

    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    d = np.load(pred_path)

    position = np.asarray(d["position"], dtype=np.float64)
    r = np.sqrt(np.sum(position**2, axis=-1))

    # ── Potential plots ────────────────────────────────────────────────
    if "potential" in metrics:
        aligned = np.asarray(d["aligned_predicted_potential"], dtype=np.float64)
        truth = np.asarray(d["truth_potential"], dtype=np.float64)
        if truth.size == 0:
            print("Truth potential array is empty; skipping potential plots.")
        else:
            pm = metrics["potential"]

            # scatter: aligned predicted vs truth
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=args.dpi)
            rng = np.random.default_rng(0)
            n_show = min(20000, truth.size)
            sel = rng.choice(truth.size, n_show, replace=False)
            ax.scatter(truth[sel], aligned[sel], s=2, alpha=0.25, color="tab:blue")
            lo = float(min(truth.min(), aligned.min()))
            hi = float(max(truth.max(), aligned.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
            ax.set_xlabel(r"$\Phi_{truth}$ [(km/s)$^2$]")
            ax.set_ylabel(r"$\Phi_{model}$ (offset-aligned) [(km/s)$^2$]")
            ax.set_title(
                f"Potential scatter (Pearson r={pm['pearson_r']:.3f}, "
                f"nRMSE={pm['normalized_rmse']:.3f})"
            )
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "potential_scatter.png")
            plt.close(fig)
            print(f"saved {out_dir/'potential_scatter.png'}")

            # radial profile
            edges = np.linspace(0, float(np.percentile(r, 99)), 13)
            r_centers = np.sqrt(edges[:-1] * edges[1:])
            med_m, p25_m, p75_m = _radial_bin_median(r, aligned, edges)
            med_t, p25_t, p75_t = _radial_bin_median(r, truth, edges)

            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=args.dpi)
            ax.plot(r_centers, med_t, "k-o", lw=2, ms=4, label="truth median")
            ax.fill_between(r_centers, p25_t, p75_t, color="k", alpha=0.15)
            ax.plot(r_centers, med_m, "--", color="tab:orange", lw=1.8, ms=4, label="model median")
            ax.fill_between(r_centers, p25_m, p75_m, color="tab:orange", alpha=0.15)
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel(r"$\Phi$ [(km/s)$^2$]")
            ax.set_title("Radial potential profile (offset-aligned)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "potential_radial_profile.png")
            plt.close(fig)
            print(f"saved {out_dir/'potential_radial_profile.png'}")

            # residual vs r
            resid = aligned - truth
            abs_resid = np.abs(resid)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=args.dpi)
            rng = np.random.default_rng(0)
            sel = rng.choice(r.size, min(20000, r.size), replace=False)
            axes[0].scatter(r[sel], resid[sel], s=2, alpha=0.25, color="tab:purple")
            axes[0].axhline(0.0, color="k", lw=1)
            axes[0].set_xlabel("r [kpc]")
            axes[0].set_ylabel(r"$\Phi_{model}-\Phi_{truth}$ [(km/s)$^2$]")
            axes[0].set_title("Signed residual vs r")
            axes[0].grid(True, alpha=0.3)
            axes[1].scatter(r[sel], abs_resid[sel], s=2, alpha=0.25, color="tab:red")
            axes[1].set_xlabel("r [kpc]")
            axes[1].set_ylabel(r"$|\Phi_{model}-\Phi_{truth}|$ [(km/s)$^2$]")
            axes[1].set_yscale("log")
            axes[1].set_title("Absolute residual vs r")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "potential_residual_vs_r.png")
            plt.close(fig)
            print(f"saved {out_dir/'potential_residual_vs_r.png'}")

            # error histogram
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=args.dpi)
            ax.hist(resid, bins=60, color="tab:gray", alpha=0.85)
            ax.axvline(0.0, color="k", lw=1)
            ax.set_xlabel(r"$\Phi_{model}-\Phi_{truth}$ [(km/s)$^2$]")
            ax.set_ylabel("count")
            ax.set_title(
                f"Signed error (MAE={pm['mae']:.1f}, RMSE={pm['rmse']:.1f}, "
                f"p90|e|={pm['p90_abs_error']:.1f})"
            )
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "potential_error_hist.png")
            plt.close(fig)
            print(f"saved {out_dir/'potential_error_hist.png'}")

    # ── Acceleration plots (optional) ─────────────────────────────────
    if "acceleration" in metrics:
        pred_acc = np.asarray(d["predicted_acceleration"], dtype=np.float64)
        truth_acc = np.asarray(d["truth_acceleration"], dtype=np.float64)
        if truth_acc.size == 0 or truth_acc.shape == (0, 3):
            print("Truth acceleration array is empty; skipping acceleration plots.")
        else:
            am = metrics["acceleration"]
            pred_mag = np.linalg.norm(pred_acc, axis=-1)
            truth_mag = np.linalg.norm(truth_acc, axis=-1)

            # scatter
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=args.dpi)
            rng = np.random.default_rng(0)
            sel = rng.choice(truth_mag.size, min(20000, truth_mag.size), replace=False)
            ax.scatter(truth_mag[sel], pred_mag[sel], s=2, alpha=0.25, color="tab:green")
            lo = float(min(truth_mag.min(), pred_mag.min()))
            hi = float(max(truth_mag.max(), pred_mag.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
            ax.set_xlabel(r"$|a_{truth}|$")
            ax.set_ylabel(r"$|a_{model}|$")
            ax.set_title(
                f"Accel magnitude scatter (median rel err="
                f"{am.get('median_relative_error', float('nan')):.3f})"
            )
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "acceleration_scatter.png")
            plt.close(fig)
            print(f"saved {out_dir/'acceleration_scatter.png'}")

            # radial profile
            if "radial_acceleration" in metrics:
                ra = metrics["radial_acceleration"]
                rows = ra["bins"]
                r_centers = np.sqrt(
                    np.asarray([row["r_left"] for row in rows])
                    * np.asarray([row["r_right"] for row in rows])
                )
                median_relative = np.asarray(
                    [row["median_relative_error"] for row in rows]
                )
                p90_relative = np.asarray(
                    [row["p90_relative_error"] for row in rows]
                )
                fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=args.dpi)
                ax.plot(
                    r_centers,
                    median_relative,
                    "o-",
                    color="tab:green",
                    lw=1.8,
                    ms=4,
                    label="median",
                )
                ax.plot(
                    r_centers,
                    p90_relative,
                    "s--",
                    color="tab:orange",
                    lw=1.5,
                    ms=4,
                    label="p90",
                )
                ax.set_xlabel("r [kpc]")
                ax.set_ylabel(r"$|a_{\rm model}-a_{\rm truth}|/|a_{\rm truth}|$")
                ax.set_title("Radial acceleration relative error")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out_dir / "acceleration_radial_profile.png")
                plt.close(fig)
                print(f"saved {out_dir/'acceleration_radial_profile.png'}")

    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
