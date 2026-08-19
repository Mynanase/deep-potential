"""Plot the analytic Plummer DF score field (plummer_score_std_batch) and
compare it with the trained FFJORD DF score on real data points.

Part 1 (CPU): shape of the analytic score in the (r, v) plane.
Part 2 (GPU): analytic score vs trained-flow score on real samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/localdisk/kosmos/my-deep-potential")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/localdisk/kosmos/my-deep-potential")
DF_RUN = ROOT / "runs/plummer_rcut/full/trial_00/df"
DATA = ROOT / "data/plummer_n524288_train.h5"
OUT = ROOT / "runs/plummer_rcut/full/eval/plots/score_compare"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Part 1: analytic score shape in the (r, v) plane
#   E = 0.5 v^2 - 1/sqrt(1 + r^2),  f ~ (-E)^(7/2)
#   d log f / dr = (7/2) * r / ((1+r^2)^{3/2} * E)          (E < 0)
#   d log f / dv = (7/2) * v / E
# ---------------------------------------------------------------------------
def plot_score_shape() -> None:
    r = np.linspace(0.001, 5.0, 200)
    v = np.linspace(0.001, 1.6, 200)
    RR, VV = np.meshgrid(r, v, indexing="ij")
    E = 0.5 * VV**2 - 1.0 / np.sqrt(1.0 + RR**2)
    bound = E < 0.0
    sr = np.zeros_like(E)
    sv = np.zeros_like(E)
    sr[bound] = 3.5 * RR[bound] / ((1.0 + RR[bound] ** 2) ** 1.5 * E[bound])
    sv[bound] = 3.5 * VV[bound] / E[bound]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    for ax, field, tag in [
        (axes[0], sr, r"$\partial_r\log f$"),
        (axes[1], sv, r"$\partial_v\log f$"),
    ]:
        img = np.where(bound, np.log10(np.abs(field)), np.nan)
        im = ax.pcolormesh(VV, RR, img, cmap="magma", shading="auto")
        fig.colorbar(im, ax=ax, label=rf"$\log_{{10}}|{tag}|$")
        # boundary E = 0
        vb = np.sqrt(2.0 / np.sqrt(1.0 + r**2))
        ax.plot(vb, r, color="cyan", lw=1.0, ls="--", label=r"$E=0$")
        ax.set_xlabel("v")
        ax.set_ylabel("r")
        ax.set_title(f"Analytic Plummer score {tag}")
        ax.legend(fontsize=8)
    fig.suptitle("Analytic Plummer DF score (r, v) plane")
    fig.tight_layout()
    fig.savefig(OUT / "score_shape_rv.png", bbox_inches="tight")
    plt.close(fig)

    # Slices at fixed radii
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    for ax, r_fix in zip(axes, [0.3, 1.0, 3.0]):
        E_ = 0.5 * v**2 - 1.0 / np.sqrt(1.0 + r_fix**2)
        m = E_ < 0
        ax.plot(v[m], 3.5 * v[m] / E_[m], color="C0", label=r"$\partial_v\log f$")
        ax.plot(v[m], 3.5 * r_fix / ((1.0 + r_fix**2) ** 1.5 * E_[m]),
                color="C2", ls="--", label=r"$\partial_r\log f$")
        ax.set_xlabel("v")
        ax.set_title(f"r = {r_fix}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Analytic Plummer DF score slices (E < 0)")
    fig.tight_layout()
    fig.savefig(OUT / "score_shape_slices.png", bbox_inches="tight")
    plt.close(fig)
    print("[1] wrote", OUT / "score_shape_rv.png", OUT / "score_shape_slices.png")


# ---------------------------------------------------------------------------
# Part 2: analytic vs trained-flow score on real samples (needs GPU)
# ---------------------------------------------------------------------------
def plot_score_vs_flow() -> None:
    import jax.numpy as jnp

    from dpjax.flows.api import score_apply
    from experiments.datasets.phase_space import load_eta_h5
    from experiments.validation.plummer import plummer_score_std_batch
    from experiments.workflows.artifacts import load_df

    print("[2] loading data + DF model ...")
    eta_real = load_eta_h5(DATA)
    df_model, df_params, normalizer, df_cfg, _ = load_df(DF_RUN)
    mean = np.asarray(normalizer.mean, dtype=np.float32)
    std = np.asarray(normalizer.std, dtype=np.float32)
    eta_std = jnp.asarray((eta_real - mean[None, :]) / std[None, :])

    rng_key = np.random.default_rng(0).integers(0, 2**31)
    n = 65_536
    idx = np.arange(eta_real.shape[0])
    rng = np.random.default_rng(rng_key)
    rng.shuffle(idx)
    sel = idx[:n]
    eta_std_sel = eta_std[sel]

    print("[2] analytic score ...")
    s_analytic_std = np.asarray(
        plummer_score_std_batch(eta_std_sel, jnp.asarray(mean), jnp.asarray(std)),
        dtype=np.float32,
    )
    print("[2] flow score ...")
    s_flow_std = np.asarray(
        score_apply(df_model, df_params, eta_std_sel, dict(df_cfg.get("flow") or {})),
        dtype=np.float32,
    )
    # physical gradients: d(log f)/d eta_phys = score_std / std
    s_analytic = s_analytic_std / std[None, :]
    s_flow = s_flow_std / std[None, :]

    r = np.linalg.norm(eta_real[sel, :3], axis=1)
    vmag = np.linalg.norm(eta_real[sel, 3:], axis=1)
    rhat = eta_real[sel, :3] / r[:, None]
    vhat = eta_real[sel, 3:] / np.maximum(vmag[:, None], 1e-12)

    # radial projections of the position score and speed score
    sa_r = np.sum(s_analytic[:, :3] * rhat, axis=1)
    sf_r = np.sum(s_flow[:, :3] * rhat, axis=1)
    sa_v = np.sum(s_analytic[:, 3:] * vhat, axis=1)
    sf_v = np.sum(s_flow[:, 3:] * vhat, axis=1)

    def scatter2d(ax, a, b, xlabel, ylabel, title, lim):
        hb = ax.hist2d(a, b, bins=120, cmap="turbo", norm=matplotlib.colors.LogNorm())
        fig.colorbar(hb[3], ax=ax, label="count")
        lim = max(abs(lim[0]), abs(lim[1]))
        ax.plot([-lim, lim], [-lim, lim], color="w", lw=0.8, ls="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    ag = np.nanpercentile(np.abs(sa_r), 99.8)
    fg = np.nanpercentile(np.abs(sf_r), 99.8)
    lim = max(ag, fg) * 1.05
    scatter2d(axes[0], sa_r, sf_r, "analytic score$_r$", "flow score$_r$",
              "position score along $\\hat r$", [lim, -lim * 0.4])
    avg = np.nanpercentile(np.abs(sa_v), 99.8)
    fvgm = np.nanpercentile(np.abs(sf_v), 99.8)
    limv = max(avg, fvgm) * 1.05
    scatter2d(axes[1], sa_v, sf_v, "analytic score$_v$", "flow score$_v$",
              "velocity score along $\\hat v$", [limv, -limv * 0.4])
    fig.suptitle("DF score: analytic Plummer vs trained FFJORD (real data)")
    fig.tight_layout()
    fig.savefig(OUT / "score_analytic_vs_flow.png", bbox_inches="tight")
    plt.close(fig)

    # Residual vs radius (binned medians)
    edges = np.geomspace(max(r.min(), 1e-3), r.max(), 20)
    medians_pos = []
    medians_vel = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() == 0:
            continue
        medians_pos.append((np.sqrt(lo * hi), np.median(np.abs(sa_r[m] - sf_r[m]))))
        medians_vel.append((np.sqrt(lo * hi), np.median(np.abs(sa_v[m] - sf_v[m]))))
    medians_pos = np.array(medians_pos)
    medians_vel = np.array(medians_vel)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(medians_pos[:, 0], medians_pos[:, 1], "o-", color="C2", label="position component")
    ax.plot(medians_vel[:, 0], medians_vel[:, 1], "s-", color="C0", label="velocity component")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("median |analytic - flow| score")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    ax.set_title("DF score difference vs radius (real data)")
    fig.tight_layout()
    fig.savefig(OUT / "score_diff_vs_r.png", bbox_inches="tight")
    plt.close(fig)
    print("[2] wrote", OUT / "score_analytic_vs_flow.png", OUT / "score_diff_vs_r.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-shape-only", action="store_true",
                        help="Draw only the analytic score shape (no flow comparison).")
    args = parser.parse_args()
    plot_score_shape()
    if not args.cpu_shape_only:
        plot_score_vs_flow()


if __name__ == "__main__":
    main()