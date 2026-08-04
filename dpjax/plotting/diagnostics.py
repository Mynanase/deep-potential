"""Diagnostic plotting functions for Deep Potential evaluation.

All functions are pure numpy + matplotlib — no JAX or model dependency.
They consume pre-computed `.npz` data saved by ``eval_df.py`` / ``eval_phi.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ── 1. Phase-space overview grid ─────────────────────────────────────────

def plot_phase_space_grid(
    eta: np.ndarray,
    *,
    eta_sample: Optional[np.ndarray] = None,
    xlim: tuple[float, float] = (-3.0, 3.0),
    vlim: tuple[float, float] = (-1.5, 1.5),
    bins_2d: int = 31,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Render a 3×3 phase-space overview.

    Row 0: spatial projections (x-y, x-z, y-z)
    Row 1: velocity projections (vx-vy, vx-vz, vy-vz)
    Row 2: r-v diagram, spatial isotropy, velocity isotropy

    If *eta_sample* is provided, a second (outline) histogram is overlaid.
    """
    import matplotlib.pyplot as plt

    eta = np.asarray(eta)

    def _vec2ang(x: np.ndarray):
        phi = np.arctan2(x[:, 1], x[:, 0])
        theta = np.arctan2(x[:, 2], np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))
        return theta, phi

    fig, ax_arr = plt.subplots(3, 3, figsize=(13, 12), dpi=dpi)
    fig.subplots_adjust(wspace=0.30, hspace=0.25)

    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax_arr[0, k].hist2d(eta[:, i], eta[:, j], bins=bins_2d, range=[xlim, xlim])
        ax_arr[1, k].hist2d(eta[:, i + 3], eta[:, j + 3], bins=bins_2d, range=[vlim, vlim])
        dim_labels = ["x", "y", "z"]
        ax_arr[0, k].set_xlabel(rf"${dim_labels[i]}$")
        ax_arr[0, k].set_ylabel(rf"${dim_labels[j]}$", labelpad=-5)
        ax_arr[1, k].set_xlabel(rf"$v_{dim_labels[i]}$")
        ax_arr[1, k].set_ylabel(rf"$v_{dim_labels[j]}$", labelpad=-5)
        ax_arr[0, k].set_aspect("equal")
        ax_arr[1, k].set_aspect("equal")

    r = np.sqrt(np.sum(eta[:, :3] ** 2, axis=1))
    v = np.sqrt(np.sum(eta[:, 3:] ** 2, axis=1))
    ax_arr[2, 0].hist2d(r, v, bins=bins_2d, range=[(0.0, 5.0), (0.0, 1.5)])
    ax_arr[2, 0].set_xlabel(r"$r$")
    ax_arr[2, 0].set_ylabel(r"$v$", labelpad=0)

    iso_bins = 11
    v0 = eta.shape[0] / iso_bins ** 2
    dv = 0.5 * v0

    theta, phi = _vec2ang(eta[:, :3])
    ax_arr[2, 1].hist2d(phi, np.sin(theta), bins=iso_bins, vmin=v0 - dv, vmax=v0 + dv, cmap="bwr_r")
    ax_arr[2, 1].set_xlabel(r"$\varphi_x$")
    ax_arr[2, 1].set_ylabel(r"$\sin \theta_x$", labelpad=-5)

    theta, phi = _vec2ang(eta[:, 3:])
    ax_arr[2, 2].hist2d(phi, np.sin(theta), bins=iso_bins, vmin=v0 - dv, vmax=v0 + dv, cmap="bwr_r")
    ax_arr[2, 2].set_xlabel(r"$\varphi_v$")
    ax_arr[2, 2].set_ylabel(r"$\sin \theta_v$", labelpad=-5)

    for a in ax_arr[2]:
        a.set_aspect("auto")

    if eta_sample is not None:
        eta_s = np.asarray(eta_sample)
        for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
            ax_arr[0, k].hist2d(eta_s[:, i], eta_s[:, j], bins=bins_2d, range=[xlim, xlim], alpha=0.0)
            counts_s, xedges, yedges = np.histogram2d(eta_s[:, i], eta_s[:, j], bins=bins_2d, range=[xlim, xlim])
            ax_arr[0, k].contour(
                counts_s.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                colors="r",
                linewidths=0.5,
                levels=5,
                alpha=0.6,
            )
        r_s = np.sqrt(np.sum(eta_s[:, :3] ** 2, axis=1))
        v_s = np.sqrt(np.sum(eta_s[:, 3:] ** 2, axis=1))
        counts_rv, xedges, yedges = np.histogram2d(r_s, v_s, bins=bins_2d, range=[(0.0, 5.0), (0.0, 1.5)])
        ax_arr[2, 0].contour(
            counts_rv.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            colors="r",
            linewidths=0.5,
            levels=5,
            alpha=0.6,
        )

    fig.suptitle("Phase-Space Overview", fontsize=16)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"phase_space_grid.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 2. Score gradient scatter comparison ─────────────────────────────────

def plot_score_comparison(
    score_true: np.ndarray,
    score_est: np.ndarray,
    *,
    dim_labels: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz"),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Compare true and learned scores per dimension with slope/R² annotations."""
    import matplotlib.pyplot as plt

    score_true = np.asarray(score_true)
    score_est = np.asarray(score_est)

    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9), dpi=dpi)

    for i, ax in enumerate(ax_arr.flat):
        ax.set_aspect("equal")
        ax.scatter(score_true[:, i], score_est[:, i], alpha=0.1, s=2, edgecolors="none")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lo = min(xlim[0], ylim[0])
        hi = max(xlim[1], ylim[1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], c="k", alpha=0.25)

        mask = np.isfinite(score_true[:, i]) & np.isfinite(score_est[:, i])
        if mask.sum() > 10:
            st, se = score_true[mask, i], score_est[mask, i]
            slope = float(np.sum(st * se) / (np.sum(st ** 2) + 1e-30))
            ss_res = np.sum((se - slope * st) ** 2)
            ss_tot = np.sum((se - np.mean(se)) ** 2) + 1e-30
            r2 = 1.0 - ss_res / ss_tot
        else:
            slope, r2 = float("nan"), float("nan")

        ax.text(
            0.05, 0.95, f"slope={slope:.3f}\nR\u00b2={r2:.3f}",
            ha="left", va="top", transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.set_xlabel("true")
        ax.set_ylabel("learned")
        ax.set_title(rf"$\partial \log f / \partial {dim_labels[i]}$")

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Score Gradient: True vs Learned", fontsize=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"score_comparison.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 3. Score gradient residual histograms ────────────────────────────────

def plot_score_residual_hist(
    score_true: np.ndarray,
    score_est: np.ndarray,
    *,
    dim_labels: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz"),
    resid_range: tuple[float, float] = (-0.05, 0.05),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Plot score residuals per dimension with σ and kurtosis."""
    import matplotlib.pyplot as plt

    score_true = np.asarray(score_true)
    score_est = np.asarray(score_est)
    resid = score_est - score_true

    fig, ax_arr = plt.subplots(2, 3, figsize=(16, 9), dpi=dpi)

    for i, ax in enumerate(ax_arr.flat):
        r_i = resid[:, i]
        ax.hist(r_i, bins=51, range=resid_range, log=True)
        ax.set_xlabel("(learned) - (true)")
        ax.set_title(rf"$\partial \log f / \partial {dim_labels[i]}$")

        sigma = float(np.std(r_i))
        mu = float(np.mean(r_i))
        centered = r_i - mu
        m2 = float(np.mean(centered ** 2))
        m4 = float(np.mean(centered ** 4))
        kurt = m4 / (m2 ** 2 + 1e-12) - 3.0

        ax.text(
            0.95, 0.95, f"$\\sigma = {sigma:.4f}$\n$\\kappa = {kurt:.2f}$",
            ha="right", va="top", transform=ax.transAxes, fontsize=10,
        )

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.91, bottom=0.06)
    fig.suptitle("Score Gradient Residual Histograms", fontsize=20)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"score_residual_hist.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 4. r-v distribution comparison ──────────────────────────────────────

def plot_rv_comparison(
    n_ideal: np.ndarray,
    n_samp: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    n_flow_total: int,
    *,
    r_lim: tuple[float, float] = (0.0, 5.0),
    v_lim: tuple[float, float] = (0.0, 1.5),
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Compare ideal and sampled r-v distributions on linear and log scales.

    Parameters
    ----------
    n_ideal : (Nv, Nr) ideal density grid
    n_samp : (Nv, Nr) histogram of flow samples (already transposed to match n_ideal)
    r, v : 1-D bin-center arrays
    n_flow_total : total number of flow samples used to build n_samp
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_ideal = np.asarray(n_ideal)
    n_samp = np.asarray(n_samp)

    extent = r_lim + v_lim

    fig, ax_arr = plt.subplots(3, 2, figsize=(11, 16), dpi=dpi)
    fig.subplots_adjust(left=0.1)

    # Row 0: ideal
    ax_arr[0, 0].imshow(n_ideal, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    img_log = np.log(np.clip(n_ideal, 1e-30, np.inf))
    vmax_log = np.max(img_log)
    ax_arr[0, 1].imshow(
        img_log, extent=extent, vmax=vmax_log, vmin=vmax_log - 10.0,
        origin="lower", aspect="auto", interpolation="nearest",
    )

    # Row 1: flow samples
    ax_arr[1, 0].imshow(n_samp, extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    n_samp_log = np.log(np.clip(n_samp, 1.0, np.inf))
    vmax_s = np.max(n_samp_log)
    ax_arr[1, 1].imshow(
        n_samp_log, extent=extent, vmax=max(vmax_s, 1.0), vmin=max(vmax_s - 10.0, 0.0),
        origin="lower", aspect="auto", interpolation="nearest",
    )

    # Row 2: residuals
    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dv = v[1] - v[0] if len(v) > 1 else 1.0
    n0 = n_ideal * dr * dv * n_flow_total
    denom = np.clip(n0, 1e-12, np.inf)
    rel_resid = (n_samp - n0) / denom

    ax_arr[2, 0].imshow(
        rel_resid, extent=extent, vmax=0.1, vmin=-0.1,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )
    log_resid = np.log(np.clip(n_samp, 1.0, np.inf)) - np.log(np.clip(n0, 1.0, np.inf))
    ax_arr[2, 1].imshow(
        log_resid, extent=extent, vmax=1.0, vmin=-1.0,
        origin="lower", aspect="auto", cmap="coolwarm_r", interpolation="nearest",
    )

    # Zero-energy line
    r_line = np.linspace(r_lim[0] + 0.01, r_lim[1], 200)
    v_line = np.sqrt(2.0) * (1.0 + r_line ** 2) ** (-0.25)
    for a in ax_arr.flat:
        a.plot(r_line, v_line, c="r")
        a.set_xlabel(r"$r$")
        a.set_ylabel(r"$v$")
        a.text(0.95, 0.95, r"$E > 0$", ha="right", va="top", fontsize=14, c="r", transform=a.transAxes)

    # Row labels
    for idx, label in enumerate(["Ideal DF", "Samples from Flow DF", "Residuals (Flow - Ideal)"]):
        pos = ax_arr[idx, 0].get_position()
        y_txt = 0.5 * (pos.y0 + pos.y1)
        fig.text(0.02, y_txt, label, rotation=90.0, ha="left", va="center", fontsize=16)

    ax_arr[0, 0].set_title("Linear Scale", fontsize=16)
    ax_arr[0, 1].set_title("Log Scale", fontsize=16)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"rv_comparison.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 5. Phi / ρ / |a| 2D slices ──────────────────────────────────────────

def plot_phi_rho_slice(
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    rho: np.ndarray,
    acc_mag: np.ndarray,
    *,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Plot three 2D slices: Φ(x,y), ρ(x,y), and |a|(x,y).

    Parameters
    ----------
    x, y : 1-D grid arrays
    phi, rho, acc_mag : 2-D images with shape (len(y), len(x))
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    rmax_x = max(abs(x[0]), abs(x[-1]))
    rmax_y = max(abs(y[0]), abs(y[-1]))
    extent = [-rmax_x, rmax_x, -rmax_y, rmax_y]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=dpi)

    # Phi (mean-subtracted)
    phi0 = phi - np.nanmean(phi)
    vmin, vmax = np.nanpercentile(phi0, [1, 99])
    if vmin * vmax < 0:
        divnorm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(vmin), vmax=float(vmax))
        im0 = axes[0].imshow(phi0, extent=extent, origin="lower", cmap="seismic", norm=divnorm)
    else:
        im0 = axes[0].imshow(phi0, extent=extent, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(r"$\Phi(x,y)$ (mean-sub.)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # ρ: retain the sign.  Clipping negative Laplacians before LogNorm makes
    # unphysical regions look like ordinary low-density pixels.
    finite_rho = rho[np.isfinite(rho)]
    has_negative = finite_rho.size > 0 and np.any(finite_rho < 0.0)
    if has_negative:
        rho_scale = max(
            float(np.nanpercentile(np.abs(finite_rho), 99.0)),
            1.0e-12,
        )
        rho_nonzero = np.abs(finite_rho[finite_rho != 0.0])
        rho_linthresh = (
            max(
                float(np.nanpercentile(rho_nonzero, 10.0)),
                rho_scale * 1.0e-4,
                1.0e-12,
            )
            if rho_nonzero.size
            else rho_scale * 1.0e-4
        )
        im1 = axes[1].imshow(
            rho,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            norm=colors.SymLogNorm(
                linthresh=rho_linthresh,
                vmin=-rho_scale,
                vmax=rho_scale,
            ),
        )
        negative_fraction = float(np.mean(finite_rho < 0.0))
        density_title = (
            r"signed $\rho = \nabla^2\Phi/(4\pi)$"
            f"\nnegative pixels: {negative_fraction:.1%}"
        )
    else:
        rho_pos = np.ma.masked_less_equal(rho, 0.0)
        positive_rho = finite_rho[finite_rho > 0.0]
        if positive_rho.size:
            vmin_r, vmax_r = np.nanpercentile(positive_rho, [5, 99])
        else:
            vmin_r, vmax_r = 1.0e-12, 1.0e-11
        im1 = axes[1].imshow(
            rho_pos,
            extent=extent,
            origin="lower",
            cmap="magma",
            norm=colors.LogNorm(
                vmin=max(float(vmin_r), 1.0e-12),
                vmax=max(float(vmax_r), 1.0e-11),
            ),
        )
        density_title = r"$\rho = \nabla^2\Phi/(4\pi)$"
    axes[1].set_title(density_title)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # |a| (log scale)
    vmin_a, vmax_a = np.nanpercentile(acc_mag, [5, 99])
    im2 = axes[2].imshow(
        acc_mag, extent=extent, origin="lower", cmap="cubehelix",
        norm=colors.LogNorm(vmin=max(vmin_a, 1e-12), vmax=max(vmax_a, 1e-11)),
    )
    axes[2].set_title(r"$|\mathbf{a}(x,y)|$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(r"Potential / Density / Acceleration (z = 0 slice)", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"phi_rho_acc_slice.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── 6. Radial density profile ρ(r) ──────────────────────────────────────

def plot_density_profile(
    r: np.ndarray,
    rho_learned: np.ndarray,
    rho_analytic: np.ndarray,
    *,
    phi_learned_shift: Optional[np.ndarray] = None,
    phi_true: Optional[np.ndarray] = None,
    ar_learned: Optional[np.ndarray] = None,
    ar_true: Optional[np.ndarray] = None,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Radial profiles: ρ(r) learned vs analytic, optionally Φ(r) and a_r(r).

    If ``phi_learned_shift`` / ``phi_true`` and ``ar_learned`` / ``ar_true``
    are provided, the figure has 3 columns; otherwise only the ρ(r) panel.
    """
    import matplotlib.pyplot as plt

    has_phi = phi_learned_shift is not None and phi_true is not None
    has_ar = ar_learned is not None and ar_true is not None
    n_cols = 1 + int(has_phi) + int(has_ar)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    col = 0
    if has_phi:
        axes[col].plot(r, phi_true, "k-", lw=2, label="analytic")
        axes[col].plot(r, phi_learned_shift, "--", lw=1.5, label="learned")
        axes[col].set_xscale("log")
        axes[col].set_xlabel("r")
        axes[col].set_ylabel(r"$\Phi(r)$")
        axes[col].set_title("Gravitational Potential")
        axes[col].legend()
        axes[col].grid(True, alpha=0.2)
        col += 1

    if has_ar:
        axes[col].plot(r, ar_true, "k-", lw=2, label="analytic")
        axes[col].plot(r, ar_learned, "--", lw=1.5, label="learned")
        axes[col].set_xscale("log")
        axes[col].set_xlabel("r")
        axes[col].set_ylabel(r"$a_r(r)$")
        axes[col].set_title("Radial Acceleration")
        axes[col].legend()
        axes[col].grid(True, alpha=0.2)
        col += 1

    axes[col].plot(r, rho_analytic, "k-", lw=2, label="analytic")
    axes[col].plot(r, rho_learned, "--", lw=1.5, label="learned", color="tab:orange")
    axes[col].set_xscale("log")
    axes[col].set_yscale("log")
    axes[col].set_xlabel("r")
    axes[col].set_ylabel(r"$\rho(r)$")
    axes[col].set_title("Mass Density")
    axes[col].legend()
    axes[col].grid(True, alpha=0.2)

    fig.suptitle("Plummer Sphere: Learned vs Analytic", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"density_profile.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


def plot_plummer_figure3(
    r: np.ndarray,
    phi_true: np.ndarray,
    phi_learned_shift: np.ndarray,
    rho_analytic: np.ndarray,
    rho_learned: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    phi_slice: np.ndarray,
    rho_slice: np.ndarray,
    *,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    filename: str = "plummer_figure3",
):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.gridspec import GridSpec

    r = np.asarray(r)
    phi_true = np.asarray(phi_true)
    phi_learned_shift = np.asarray(phi_learned_shift)
    rho_analytic = np.asarray(rho_analytic)
    rho_learned = np.asarray(rho_learned)
    x = np.asarray(x)
    y = np.asarray(y)
    phi_slice = np.asarray(phi_slice)
    rho_slice = np.asarray(rho_slice)

    rho0 = float(3.0 / (4.0 * np.pi))
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    rho_true_slice = rho0 * (1.0 + R**2) ** (-2.5)

    phi_display = phi_slice - np.nanmax(phi_slice)
    phi_display = phi_display - np.nanmin(phi_display) + np.nanmin(phi_true)
    phi_display = phi_display + (float(phi_true[0]) - float(np.nanmin(phi_display)))
    rho_ratio = rho_slice / rho0

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    fig = plt.figure(figsize=(10.4, 5.2), dpi=dpi)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.45, 0.08, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 0.9],
        wspace=0.32,
        hspace=0.18,
    )

    ax_phi_r = fig.add_subplot(gs[0, 0])
    ax_rho_r = fig.add_subplot(gs[1, 0], sharex=ax_phi_r)
    ax_resid = fig.add_subplot(gs[2, 0], sharex=ax_phi_r)
    ax_phi = fig.add_subplot(gs[:, 2])
    ax_rho = fig.add_subplot(gs[:, 3])

    ax_phi_r.plot(r, phi_true, color="orange", lw=2.0, label="Truth")
    ax_phi_r.scatter(r, phi_learned_shift, s=5, color="tab:blue", alpha=0.55, label="Estimate", edgecolors="none")
    ax_phi_r.set_xscale("log")
    ax_phi_r.set_ylabel(r"$\Phi$")
    ax_phi_r.legend(loc="upper left", fontsize=8, frameon=True)

    ax_rho_r.plot(r, rho_analytic, color="orange", lw=2.0)
    ax_rho_r.scatter(r, rho_learned, s=5, color="tab:blue", alpha=0.45, edgecolors="none")
    ax_rho_r.set_xscale("log")
    ax_rho_r.set_ylabel(r"$\rho$")

    ax_resid.scatter(r, rho_learned - rho_analytic, s=5, color="tab:green", alpha=0.25, edgecolors="none")
    ax_resid.axhline(0.0, color="0.4", lw=0.8, alpha=0.7)
    ax_resid.set_xscale("log")
    ax_resid.set_xlabel(r"$r$")
    ax_resid.set_ylabel(r"$\rho^*-\rho$")

    for ax in (ax_phi_r, ax_rho_r):
        ax.tick_params(labelbottom=False)
    for ax in (ax_phi_r, ax_rho_r, ax_resid):
        ax.grid(True, alpha=0.12)

    phi_vmin, phi_vmax = np.nanpercentile(phi_display, [1.0, 99.0])
    im_phi = ax_phi.imshow(
        phi_display,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=float(phi_vmin),
        vmax=float(phi_vmax),
        interpolation="nearest",
    )
    ax_phi.set_xlabel(r"$x$")
    ax_phi.set_ylabel(r"$y$")
    ax_phi.set_aspect("equal")

    rho_pos = np.clip(rho_ratio, 1.0e-6, np.inf)
    positive = rho_pos[np.isfinite(rho_pos) & (rho_pos > 0.0)]
    rho_vmin, rho_vmax = np.nanpercentile(positive, [2.0, 99.5])
    im_rho = ax_rho.imshow(
        rho_pos,
        extent=extent,
        origin="lower",
        cmap="viridis",
        norm=colors.LogNorm(vmin=max(float(rho_vmin), 1.0e-6), vmax=max(float(rho_vmax), 1.0e-5)),
        interpolation="nearest",
    )
    levels = [1.0e-3, 1.0e-2, 1.0e-1]
    contour_values = [rho0 * level for level in levels]
    ax_rho.contour(X, Y, rho_true_slice, levels=contour_values, colors="0.25", linewidths=0.7, alpha=0.65)
    ax_rho.set_xlabel(r"$x$")
    ax_rho.tick_params(labelleft=False)
    ax_rho.set_aspect("equal")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, orientation="horizontal", fraction=0.08, pad=0.04, location="top")
    cbar_phi.set_label(r"$\Phi^*$")
    cbar_rho = fig.colorbar(im_rho, ax=ax_rho, orientation="horizontal", fraction=0.08, pad=0.04, location="top")
    cbar_rho.set_label(r"$\rho^*/\rho(r=0)$")

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_potential_density_overview(
    r: np.ndarray,
    phi_learned: np.ndarray,
    rho_learned: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    phi_slice: np.ndarray,
    rho_slice: np.ndarray,
    *,
    ar_learned: Optional[np.ndarray] = None,
    phi_true: Optional[np.ndarray] = None,
    rho_true: Optional[np.ndarray] = None,
    ar_true: Optional[np.ndarray] = None,
    data_xy: Optional[np.ndarray] = None,
    title: str = "Potential / Density Overview",
    density_label: str = r"$\rho=\nabla^2\Phi/(4\pi G)$",
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    filename: str = "potential_density_overview",
):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.gridspec import GridSpec

    r = np.asarray(r)
    phi_learned = np.asarray(phi_learned)
    rho_learned = np.asarray(rho_learned)
    x = np.asarray(x)
    y = np.asarray(y)
    phi_slice = np.asarray(phi_slice)
    rho_slice = np.asarray(rho_slice)
    ar_learned = None if ar_learned is None else np.asarray(ar_learned)
    phi_true = None if phi_true is None else np.asarray(phi_true)
    rho_true = None if rho_true is None else np.asarray(rho_true)
    ar_true = None if ar_true is None else np.asarray(ar_true)
    data_xy = None if data_xy is None else np.asarray(data_xy)

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    fig = plt.figure(figsize=(10.8, 5.3), dpi=dpi)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        width_ratios=[1.55, 0.08, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 0.9],
        wspace=0.34,
        hspace=0.18,
    )

    ax_phi_r = fig.add_subplot(gs[0, 0])
    ax_rho_r = fig.add_subplot(gs[1, 0], sharex=ax_phi_r)
    ax_aux = fig.add_subplot(gs[2, 0], sharex=ax_phi_r)
    ax_phi = fig.add_subplot(gs[:, 2])
    ax_rho = fig.add_subplot(gs[:, 3])

    if phi_true is not None:
        ax_phi_r.plot(r, phi_true, color="orange", lw=2.0, label="Truth")
        ax_phi_r.scatter(r, phi_learned, s=5, color="tab:blue", alpha=0.55, label="Estimate", edgecolors="none")
    else:
        ax_phi_r.plot(r, phi_learned, color="tab:blue", lw=1.7, label="Estimate")
    ax_phi_r.set_xscale("log")
    ax_phi_r.set_ylabel(r"$\Phi^*$")
    ax_phi_r.legend(loc="best", fontsize=8, frameon=True)

    if rho_true is not None:
        ax_rho_r.plot(r, rho_true, color="orange", lw=2.0)
        ax_rho_r.scatter(r, rho_learned, s=5, color="tab:blue", alpha=0.45, edgecolors="none")
    else:
        ax_rho_r.plot(r, rho_learned, color="tab:blue", lw=1.4)
    ax_rho_r.set_xscale("log")
    ax_rho_r.set_ylabel(density_label)

    if rho_true is not None:
        ax_aux.scatter(r, rho_learned - rho_true, s=5, color="tab:green", alpha=0.25, edgecolors="none")
        ax_aux.axhline(0.0, color="0.4", lw=0.8, alpha=0.7)
        ax_aux.set_ylabel(r"$\rho^*-\rho$")
    elif ar_learned is not None:
        if ar_true is not None:
            ax_aux.plot(r, ar_true, color="orange", lw=2.0, label="Truth")
            ax_aux.scatter(r, ar_learned, s=5, color="tab:blue", alpha=0.45, label="Estimate", edgecolors="none")
            ax_aux.legend(loc="best", fontsize=8, frameon=True)
        else:
            ax_aux.plot(r, ar_learned, color="tab:purple", lw=1.4)
        ax_aux.set_ylabel(r"$a_r^*$")
    else:
        ax_aux.plot(r, np.zeros_like(r), color="0.5", lw=0.8)
        ax_aux.set_ylabel("aux")
    ax_aux.set_xscale("log")
    ax_aux.set_xlabel(r"$r$")

    for ax in (ax_phi_r, ax_rho_r):
        ax.tick_params(labelbottom=False)
    for ax in (ax_phi_r, ax_rho_r, ax_aux):
        ax.grid(True, alpha=0.12)

    phi_display = phi_slice - np.nanmedian(phi_slice)
    phi_vmin, phi_vmax = np.nanpercentile(phi_display, [1.0, 99.0])
    if phi_vmin < 0.0 < phi_vmax:
        phi_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=float(phi_vmin), vmax=float(phi_vmax))
        im_phi = ax_phi.imshow(phi_display, extent=extent, origin="lower", cmap="seismic", norm=phi_norm, interpolation="nearest")
    else:
        im_phi = ax_phi.imshow(
            phi_display,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=float(phi_vmin),
            vmax=float(phi_vmax),
            interpolation="nearest",
        )
    ax_phi.set_xlabel(r"$x$")
    ax_phi.set_ylabel(r"$y$")
    ax_phi.set_aspect("equal")

    finite_rho = rho_slice[np.isfinite(rho_slice)]
    has_negative = finite_rho.size > 0 and np.any(finite_rho < 0.0)
    if has_negative:
        rho_v = float(max(np.nanpercentile(np.abs(finite_rho), 99.0), 1.0e-12))
        rho_linthresh = float(max(np.nanpercentile(np.abs(finite_rho), 20.0), rho_v * 1.0e-4, 1.0e-12))
        im_rho = ax_rho.imshow(
            rho_slice,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            norm=colors.SymLogNorm(linthresh=rho_linthresh, vmin=-rho_v, vmax=rho_v),
            interpolation="nearest",
        )
    else:
        rho_pos = np.clip(rho_slice, 1.0e-12, np.inf)
        rho_vmin, rho_vmax = np.nanpercentile(rho_pos, [2.0, 99.5])
        im_rho = ax_rho.imshow(
            rho_pos,
            extent=extent,
            origin="lower",
            cmap="viridis",
            norm=colors.LogNorm(vmin=max(float(rho_vmin), 1.0e-12), vmax=max(float(rho_vmax), 1.0e-11)),
            interpolation="nearest",
        )
    if data_xy is not None and data_xy.size > 0:
        counts, x_edges, y_edges = np.histogram2d(data_xy[:, 0], data_xy[:, 1], bins=80, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        positive_counts = counts[counts > 0]
        if positive_counts.size > 0:
            levels = np.percentile(positive_counts, [50.0, 80.0, 95.0])
            levels = np.unique(levels)
            if levels.size > 0:
                ax_rho.contour(
                    0.5 * (x_edges[:-1] + x_edges[1:]),
                    0.5 * (y_edges[:-1] + y_edges[1:]),
                    counts.T,
                    levels=levels,
                    colors="0.15",
                    linewidths=0.65,
                    alpha=0.7,
                )
    ax_rho.set_xlabel(r"$x$")
    ax_rho.tick_params(labelleft=False)
    ax_rho.set_aspect("equal")

    cbar_phi = fig.colorbar(im_phi, ax=ax_phi, orientation="horizontal", fraction=0.08, pad=0.04, location="top")
    cbar_phi.set_label(r"$\Phi^*-\mathrm{median}(\Phi^*)$")
    cbar_rho = fig.colorbar(im_rho, ax=ax_rho, orientation="horizontal", fraction=0.08, pad=0.04, location="top")
    cbar_rho.set_label(density_label)
    fig.suptitle(title, fontsize=13)

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_auriga_potential_comparison(
    positions: np.ndarray,
    truth: np.ndarray,
    aligned_model: np.ndarray,
    *,
    metrics: Optional[dict[str, float | int]] = None,
    potential_unit: str = r"$(\mathrm{km}\,\mathrm{s}^{-1})^2$",
    radial_bins: int = 24,
    max_scatter_points: int = 50000,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    filename: str = "potential_truth_comparison",
):
    """Compare offset-aligned model potential directly with simulator truth."""
    import matplotlib.pyplot as plt

    positions = np.asarray(positions, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    aligned_model = np.asarray(aligned_model, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if truth.shape != (positions.shape[0],) or aligned_model.shape != truth.shape:
        raise ValueError("truth and aligned_model must have shape (N,).")
    finite = (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(truth)
        & np.isfinite(aligned_model)
    )
    if not np.any(finite):
        raise ValueError("No finite potential comparison rows are available.")
    positions = positions[finite]
    truth = truth[finite]
    aligned_model = aligned_model[finite]
    radius = np.linalg.norm(positions, axis=1)
    residual = aligned_model - truth

    if positions.shape[0] > int(max_scatter_points):
        plot_index = np.linspace(
            0,
            positions.shape[0] - 1,
            int(max_scatter_points),
            dtype=np.int64,
        )
    else:
        plot_index = np.arange(positions.shape[0])

    radial_max = float(max(np.percentile(radius, 99.5), 1.0e-9))
    radial_edges = np.linspace(0.0, radial_max, int(radial_bins) + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    radial_index = np.digitize(radius, radial_edges) - 1

    def radial_stat(values: np.ndarray, percentile: float) -> np.ndarray:
        return np.asarray(
            [
                np.percentile(values[radial_index == index], percentile)
                if np.count_nonzero(radial_index == index) >= 3
                else np.nan
                for index in range(int(radial_bins))
            ]
        )

    truth_median = radial_stat(truth, 50.0)
    model_median = radial_stat(aligned_model, 50.0)
    residual_p16 = radial_stat(residual, 16.0)
    residual_median = radial_stat(residual, 50.0)
    residual_p84 = radial_stat(residual, 84.0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.5),
        dpi=dpi,
        constrained_layout=True,
    )
    hb = axes[0].hexbin(
        truth[plot_index],
        aligned_model[plot_index],
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    comparison_values = np.concatenate([truth[plot_index], aligned_model[plot_index]])
    comparison_lo, comparison_hi = np.percentile(comparison_values, [0.5, 99.5])
    axes[0].plot(
        [comparison_lo, comparison_hi],
        [comparison_lo, comparison_hi],
        color="tab:red",
        lw=1.2,
        ls="--",
    )
    axes[0].set_xlim(comparison_lo, comparison_hi)
    axes[0].set_ylim(comparison_lo, comparison_hi)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(rf"$\Phi_{{\rm truth}}$ {potential_unit}")
    axes[0].set_ylabel(rf"$\Phi_{{\rm model}}+C$ {potential_unit}")
    axes[0].set_title("Pointwise potential comparison")
    fig.colorbar(hb, ax=axes[0], label="log10 point count")

    axes[1].scatter(
        radius[plot_index],
        truth[plot_index],
        s=2,
        alpha=0.08,
        color="black",
        edgecolors="none",
        label="simulator truth",
    )
    axes[1].scatter(
        radius[plot_index],
        aligned_model[plot_index],
        s=2,
        alpha=0.08,
        color="tab:orange",
        edgecolors="none",
        label="model + fitted offset",
    )
    axes[1].plot(radial_centers, truth_median, color="black", lw=2.0)
    axes[1].plot(
        radial_centers,
        model_median,
        color="tab:orange",
        lw=2.0,
        ls="--",
    )
    axes[1].set_xlim(0.0, radial_max)
    axes[1].set_xlabel("r [kpc]")
    axes[1].set_ylabel(rf"$\Phi$ {potential_unit}")
    axes[1].set_title("Radial structure")
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        radius[plot_index],
        residual[plot_index],
        s=2,
        alpha=0.08,
        color="tab:blue",
        edgecolors="none",
    )
    axes[2].fill_between(
        radial_centers,
        residual_p16,
        residual_p84,
        color="tab:blue",
        alpha=0.2,
        label="16–84 percentile",
    )
    axes[2].plot(
        radial_centers,
        residual_median,
        color="tab:blue",
        lw=2.0,
        label="median",
    )
    axes[2].axhline(0.0, color="0.2", lw=0.9, ls="--")
    axes[2].set_xlim(0.0, radial_max)
    axes[2].set_xlabel("r [kpc]")
    axes[2].set_ylabel(rf"$\Phi_{{\rm model}}+C-\Phi_{{\rm truth}}$ {potential_unit}")
    axes[2].set_title("Offset-aligned residual")
    axes[2].legend(fontsize=8)
    if metrics is not None:
        nrmse = metrics.get("normalized_rmse", float("nan"))
        pearson = metrics.get("pearson_r", float("nan"))
        axes[2].text(
            0.03,
            0.97,
            f"NRMSE={float(nrmse):.3g}\nPearson r={float(pearson):.3g}",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Auriga potential recovery (one global additive offset fitted)",
        fontsize=14,
    )

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(
                fig_dir / f"{filename}.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)
        return None
    return fig


def plot_potential_rz_by_phi(
    phi_edges: np.ndarray,
    cylindrical_radius_edges: np.ndarray,
    z_edges: np.ndarray,
    model_potential: np.ndarray,
    *,
    truth_potential: Optional[np.ndarray] = None,
    truth_count: Optional[np.ndarray] = None,
    min_cell_count: int = 1,
    potential_unit: str = r"$(\mathrm{km}\,\mathrm{s}^{-1})^2$",
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    filename: str = "potential_rz_by_phi",
):
    """Plot dense model ``R-z`` slices at azimuth centers, with truth if given."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    phi_edges = np.asarray(phi_edges, dtype=np.float64)
    cylindrical_radius_edges = np.asarray(
        cylindrical_radius_edges,
        dtype=np.float64,
    )
    z_edges = np.asarray(z_edges, dtype=np.float64)
    model_potential = np.asarray(model_potential, dtype=np.float64)
    expected_shape = (
        phi_edges.size - 1,
        cylindrical_radius_edges.size - 1,
        z_edges.size - 1,
    )
    if model_potential.shape != expected_shape:
        raise ValueError(
            f"Expected model_potential shape {expected_shape}, "
            f"got {model_potential.shape}."
        )
    if truth_potential is not None:
        truth_potential = np.asarray(truth_potential, dtype=np.float64)
        if truth_potential.shape != expected_shape:
            raise ValueError(
                f"Expected truth_potential shape {expected_shape}, "
                f"got {truth_potential.shape}."
            )
    if truth_count is not None:
        truth_count = np.asarray(truth_count)
        if truth_count.shape != expected_shape:
            raise ValueError(
                f"Expected truth_count shape {expected_shape}, "
                f"got {truth_count.shape}."
            )
        if truth_potential is not None:
            truth_potential = truth_potential.copy()
            truth_potential[truth_count < int(min_cell_count)] = np.nan

    n_phi = expected_shape[0]
    n_rows = 3 if truth_potential is not None else 1
    shared_values = [model_potential[np.isfinite(model_potential)]]
    if truth_potential is not None:
        shared_values.append(truth_potential[np.isfinite(truth_potential)])
    shared = np.concatenate([values for values in shared_values if values.size])
    if shared.size == 0:
        raise ValueError("Potential slices contain no finite values.")
    potential_vmin, potential_vmax = np.percentile(shared, [1.0, 99.0])
    if potential_vmax <= potential_vmin:
        potential_vmax = potential_vmin + 1.0

    cmap_potential = plt.get_cmap("viridis").with_extremes(bad="#d9d9d9")
    potential_norm = colors.Normalize(
        vmin=float(potential_vmin),
        vmax=float(potential_vmax),
    )
    residual = None
    residual_norm = None
    if truth_potential is not None:
        residual = model_potential - truth_potential
        finite_residual = np.abs(residual[np.isfinite(residual)])
        residual_limit = (
            max(float(np.percentile(finite_residual, 99.0)), 1.0e-12)
            if finite_residual.size
            else 1.0
        )
        residual_norm = colors.TwoSlopeNorm(
            vmin=-residual_limit,
            vcenter=0.0,
            vmax=residual_limit,
        )

    fig, axes = plt.subplots(
        n_rows,
        n_phi,
        figsize=(max(14.0, 3.0 * n_phi), 3.15 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
        dpi=dpi,
    )
    potential_mappable = None
    residual_mappable = None
    for phi_index in range(n_phi):
        phi_left = np.degrees(phi_edges[phi_index])
        phi_right = np.degrees(phi_edges[phi_index + 1])
        if truth_potential is not None:
            potential_mappable = axes[0, phi_index].pcolormesh(
                cylindrical_radius_edges,
                z_edges,
                np.ma.masked_invalid(truth_potential[phi_index].T),
                cmap=cmap_potential,
                norm=potential_norm,
                shading="auto",
            )
            model_row = 1
            residual_row = 2
        else:
            model_row = 0
            residual_row = None
        potential_mappable = axes[model_row, phi_index].pcolormesh(
            cylindrical_radius_edges,
            z_edges,
            model_potential[phi_index].T,
            cmap=cmap_potential,
            norm=potential_norm,
            shading="auto",
        )
        if residual_row is not None and residual is not None:
            residual_mappable = axes[residual_row, phi_index].pcolormesh(
                cylindrical_radius_edges,
                z_edges,
                np.ma.masked_invalid(residual[phi_index].T),
                cmap="coolwarm",
                norm=residual_norm,
                shading="auto",
            )
        axes[0, phi_index].set_title(
            rf"${phi_left:.0f}^\circ\leq\phi<{phi_right:.0f}^\circ$"
        )
        axes[-1, phi_index].set_xlabel("R [kpc]")

    if truth_potential is not None:
        axes[0, 0].set_ylabel("simulator median\nz [kpc]")
        axes[1, 0].set_ylabel("model at bin center\nz [kpc]")
        axes[2, 0].set_ylabel("model − simulator\nz [kpc]")
    else:
        axes[0, 0].set_ylabel("model\nz [kpc]")
    fig.colorbar(
        potential_mappable,
        ax=axes[: (2 if truth_potential is not None else 1), :].ravel().tolist(),
        label=rf"$\Phi+C$ {potential_unit}",
        shrink=0.86,
    )
    if residual_mappable is not None:
        fig.colorbar(
            residual_mappable,
            ax=axes[-1, :].ravel().tolist(),
            label=rf"$\Delta\Phi$ {potential_unit}",
            shrink=0.86,
        )
    coverage_note = (
        f"; simulator cells require ≥{int(min_cell_count)} particles"
        if truth_potential is not None
        else ""
    )
    unsupported_note = (
        " (gray = unsupported truth)"
        if truth_potential is not None
        else ""
    )
    fig.suptitle(
        "Auriga potential R-z slices by azimuth"
        + coverage_note
        + unsupported_note,
        fontsize=14,
    )

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(
                fig_dir / f"{filename}.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)
        return None
    return fig


def plot_laplacian_density_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    density: np.ndarray,
    *,
    density_label: str = r"$\nabla^2\Phi/(4\pi G)$",
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
    filename: str = "laplacian_density_diagnostics",
):
    """Expose signed Laplacian density instead of clipping it for log display."""
    import matplotlib.pyplot as plt
    from matplotlib import colors

    from dpjax.physics.units import summarize_density_sign

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if density.shape != (y.size, x.size):
        raise ValueError(
            f"Expected density shape ({y.size}, {x.size}), got {density.shape}."
        )
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise ValueError("density contains no finite values.")
    summary = summarize_density_sign(density)
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    absolute = np.abs(finite)
    scale = max(float(np.percentile(absolute, 99.0)), 1.0e-12)
    nonzero = absolute[absolute > 0.0]
    linthresh = (
        max(float(np.percentile(nonzero, 10.0)), scale * 1.0e-4, 1.0e-12)
        if nonzero.size
        else scale * 1.0e-4
    )
    signed_norm = colors.SymLogNorm(
        linthresh=linthresh,
        vmin=-scale,
        vmax=scale,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.4),
        constrained_layout=True,
        dpi=dpi,
    )
    im_signed = axes[0].imshow(
        density,
        extent=extent,
        origin="lower",
        cmap="coolwarm",
        norm=signed_norm,
        interpolation="nearest",
    )
    axes[0].set_title("Signed raw density (symmetric log)")
    fig.colorbar(im_signed, ax=axes[0], label=density_label)

    positive = np.ma.masked_less_equal(density, 0.0)
    positive_values = finite[finite > 0.0]
    positive_cmap = plt.get_cmap("magma").with_extremes(bad="#d9d9d9")
    if positive_values.size:
        positive_vmin, positive_vmax = np.percentile(
            positive_values,
            [2.0, 99.0],
        )
        positive_vmin = max(float(positive_vmin), 1.0e-12)
        positive_vmax = max(float(positive_vmax), positive_vmin * 1.01)
    else:
        positive_vmin, positive_vmax = 1.0e-12, 1.0e-11
    im_positive = axes[1].imshow(
        positive,
        extent=extent,
        origin="lower",
        cmap=positive_cmap,
        norm=colors.LogNorm(vmin=positive_vmin, vmax=positive_vmax),
        interpolation="nearest",
    )
    axes[1].set_title("Positive density only (gray = non-positive)")
    fig.colorbar(im_positive, ax=axes[1], label=density_label)

    sign_image = np.full(density.shape, np.nan)
    sign_image[np.isfinite(density) & (density < 0.0)] = -1.0
    sign_image[np.isfinite(density) & (density == 0.0)] = 0.0
    sign_image[np.isfinite(density) & (density > 0.0)] = 1.0
    sign_cmap = colors.ListedColormap(["#3b4cc0", "#eeeeee", "#f4987a"])
    sign_norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], sign_cmap.N)
    im_sign = axes[2].imshow(
        sign_image,
        extent=extent,
        origin="lower",
        cmap=sign_cmap,
        norm=sign_norm,
        interpolation="nearest",
    )
    axes[2].set_title(
        "Laplacian sign\n"
        f"negative={summary['negative_fraction']:.1%}, "
        f"non-positive={summary['nonpositive_fraction']:.1%}"
    )
    sign_colorbar = fig.colorbar(im_sign, ax=axes[2], ticks=[-1.0, 0.0, 1.0])
    sign_colorbar.ax.set_yticklabels(["negative", "zero", "positive"])
    for ax in axes:
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_aspect("equal")
    fig.suptitle(
        "Total-density diagnostic from the raw potential Laplacian",
        fontsize=14,
    )

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(
                fig_dir / f"{filename}.{fmt}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(fig)
        return None
    return fig


# ── 7. CBE residual spatial map ──────────────────────────────────────────

def plot_residual_spatial(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    residual: np.ndarray,
    *,
    fig_dir: Optional[str | Path] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """2D hexbin map of CBE residual magnitude in (x,y) and (x,z) projections.

    Parameters
    ----------
    x, y, z : (N,) physical spatial coordinates
    residual : (N,) CBE residual values
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    abs_r = np.abs(residual)
    vmax = float(np.percentile(abs_r, 99))
    vmax = max(vmax, 1e-6)

    fig, (ax1, ax2, cax) = plt.subplots(
        1, 3, figsize=(12, 4.5), dpi=dpi,
        gridspec_kw=dict(width_ratios=[1, 1, 0.04]),
    )

    hb1 = ax1.hexbin(
        x, y, C=abs_r, reduce_C_function=np.median,
        gridsize=50, cmap="inferno",
        norm=colors.LogNorm(vmin=max(vmax * 1e-3, 1e-8), vmax=vmax),
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("CBE residual |r|  (x-y)")
    ax1.set_aspect("equal")

    hb2 = ax2.hexbin(
        x, z, C=abs_r, reduce_C_function=np.median,
        gridsize=50, cmap="inferno",
        norm=colors.LogNorm(vmin=max(vmax * 1e-3, 1e-8), vmax=vmax),
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("CBE residual |r|  (x-z)")
    ax2.set_aspect("equal")

    cb = fig.colorbar(hb2, cax=cax)
    cb.set_label("median |residual|")

    fig.suptitle("CBE Residual Spatial Distribution", fontsize=13)
    fig.tight_layout()

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"residual_spatial.{fmt}", dpi=dpi)
        plt.close(fig)
        return None
    return fig


# ── Auriga DF ensemble diagnostics ───────────────────────────────────────


def plot_auriga_df_ensemble(
    metrics_json: str | Path,
    diagnostics_npz: str | Path,
    fig_dir: Optional[str | Path] = None,
    *,
    run_dirs: Optional[list[str | Path]] = None,
    fig_fmt: Iterable[str] = ("png",),
    dpi: int = 150,
):
    """Render Halo DF diagnostics from ``eval_auriga_df`` outputs.

    Consumes the JSON + NPZ written by the Auriga DF evaluation workflow and
    produces radial density, local spatial density, conditional velocity, and
    score figures.  Distribution and Stein diagnostics work with one DF;
    score-repeatability plots are added when at least two models are present.

    - ``density_profile.png`` – radial stellar-tracer density
    - ``df_spatial_rz_by_phi.png`` – data/model/log-ratio R-z wedges
    - ``velocity_marginals_by_{r,theta,phi}.png`` – requested histogram grids
    - ``score_consistency.png`` – ensemble-only score repeatability
    - ``score_per_dim_hist.png`` – per-dimension score distribution across seeds

    Parameters
    ----------
    metrics_json, diagnostics_npz : paths produced by ``eval_auriga_df``
    fig_dir : output directory; defaults to the diagnostics NPZ parent
    run_dirs : optional list of per-seed training run directories, used to
        also overlay the training curves in one of the panels
    """
    import json

    import matplotlib.pyplot as plt
    from matplotlib import colors

    metrics_json = Path(metrics_json)
    diagnostics_npz = Path(diagnostics_npz)
    if fig_dir is None:
        fig_dir = diagnostics_npz.parent
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = json.loads(metrics_json.read_text(encoding="utf-8"))
    d = np.load(diagnostics_npz, allow_pickle=False)
    result = {"fig_dir": str(fig_dir)}

    edges = d["radial_edges"]
    r_centers = np.sqrt(edges[:-1] * edges[1:])
    ref = d["reference_density"]
    mdl = d["model_density"]
    mdl_by_model = (
        d["model_density_by_model"]
        if "model_density_by_model" in d.files
        else mdl[None, :]
    )
    model_labels = (
        [str(value) for value in d["model_labels"]]
        if "model_labels" in d.files
        else [f"model_{index}" for index in range(mdl_by_model.shape[0])]
    )
    dm = m["density_profile"]

    # ── density profile ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].loglog(
        r_centers,
        ref,
        "k-o",
        lw=2,
        ms=5,
        label="data: stellar tracer mass",
    )
    for model_index, density in enumerate(mdl_by_model):
        axes[0].loglog(
            r_centers,
            density,
            lw=1.0,
            alpha=0.45,
            label=model_labels[model_index],
        )
    axes[0].loglog(
        r_centers,
        mdl,
        "--",
        color="tab:orange",
        lw=2.2,
        marker="s",
        ms=5,
        label="DF median" if mdl_by_model.shape[0] > 1 else "DF model",
    )
    axes[0].set_xlabel("r [kpc]")
    axes[0].set_ylabel(r"normalized stellar-tracer density [kpc$^{-3}$]")
    axes[0].set_title(
        f"Halo12 DF spatial marginal "
        f"(log10_RMSE={dm['log10_rmse_dex']:.3f} dex)"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")

    rel = np.abs(mdl - ref) / np.maximum(ref, 1e-12)
    axes[1].semilogx(r_centers, rel, "s-", color="tab:red", lw=1.5)
    axes[1].axhline(0.30, ls="--", color="k", alpha=0.5, label="p90 gate 0.30")
    axes[1].axhline(
        dm["median_fractional_error"], ls=":", color="tab:blue",
        label=f"median={dm['median_fractional_error']:.3f}",
    )
    axes[1].set_xlabel("r [kpc]")
    axes[1].set_ylabel(r"$|\rho_{model}-\rho_{data}|/\rho_{data}$")
    axes[1].set_title(f"Fractional error (p90={dm['p90_fractional_error']:.3f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in fig_fmt:
        fig.savefig(fig_dir / f"density_profile.{fmt}", dpi=dpi)
    plt.close(fig)
    result["density_profile"] = str(fig_dir / "density_profile.png")

    # ── cylindrical R-z density in azimuth wedges ──────────────────
    if "spatial_reference_density" in d.files:
        phi_edges = d["spatial_phi_edges"]
        cylindrical_radius_edges = d["spatial_r_edges"]
        z_edges = d["spatial_z_edges"]
        reference_density = d["spatial_reference_density"]
        model_density = d["spatial_model_median_density"]
        reference_count = d["spatial_reference_count"]
        model_count = np.median(d["spatial_model_count"], axis=0)
        min_cell_count = int(d["spatial_min_cell_count"])
        n_phi = phi_edges.size - 1

        positive = np.concatenate(
            [
                reference_density[reference_density > 0],
                model_density[model_density > 0],
            ]
        )
        density_vmin, density_vmax = np.percentile(positive, [5.0, 99.5])
        if density_vmax <= density_vmin:
            density_vmax = density_vmin * 10.0
        density_norm = colors.LogNorm(
            vmin=max(float(density_vmin), np.finfo(float).tiny),
            vmax=float(density_vmax),
        )

        valid_ratio = (
            (reference_count >= min_cell_count)
            & (model_count >= min_cell_count)
            & (reference_density > 0)
            & (model_density > 0)
        )
        log_ratio = np.full_like(reference_density, np.nan)
        log_ratio[valid_ratio] = np.log10(
            model_density[valid_ratio] / reference_density[valid_ratio]
        )
        finite_ratio = np.abs(log_ratio[np.isfinite(log_ratio)])
        ratio_limit = (
            max(float(np.percentile(finite_ratio, 98.0)), 0.1)
            if finite_ratio.size
            else 1.0
        )
        ratio_norm = colors.TwoSlopeNorm(
            vmin=-ratio_limit,
            vcenter=0.0,
            vmax=ratio_limit,
        )

        fig, axes = plt.subplots(
            3,
            n_phi,
            figsize=(max(14.0, 3.0 * n_phi), 10.0),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(3, n_phi)
        density_mappable = None
        ratio_mappable = None
        for phi_index in range(n_phi):
            phi_left = np.degrees(phi_edges[phi_index])
            phi_right = np.degrees(phi_edges[phi_index + 1])
            data_image = np.ma.masked_less_equal(
                reference_density[phi_index].T,
                0.0,
            )
            model_image = np.ma.masked_less_equal(
                model_density[phi_index].T,
                0.0,
            )
            density_mappable = axes[0, phi_index].pcolormesh(
                cylindrical_radius_edges,
                z_edges,
                data_image,
                cmap="magma",
                norm=density_norm,
                shading="auto",
            )
            axes[1, phi_index].pcolormesh(
                cylindrical_radius_edges,
                z_edges,
                model_image,
                cmap="magma",
                norm=density_norm,
                shading="auto",
            )
            ratio_mappable = axes[2, phi_index].pcolormesh(
                cylindrical_radius_edges,
                z_edges,
                np.ma.masked_invalid(log_ratio[phi_index].T),
                cmap="coolwarm",
                norm=ratio_norm,
                shading="auto",
            )
            axes[0, phi_index].set_title(
                rf"${phi_left:.0f}^\circ\leq\phi<{phi_right:.0f}^\circ$"
            )
            axes[2, phi_index].set_xlabel("R [kpc]")
        axes[0, 0].set_ylabel("data\nz [kpc]")
        axes[1, 0].set_ylabel("DF model\nz [kpc]")
        axes[2, 0].set_ylabel(r"$\log_{10}(\rho_{\rm DF}/\rho_\star)$" "\nz [kpc]")
        fig.colorbar(
            density_mappable,
            ax=axes[:2, :].ravel().tolist(),
            label=r"normalized stellar-tracer density [kpc$^{-3}$]",
            shrink=0.85,
        )
        fig.colorbar(
            ratio_mappable,
            ax=axes[2, :].ravel().tolist(),
            label="log10 density ratio",
            shrink=0.85,
        )
        fig.suptitle(
            "Halo12 stellar-tracer density by azimuth wedge "
            f"(cells require at least {min_cell_count} samples)",
            fontsize=14,
        )
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"df_spatial_rz_by_phi.{fmt}", dpi=dpi)
        plt.close(fig)
        result["df_spatial_rz_by_phi"] = str(
            fig_dir / "df_spatial_rz_by_phi.png"
        )

    # ── requested conditional spherical-velocity marginals ─────────
    if "conditional_velocity_edges" in d.files:
        velocity_edges = d["conditional_velocity_edges"]
        velocity_labels = (r"$v_r$", r"$v_\theta$", r"$v_\phi$")
        line_colors = plt.get_cmap("tab10")
        for coordinate_name in ("r", "theta", "phi"):
            key_prefix = f"conditional_{coordinate_name}_"
            if f"{key_prefix}reference_hist" not in d.files:
                continue
            coordinate_edges = d[f"{key_prefix}edges"]
            reference_hist = d[f"{key_prefix}reference_hist"]
            model_hist = d[f"{key_prefix}model_hist"]
            reference_effective_count = d[
                f"{key_prefix}reference_effective_count"
            ]
            model_count_conditional = d[f"{key_prefix}model_count"]
            wasserstein = d[f"{key_prefix}wasserstein"]
            n_rows = coordinate_edges.size - 1

            fig, axes = plt.subplots(
                n_rows,
                3,
                figsize=(13.5, max(3.5, 1.9 * n_rows)),
                sharex="col",
                squeeze=False,
                constrained_layout=True,
            )
            for row in range(n_rows):
                for velocity_index in range(3):
                    ax = axes[row, velocity_index]
                    ax.stairs(
                        reference_hist[row, velocity_index],
                        velocity_edges[velocity_index],
                        color="black",
                        lw=1.8,
                        label="data",
                    )
                    for model_index in range(model_hist.shape[0]):
                        ax.stairs(
                            model_hist[
                                model_index,
                                row,
                                velocity_index,
                            ],
                            velocity_edges[velocity_index],
                            color=line_colors(model_index % 10),
                            lw=1.1,
                            alpha=0.85,
                            label=model_labels[model_index],
                        )
                    w1_values = wasserstein[:, row, velocity_index]
                    finite_w1 = w1_values[np.isfinite(w1_values)]
                    if finite_w1.size:
                        ax.text(
                            0.98,
                            0.93,
                            f"W1={np.median(finite_w1):.2g}",
                            ha="right",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=7,
                        )
                    ax.grid(True, alpha=0.2)
                    if row == 0:
                        ax.set_title(velocity_labels[velocity_index])
                    if row == n_rows - 1:
                        ax.set_xlabel(r"velocity [km s$^{-1}$]")

                left = coordinate_edges[row]
                right = coordinate_edges[row + 1]
                if coordinate_name == "r":
                    interval = f"{left:.2g} ≤ r < {right:.2g} kpc"
                else:
                    interval = (
                        f"{np.degrees(left):.0f}° ≤ {coordinate_name} "
                        f"< {np.degrees(right):.0f}°"
                    )
                median_model_count = int(
                    np.median(model_count_conditional[:, row])
                )
                axes[row, 0].set_ylabel(
                    f"{interval}\nPDF\n"
                    f"N_eff={reference_effective_count[row]:.0f}, "
                    f"N_m≈{median_model_count}"
                )
            axes[0, -1].legend(loc="upper left", fontsize=7)
            fig.suptitle(
                f"Velocity marginals conditioned only on {coordinate_name}; "
                "all other coordinates are marginalized",
                fontsize=13,
            )
            filename = f"velocity_marginals_by_{coordinate_name}"
            for fmt in fig_fmt:
                fig.savefig(fig_dir / f"{filename}.{fmt}", dpi=dpi)
            plt.close(fig)
            result[filename] = str(fig_dir / f"{filename}.png")

    # ── score consistency ───────────────────────────────────────────
    scores = d["scores"]  # (n_models, N, 6)
    n_models, n_pts, dim = scores.shape
    if len(model_labels) < n_models:
        model_labels = [f"model_{i}" for i in range(n_models)]
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    se = m.get("score_ensemble")

    if n_models >= 2 and se is not None:
        def _cos(a, b):
            na = np.linalg.norm(a, axis=-1)
            nb = np.linalg.norm(b, axis=-1)
            return np.sum(a * b, axis=-1) / (na * nb + 1e-12)

        pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pairs.append(_cos(scores[i], scores[j]))
        all_pairs = np.concatenate(pairs)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(
            labels,
            se["relative_mad_by_dimension"],
            color="tab:purple",
            alpha=0.85,
        )
        axes[0].axhline(
            0.20,
            ls="--",
            color="k",
            alpha=0.5,
            label="per-dim gate 0.20",
        )
        axes[0].axhline(
            0.10,
            ls=":",
            color="k",
            alpha=0.5,
            label="median gate 0.10",
        )
        axes[0].set_ylabel("6D relative MAD")
        axes[0].set_title(
            f"Per-dimension MAD "
            f"(median={se['median_relative_mad']:.3f})"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(all_pairs, bins=60, color="tab:green", alpha=0.8)
        axes[1].axvline(
            0.95,
            ls="--",
            color="k",
            label=(
                "median gate 0.95 "
                f"(got {se['pairwise_cosine_median']:.3f})"
            ),
        )
        axes[1].axvline(
            0.80,
            ls=":",
            color="k",
            label=f"p10 gate 0.80 (got {se['pairwise_cosine_p10']:.3f})",
        )
        axes[1].set_xlabel(
            r"pairwise $\cos(\nabla\log f_i, \nabla\log f_j)$"
        )
        axes[1].set_ylabel("count")
        axes[1].set_title("Ensemble score consistency")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in fig_fmt:
            fig.savefig(fig_dir / f"score_consistency.{fmt}", dpi=dpi)
        plt.close(fig)
        result["score_consistency"] = str(
            fig_dir / "score_consistency.png"
        )

    # ── per-dimension score distribution ───────────────────────────
    rng = np.random.default_rng(0)
    idx = rng.choice(n_pts, size=min(2000, n_pts), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for k in range(6):
        for mi in range(n_models):
            axes[k].hist(
                scores[mi, idx, k], bins=40, histtype="step", lw=1.2,
                label=model_labels[mi],
            )
        axes[k].set_xlabel(f"score[{labels[k]}]")
        axes[k].set_ylabel("count")
        axes[k].legend(fontsize=8)
        axes[k].grid(True, alpha=0.3)
    plt.suptitle(
        "Per-dimension physical-score distribution"
        + (" across models" if n_models > 1 else "")
    )
    plt.tight_layout()
    for fmt in fig_fmt:
        fig.savefig(fig_dir / f"score_per_dim_hist.{fmt}", dpi=dpi)
    plt.close(fig)
    result["score_per_dim_hist"] = str(
        fig_dir / "score_per_dim_hist.png"
    )

    # ── optional training curves overlay ───────────────────────────
    if run_dirs:
        from .training_curves import plot_df_training_ensemble
        plot_df_training_ensemble(run_dirs, save_dir=fig_dir, dpi=dpi)

    return result
