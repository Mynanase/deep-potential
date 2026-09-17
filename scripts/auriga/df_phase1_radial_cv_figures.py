#!/usr/bin/env python
r"""Phase-1 DF audit follow-up: figures for the radial conditional-velocity review.

Reads ONLY the persisted products of scripts/auriga/df_phase1_radial_cv_metrics.py
(runs/halo12-radial-cv-20260917/rcv_arrays.npz and the rcv_*.csv).  No model is
reloaded and no metric is recomputed, so the figures can be regenerated without a
GPU and a plotted number cannot drift from the reviewed number.

Figures, PDF (for \includegraphics) plus SVG (for preview), each built at its
final printed width and never rescaled:

  fig_dist_vr / _vth / _vT   truth vs model conditional distribution, 6 radial panels
  fig_w1_radial              W1 per radial bin and component, with intervals
  fig_mean_radial            conditional mean per radial bin
  fig_dispersion_radial      centred sigma per radial bin
  fig_correlation_radial     the three velocity-pair correlations per radial bin
  fig_residual_radial        model - data residuals of mean, sigma, correlation

Conventions: black is the held-out data, models take Okabe-Ito colours AND
distinct line styles so the comparison survives greyscale and printing, one
public legend per figure, no axes titles (the report caption carries the claim),
every panel tagged in-axes with its own subject, and every limit derived from the
persisted numbers -- this script prints those measured ranges to stdout so the
report can quote them.

Run from the repo root:  python scripts/auriga/df_phase1_radial_cv_figures.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, WIDE, save, use_style

import matplotlib.pyplot as plt

MODELS = ["w128", "w512", "w1024"]
COLOR = {"w128": PALETTE["blue"], "w512": PALETTE["orange"], "w1024": PALETTE["green"]}
LS = {"w128": "-", "w512": "--", "w1024": ":"}
MARK = {"w128": "o", "w512": "s", "w1024": "^"}
COMP = ["vr", "vth", "vT"]
COMP_MATH = {"vr": r"$v_r$", "vth": r"$v_\theta$", "vT": r"$v_\phi$"}
PAIRS = [("vr", "vth"), ("vr", "vT"), ("vth", "vT")]
PAIR_MATH = {("vr", "vth"): r"$v_r, v_\theta$", ("vr", "vT"): r"$v_r, v_\phi$",
             ("vth", "vT"): r"$v_\theta, v_\phi$"}
VBIN = 10.0      # histogram bin width in km/s, identical in every panel of a figure


def tag(ax, text):
    """Name the panel's own subject inside the axes (there are no axes titles)."""
    ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=6.5)


def display_window(a):
    """Per-component display window in km/s, from the persisted samples.

    The window is the pooled 0.05-99.95% range of the held-out data and every
    model's draws, rounded outward to 10 km/s.  It is a display choice only: W1
    and every other reported number uses all samples, and the histograms divide
    by each series' whole weight rather than renormalising inside the window.
    Deriving it here (rather than trusting a stored window) keeps the unit
    contract in one place, and the check below fails loudly if the arrays are not
    in km/s after all.
    """
    assert a["velocity_unit"].item() == "km/s", "the persisted velocities must be km/s"
    win = {}
    for c, name in enumerate(COMP):
        v = np.concatenate([a["v_true"][:, c]]
                           + [a[f"gen_k4_{m}"][:, :, c].reshape(-1) for m in MODELS])
        lo, hi = np.nanpercentile(v, [0.05, 99.95])
        win[name] = [float(np.floor(lo / 10.0) * 10.0), float(np.ceil(hi / 10.0) * 10.0)]
        span = max(abs(win[name][0]), abs(win[name][1]))
        scale = float(np.nanpercentile(np.abs(a["v_true"][:, c]), 99.9))
        assert 0.3 * span < scale < 3.0 * span, (
            f"{name}: the data 99.9% scale {scale:.1f} and the window {span:.0f} are"
            f" not in the same unit")
    return win


def write_display_table(a, win, run_dir):
    """rcv_display.csv: the drawn range and the mass each series puts outside it.

    The plan requires the out-of-window mass fraction whenever the display range
    cuts the velocity tails, reported separately for every series so the reader
    can see how much of each distribution the figure does not show.
    """
    ir, w, r_edges = a["ir_strict"], a["w_strict"].astype(np.float64), a["r_edges"]
    k = int(a["k_draws"])
    rows = []
    for b in range(len(r_edges) - 1):
        mb = ir == b
        for c, name in enumerate(COMP):
            lo, hi = float(win[name][0]), float(win[name][1])
            series = [("data", a["v_true"][mb, c], w[mb])]
            for m in MODELS:
                g = a[f"gen_k4_{m}"][mb, :, c].reshape(-1)
                series.append((m, g, np.repeat(w[mb], g.size // int(mb.sum())) / k))
            for sname, v, wgt in series:
                ok = np.isfinite(v)
                tot = float(wgt[ok].sum())
                outside = float(wgt[ok & ((v < lo) | (v > hi))].sum())
                rows.append(dict(bin=b, r_lo_kpc=float(r_edges[b] * 10),
                                 r_hi_kpc=float(r_edges[b + 1] * 10), comp=name,
                                 series=sname, window_lo_kms=lo, window_hi_kms=hi,
                                 n_samples=int(ok.sum()), mass_frac=float(tot / w.sum()),
                                 frac_out_window=float(outside / tot) if tot > 0 else float("nan")))
    pd.DataFrame(rows).to_csv(run_dir / "rcv_display.csv", index=False)
    worst = max(r["frac_out_window"] for r in rows)
    print("display window: " + ", ".join(f"{c} {v[0]:.0f}..{v[1]:.0f} km/s"
                                         for c, v in win.items())
          + f"; largest out-of-window mass fraction over series/bins: {worst:.2e}", flush=True)
    return worst


def one_legend(axes, title, ncols=4):
    """Exactly one public legend per figure, below the panels."""
    handles, seen = [], set()
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append((h, l))
    if not handles:
        return
    fig = np.ravel(axes)[0].figure
    fig.legend([h for h, _ in handles], [l for _, l in handles],
               loc="outside lower center", ncols=ncols, title=title, title_fontsize=7)


def xcenter(d):
    return 0.5 * (d["r_lo_kpc"] + d["r_hi_kpc"])


def data_curve(ax, d, col):
    """The held-out data curve, drawn once per panel in black."""
    ax.plot(xcenter(d), d[col], color="black", lw=1.3, marker="o", ms=3.0,
            label="held-out data (mass-weighted)")


def model_curves(ax, d, col, with_ci=True):
    for m in MODELS:
        dm = d[d["model"] == m].sort_values("bin")
        if not len(dm):
            continue
        err = None
        if with_ci and f"{col}_lo" in dm.columns:
            err = np.vstack([dm[col] - dm[f"{col}_lo"], dm[f"{col}_hi"] - dm[col]])
        ax.errorbar(xcenter(dm), dm[col], yerr=err, color=COLOR[m], ls=LS[m],
                    marker=MARK[m], ms=3.0, lw=1.0, capsize=2, label=m)


# ----------------------------------------------------------------------
# 1. conditional distribution: truth vs model, six radial panels
# ----------------------------------------------------------------------
def fig_dist(a, win, out_dir):
    ir, w = a["ir_strict"], a["w_strict"].astype(np.float64)
    v_true, r_edges = a["v_true"], a["r_edges"]
    ranges = {}
    for c, name in enumerate(COMP):
        lo, hi = float(win[name][0]), float(win[name][1])
        bins = np.arange(lo, hi + 1e-6, VBIN)
        fig, axes = plt.subplots(2, 3, figsize=(WIDE, WIDE * 0.60), sharex=True,
                                 sharey=True, layout="constrained")
        dens, n_pos = {}, {}
        for b in range(len(r_edges) - 1):
            mb = ir == b
            n_pos[b] = int(mb.sum())
            h, _ = np.histogram(v_true[mb, c], bins=bins, weights=w[mb])
            dens[("data", b)] = h / (VBIN * w[mb].sum())
            for m in MODELS:
                g = a[f"gen_k4_{m}"][mb, :, c].reshape(-1)
                wg = np.repeat(w[mb], g.size // n_pos[b])
                h, _ = np.histogram(g, bins=bins, weights=wg)
                dens[(m, b)] = h / (VBIN * wg.sum())
        ymax = 1.12 * max(float(np.nanmax(v)) for v in dens.values())
        for b, ax in enumerate(np.ravel(axes)):
            ax.step(bins[:-1], dens[("data", b)], where="post", color="black", lw=1.2,
                    label="held-out data (mass-weighted)")
            for m in MODELS:
                ax.step(bins[:-1], dens[(m, b)], where="post", color=COLOR[m],
                        ls=LS[m], lw=0.9, label=m)
            ax.set_ylim(0.0, ymax)
            ax.set_xlim(bins[0], bins[-1])
            tag(ax, f"{r_edges[b] * 10:.0f}–{r_edges[b + 1] * 10:.0f} kpc  (n={n_pos[b]})")
            if b % 3 == 0:
                ax.set_ylabel(r"$p(v)$  [(km s$^{-1}$)$^{-1}$]")
            if b // 3 == 1:
                ax.set_xlabel(f"{COMP_MATH[name]}  [km s$^{{-1}}$]")
        one_legend(axes, f"conditional velocity {COMP_MATH[name]}: held-out data vs "
                         f"model", ncols=4)
        save(fig, str(out_dir / f"fig_dist_{name}"))
        ranges[name] = (float(bins[0]), float(bins[-1]), ymax / 1.12)
        del dens
    return ranges


# ----------------------------------------------------------------------
# 2. per-bin curves, one panel per velocity component or component pair
# ----------------------------------------------------------------------
def curve_fig(out_dir, stem, d, col, truth_col, ylab, panels, legend, ratio=0.44):
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT, TEXT * ratio),
                             sharey=True, layout="constrained")
    for i, (ax, key) in enumerate(zip(np.ravel(axes), panels)):
        dp = d[d["panel"] == key]
        ax.set_xlabel("r  [kpc]")
        if i == 0:
            ax.set_ylabel(ylab)
        tag(ax, key)
        if truth_col is not None and len(dp):
            data_curve(ax, dp[dp["model"] == MODELS[0]].sort_values("bin"), truth_col)
        model_curves(ax, dp, col)
    one_legend(axes, legend)
    save(fig, str(out_dir / stem))


def fig_curves(stats, corr, ref, out_dir):
    st = stats[stats["sample"] == "k4"].copy()
    st["panel"] = st["comp"].map(COMP_MATH)
    c = corr[corr["sample"] == "k4"].copy()
    c["panel"] = c["pair"].map(lambda p: PAIR_MATH[tuple(p.split("|"))])
    rf = ref.copy()
    rf["panel"] = rf["comp_key"].map(COMP_MATH)

    curve_fig(out_dir, "fig_mean_radial", st, "mean_gen", "mean_true",
              "mean  [km s$^{-1}$]", [COMP_MATH[x] for x in COMP],
              "conditional mean per radial bin, K=4 (bars: 95% intervals)")
    curve_fig(out_dir, "fig_dispersion_radial", st, "sigma_gen", "sigma_true",
              r"$\sigma$  [km s$^{-1}$]", [COMP_MATH[x] for x in COMP],
              r"centred $\sigma$ per radial bin, K=4 (bars: 95% intervals)")
    curve_fig(out_dir, "fig_correlation_radial", c, "rho_gen", "rho_true",
              r"mass-weighted Pearson $\rho$", [PAIR_MATH[x] for x in PAIRS],
              "velocity-pair correlation, K=4 (bars: 95% intervals)")

    # W1 gets its own figure: the truth half-split reference scale is part of it
    fig, axes = plt.subplots(1, len(COMP), figsize=(TEXT, TEXT * 0.44), sharey=True,
                             layout="constrained")
    for i, (ax, name) in enumerate(zip(np.ravel(axes), COMP)):
        d = st[st["comp"] == name]
        r = rf[rf["comp_key"] == name]
        ax.set_xlabel("r  [kpc]")
        if i == 0:
            ax.set_ylabel("W1  [km s$^{-1}$]")
        tag(ax, COMP_MATH[name])
        ax.errorbar(xcenter(r), r["truth_half_w1_kms"], yerr=r["truth_half_w1_sd_kms"],
                    fmt="none", ecolor=MUTED, elinewidth=2.6, capsize=0,
                    label="truth half-split (finite-sample reference scale)")
        ax.plot(xcenter(r), r["truth_half_w1_kms"], ls="none", marker="_", ms=9,
                color=BASELINE)
        model_curves(ax, d, "w1")
    one_legend(axes, "W1 per radial bin, K=4 (bars: 95% intervals)")
    save(fig, str(out_dir / "fig_w1_radial"))


# ----------------------------------------------------------------------
# 3. residual panels: model minus data
# ----------------------------------------------------------------------
def fig_residual(stats, corr, out_dir):
    st = stats[stats["sample"] == "k4"]
    cc = corr[corr["sample"] == "k4"]
    rows = [("dmean", r"$\Delta$ mean  [km s$^{-1}$]", False),
            ("dsigma", r"$\Delta\sigma$  [km s$^{-1}$]", False),
            ("drho", r"$\Delta$ Pearson $\rho$", True)]
    fig, axes = plt.subplots(3, len(COMP), figsize=(WIDE, WIDE * 0.80),
                             sharex="col", sharey="row", layout="constrained")
    for ri, (col, ylab, is_corr) in enumerate(rows):
        src = cc if is_corr else st
        for k in range(len(COMP)):
            ax = axes[ri, k]
            if is_corr:
                key = "|".join(PAIRS[k])
                d0 = src[src["pair"] == key]
                tag(ax, PAIR_MATH[PAIRS[k]])
            else:
                name = COMP[k]
                d0 = src[src["comp"] == name]
                tag(ax, COMP_MATH[name])
            if ri == len(rows) - 1:
                ax.set_xlabel("r  [kpc]")
            if k == 0:
                ax.set_ylabel(ylab)
            ax.axhline(0.0, color=BASELINE, lw=0.9, zorder=1)
            model_curves(ax, d0, col)
            err = np.nanmax([np.abs(d0[col] - d0[col + "_lo"]).max(),
                             np.abs(d0[col + "_hi"] - d0[col]).max(),
                             np.abs(d0[col]).max()])
            span = max(float(err), 1e-6) * 1.35
            ax.set_ylim(-span, span)
    one_legend(axes, "model minus held-out data, K=4 (bars: 95% intervals)")
    save(fig, str(out_dir / "fig_residual_radial"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=None,
                    help="directory holding rcv_arrays.npz and the rcv_*.csv")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run = Path(args.run_dir) if args.run_dir else root / "runs" / "halo12-radial-cv-20260917"
    out_dir = run / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_style()
    a = np.load(run / "rcv_arrays.npz")
    stats = pd.read_csv(run / "rcv_stats.csv")
    corr = pd.read_csv(run / "rcv_corr.csv")
    ref = pd.read_csv(run / "rcv_reference.csv")

    k4 = stats[stats["sample"] == "k4"]
    c4 = corr[corr["sample"] == "k4"]
    print("=== RADIAL CONDITIONAL-VELOCITY FIGURES ===", flush=True)
    print(f"inputs: {run}", flush=True)
    print(f"W1 [km/s]           {k4['w1'].min():.4f} .. {k4['w1'].max():.4f}"
          f"   (truth half-split reference {ref['truth_half_w1_kms'].min():.4f}"
          f" .. {ref['truth_half_w1_kms'].max():.4f})", flush=True)
    print(f"mean bias [km/s]    {k4['dmean'].min():+.3f} .. {k4['dmean'].max():+.3f}", flush=True)
    print(f"sigma bias [km/s]   {k4['dsigma'].min():+.3f} .. {k4['dsigma'].max():+.3f}", flush=True)
    print(f"sigma [km/s]        data {k4['sigma_true'].min():.2f} .. {k4['sigma_true'].max():.2f}"
          f"   model {k4['sigma_gen'].min():.2f} .. {k4['sigma_gen'].max():.2f}", flush=True)
    print(f"correlation bias    {c4['drho'].min():+.3f} .. {c4['drho'].max():+.3f}", flush=True)

    win = display_window(a)
    write_display_table(a, win, run)
    rng = fig_dist(a, win, out_dir)
    for name, (lo, hi, peak) in rng.items():
        print(f"dist panel {name}: window [{lo:.0f}, {hi:.0f}] km/s, peak density"
              f" {peak:.4f} per km/s per panel", flush=True)
    fig_curves(stats, corr, ref, out_dir)
    fig_residual(stats, corr, out_dir)
    print(f"wrote {len(list(out_dir.glob('*.pdf')))} PDF (and the same number of SVG)"
          f" figures under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
