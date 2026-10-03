#!/usr/bin/env python
"""Main figure of the particle-truth spectrum resolution-convergence study.

Plot layer only: reads the persisted npz/json of resolution_convergence.py,
never touches particles or models.  One WIDE figure, three panels:
  (a) P_N(lambda) for N^3 in {64, 96, 128, 192} with the 10/5/4/3/2.5 kpc
      cutoff lines, nested 2x-Nyquist guard shadings, and lambda_reliable;
  (b) adjacent-grid relative differences per log-lambda bin against the
      pre-registered 10% tolerance (gate pairs highlighted);
  (c) shot-noise fraction P_AB/P_full at 96^3 and 192^3 against the 0.5
      noise-dominated reference.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orx_figstyle as ofs  # noqa: E402

COLOR = {64: ofs.PALETTE["red"], 96: ofs.PALETTE["orange"],
         128: ofs.PALETTE["green"], 192: ofs.PALETTE["blue"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="resolution_convergence.npz from the compute step")
    ap.add_argument("--json", type=Path, required=True,
                    help="resolution_convergence.json from the compute step")
    ap.add_argument("--fig-dir", type=Path,
                    default=Path("figures/resolution-convergence"))
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[10.0, 5.0, 4.0, 3.0, 2.5])
    args = ap.parse_args()
    d = np.load(args.input)
    summary = json.load(open(args.json))
    lam = d["lam"]
    centers = d["bin_centers"]
    resolutions = summary["config"]["resolutions"]
    ab_resolutions = summary["config"]["ab_resolutions"]
    rel_tol = summary["config"]["rel_tol"]
    lam_reliable = summary["conclusion"]["lambda_reliable_kpc"]
    guards = {int(n): g["guard_kpc"] for n, g in summary["grids"].items()}
    print(f"read {args.input}: resolutions {resolutions}, lambda_reliable {lam_reliable:.3f} kpc")

    ofs.use_style()
    fig, axes = ofs.figure_grid(1, 3, width=ofs.WIDE, ratio=0.42)

    # (a) spectra -----------------------------------------------------------
    ax = axes[0]
    for n in resolutions:
        ax.loglog(lam, d[f"P{n}"], color=COLOR[n], lw=1.1,
                  label=str(n) + "$^3$", zorder=3)
    for n in (96, 128, 192):
        ax.axvspan(lam.min(), guards[n], color="0.0",
                   alpha=0.05 * (192 // n), lw=0, zorder=1)
    for c in args.cutoffs:
        ax.axvline(c, color=ofs.BASELINE, lw=0.5, alpha=0.6, zorder=2)
    ax.axvline(lam_reliable, color="k", lw=1.0, ls="--", zorder=4)
    ax.text(lam_reliable * 0.97, 0.04, r"$\lambda_{\rm reliable}$",
            transform=ax.get_xaxis_transform(), rotation=90,
            ha="right", va="bottom", fontsize=6)
    ax.set_xlim(1.4, 80.0)
    ax.set_xlabel(r"Wavelength $\lambda$ [kpc]")
    ax.set_ylabel(r"Mean radial power of $\delta\rho$")
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    # (b) adjacent-grid convergence ----------------------------------------
    ax = axes[1]
    for n1, n2 in zip(resolutions[:-1], resolutions[1:]):
        rel = d[f"rel_{n1}_{n2}"]
        gate = summary["convergence"][f"{n1}-{n2}"]["gate"]
        ax.loglog(centers, rel, color=COLOR[n2], lw=1.1,
                  ls="-" if gate else "--",
                  label=str(n1) + r"$\to$" + str(n2) + (" (gate)" if gate else ""))
    ax.axhline(rel_tol, color=ofs.BASELINE, lw=0.8, ls=":")
    ax.axvline(lam_reliable, color="k", lw=1.0, ls="--")
    ax.set_xlim(1.4, 80.0)
    ax.set_xlabel(r"Wavelength $\lambda$ [kpc]")
    ax.set_ylabel(r"$|\Delta P|/P_{\rm fine}$ (bin median)")
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    # (c) shot-noise fraction ----------------------------------------------
    ax = axes[2]
    for n in ab_resolutions:
        ax.loglog(centers, d[f"noise_{n}"], color=COLOR[n], lw=1.1,
                  label=str(n) + "$^3$")
    ax.axhline(0.5, color=ofs.BASELINE, lw=0.8, ls=":")
    ax.axhline(0.2, color=ofs.BASELINE, lw=0.5, ls=":", alpha=0.6)
    ax.axvline(lam_reliable, color="k", lw=1.0, ls="--")
    ax.set_xlim(1.4, 80.0)
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel(r"Wavelength $\lambda$ [kpc]")
    ax.set_ylabel(r"$P_{A-B}/P_{\rm full}$ (bin median)")
    ax.legend(fontsize=6, frameon=False, loc="lower left")

    # explicit log-lambda ticks, applied AFTER all plotting calls: loglog and
    # set_xlim rebuild the log axis and would discard a fixed locator set
    # earlier; the auto minor labels (2x10^0, 3x10^0, ...) collide in 1.4-80 kpc
    for ax in axes:
        ax.set_xticks([2, 3, 5, 10, 20, 50])
        ax.set_xticklabels(['2', '3', '5', '10', '20', '50'])
        ax.xaxis.set_minor_formatter(NullFormatter())

    ofs.panel_labels(axes)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.fig_dir / "resolution-convergence")
    outs = ofs.save(fig, stem, formats=("png", "pdf", "svg"))
    for p in outs:
        print(p)
    print("PLOT_RESOLUTION_CONVERGENCE_DONE")


if __name__ == "__main__":
    main()
