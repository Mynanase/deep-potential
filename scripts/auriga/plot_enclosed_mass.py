#!/usr/bin/env python
"""Plotting for the Halo12 enclosed-mass audit.

CONTRACT (plan): this script only READS persisted arrays/JSON produced by
validate_enclosed_mass.py.  It must never reload a model or recompute any
quantity shown in a figure.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def plot_plummer_convergence(analytic_json, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(analytic_json) as f:
        data = json.load(f)
    plum = next(c for c in data["checks"] if c["check"] == "plummer_spherical")
    levels = [k for k in plum["results"] if k.startswith("volume_nodes_")]
    ns = [int(k.split("_")[-1]) for k in levels]
    errs = [plum["results"][k]["max_rel_err"] for k in levels]
    flux = plum["results"]["flux_nodes_800"]["max_rel_err"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(ns, errs, "o-", label="volume path (max over nodes)")
    ax.axhline(flux, color="C1", ls="--",
               label=f"flux path @800 nodes = {flux:.2e}")
    ax.axhline(plum["threshold"], color="k", ls=":", lw=1,
               label=f"engineering threshold {plum['threshold']:g}")
    ax.set(xlabel="radial nodes (log-r trapezoid)", ylabel="max relative error",
           title="Plummer analytic check (step 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_legacy_ablation(ablation_json, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(ablation_json) as f:
        data = json.load(f)
    curves = data["curves"]
    band = data["direction_band_at_outer_edges"]

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"A_legacy": ("C3", "-"), "B_stat_only": ("C2", "--"),
              "C_radius_only": ("C0", "-."), "D_both_fixed": ("C1", ":")}
    labels = {"A_legacy": "A: median + center (legacy)",
              "B_stat_only": "B: mean + center",
              "C_radius_only": "C: median + outer edge",
              "D_both_fixed": "D: mean + outer edge (corrected)"}
    for k, (color, ls) in styles.items():
        c = curves[k]
        ax.plot(c["r_cmp"], c["ratio"], ls, color=color, lw=1.5, label=labels[k])
    ax.fill_between(curves["D_both_fixed"]["r_cmp"], band["p16_ratio"],
                    band["p84_ratio"], color="C1", alpha=0.15, lw=0)
    ax.axhline(1.0, color="k", ls="-", lw=0.8)
    ax.set(xscale="log", ylabel="M_model / M_truth",
           xlabel="r [kpc] (actual comparison radius, no nearest-point labels)",
           title="Step 2: legacy 2x2 ablation (baseline, 256 directions, 40 shells)")
    ax.set_ylim(0.0, None)   # show the full band; no log-axis truncation
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step3_mass_ratio(npz_path, json_path, out_png):
    """Step 3 figure 1: dual-panel mass (top) and model/truth ratio (bottom)
    at the chosen resolution, volume path (mean over scrambles) vs flux path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        conv = json.load(f)
    d = np.load(npz_path)
    r = d["r_edges_outer"]
    dM_truth = d["dM_truth"]
    focus = d["focus_mask"]
    r_lo, r_hi = r[focus][0], r[focus][-1]
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   height_ratios=[1, 1])
    ax1.plot(r, dM_truth, "k-", lw=2, label="truth (all components)")
    for m in conv["models"]:
        vol = d[f"{m}__ratio_vol_seeds"].mean(axis=0) * dM_truth
        flx = d[f"{m}__ratio_flux_seeds"].mean(axis=0) * dM_truth
        ax1.plot(r, vol, color=colors[m], lw=1.4,
                 label=f"{m} volume (mean of 4 scrambles)")
        ax1.plot(r, flx, color=colors[m], lw=1.0, ls="--", label=f"{m} flux")
    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=1e9)
    ax1.set(ylabel=r"$\Delta M(r;0.5\,\mathrm{kpc})$ [Msun]",
            title="Step 3: dual-path enclosed mass at the chosen resolution "
                  f"(N={conv['models_detail']['baseline']['chosen_resolution']['n_dir']}, "
                  f"p={conv['models_detail']['baseline']['chosen_resolution']['per_interval']})")
    # manual y window (tuned to this run): shells inside the 0.5-210 kpc view
    # are all positive, 3e8..1.1e12 Msun; symlog kept so any future in-window
    # negative excursion is displayed, not dropped by a log axis
    ax1.set_ylim(0.0, 3.5e12)
    ax1.legend(fontsize=7)

    for m in conv["models"]:
        vols = d[f"{m}__ratio_vol_seeds"]
        flxs = d[f"{m}__ratio_flux_seeds"]
        # repeatability envelope over the 4 Sobol scrambles.  Plan wording:
        # cross-scramble spread is QUADRATURE REPEATABILITY only, never a
        # model confidence interval (a legacy-misreading trap).
        ax2.fill_between(r, vols.min(axis=0), vols.max(axis=0),
                         color=colors[m], alpha=0.15, lw=0)
        ax2.plot(r, vols.mean(axis=0), color=colors[m], lw=1.4,
                 label=f"{m} volume")
        ax2.plot(r, flxs.mean(axis=0), color=colors[m], lw=1.0, ls="--",
                 label=f"{m} flux")
    from matplotlib.patches import Patch
    ax2.axhspan(0.99, 1.01, color="0.5", alpha=0.3, lw=0,
                label="1% engineering band")
    ax2.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0,
                label=f"focus region {r_lo:g}-{r_hi:g} kpc")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set(xscale="log", xlabel="r [kpc] (truth shell outer edges)",
            ylabel=r"$\Delta M_{\rm model} / \Delta M_{\rm truth}$")
    # manual y window (tuned to this run): visible ratios + envelope span
    # -0.25..1.25, ~8% linear pad baked in
    ax2.set_ylim(-0.4, 1.4)
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.5", alpha=0.15, lw=0))
    labels.append("shaded envelope: 4-scramble min-max (quadrature "
                  "repeatability, NOT a confidence interval)")
    ax2.legend(handles, labels, fontsize=7)
    # fixed axis window (requested view limit)
    ax2.set_xlim(0.5, 210.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step3_closure_convergence(npz_path, json_path, out_png):
    """Step 3 figure 2: closure |dM_rho - dM_flux|/dM_truth vs radius at the
    chosen resolution, plus the per-radius refinement differences (direction
    512->1024 per scramble, radial 4->8), against the 1% threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        conv = json.load(f)
    d = np.load(npz_path)
    r = d["r_edges_outer"]
    dM_truth = d["dM_truth"]
    focus = d["focus_mask"]
    r_lo, r_hi = r[focus][0], r[focus][-1]
    th = conv["gate"]["threshold"]
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # last two direction levels actually swept (gate-relevant pair)
    n_levels = conv["sweeps"]["n_directions"]
    n_lo, n_hi = int(n_levels[-2]), int(n_levels[-1])
    for m in conv["models"]:
        vol = d[f"{m}__ratio_vol_seeds"]
        flx = d[f"{m}__ratio_flux_seeds"]
        per_seed = np.abs(vol - flx)           # closure of each scramble
        clo = per_seed.mean(axis=0)
        ax1.semilogy(r, np.maximum(clo, 1e-8), color=colors[m], lw=1.2,
                     label=f"{m} (mean over 4 scrambles)")
        # per-scramble min-max envelope, the same convention as the bottom
        # panel of the mass/ratio figure
        ax1.fill_between(r, np.maximum(per_seed.min(axis=0), 1e-8),
                         per_seed.max(axis=0), color=colors[m], alpha=0.2,
                         lw=0)
    ax1.axhline(th, color="k", ls=":", lw=1, label=f"threshold {th:g}")
    ax1.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
    ax1.set(ylabel=r"$\epsilon_{\rm closure}=|\Delta M_\rho-\Delta M_{\rm flux}|/M_{\rm truth}$",
            title="Step 3: two-path closure and refinement differences\n"
                  "(max over focus region is the gate statistic)")
    from matplotlib.patches import Patch
    h1, l1 = ax1.get_legend_handles_labels()
    h1.append(Patch(facecolor="0.5", alpha=0.2, lw=0))
    l1.append("shaded: per-scramble min-max closure (quadrature repeatability)")
    ax1.legend(h1, l1, fontsize=7)
    # manual log y window (tuned to this run): contains the full per-seed
    # envelope 1.5e-6 .. 5.2e-1 (the envelope width IS the result: inner
    # radii close to ~1e-6, outer-region scrambles scatter to ~1e-1)
    ax1.set_ylim(5e-7, 1.0)

    for m in conv["models"]:
        dv = np.abs(d[f"{m}__dirsweep_ratio_vol_N{n_lo}"]
                    - d[f"{m}__dirsweep_ratio_vol_N{n_hi}"])
        for i in range(dv.shape[0]):
            ax2.semilogy(r, np.maximum(dv[i], 1e-8), color=colors[m],
                         lw=0.6, alpha=0.5)
        ax2.semilogy(r, np.maximum(dv.max(axis=0), 1e-8), color=colors[m],
                     lw=1.4,
                     label=f"{m} directions {n_lo}->{n_hi} (max over scrambles)")
        dr = np.abs(d[f"{m}__radsweep_ratio_vol_p8"]
                    - d[f"{m}__radsweep_ratio_vol_p4"])
        ax2.semilogy(r, np.maximum(dr, 1e-8), color=colors[m], lw=1.2,
                     ls="--", label=f"{m} radial 4->8")
    ax2.axhline(th, color="k", ls=":", lw=1, label=f"threshold {th:g}")
    ax2.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
    ax2.set(xscale="log", xlabel="r [kpc] (truth shell outer edges)",
            ylabel="|change in model/truth ratio| between levels")
    # manual log y window (tuned to this run): refinement differences span
    # 3e-7 .. 7.7e-1 over scrambles and levels
    ax2.set_ylim(1e-7, 1.6)
    ax2.set_xlim(0.5, 210.0)  # fixed window, same as the mass/ratio figure
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step4_anchor(npz_path, json_path, out_png):
    """Step 4 figure 1: absolute mass difference dM(r;a) = model - truth for
    the three anchors.  Flux path is the PRIMARY estimator (solid); the
    volume path is an independent corroboration (dashed) whose certified
    outer-region floor is annotated.  Envelopes are 4-scramble min-max =
    quadrature repeatability, NOT a confidence interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with open(json_path) as f:
        rep = json.load(f)
    d = np.load(npz_path)
    r = d["r_edges_outer"]
    focus = d["focus_mask"]
    models = [str(x) for x in d["models"]]
    anchor_req = d["anchor_request_kpc"]
    anchor_r = d["anchor_r_kpc"]
    r_lo, r_hi = r[focus][0], r[focus][-1]
    r_sup = float(r[int(d["supplement_edge_pos"])])
    x_lo, x_hi = 0.5, 210.0
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}
    fl = rep["numerical_floors"]["per_model"]
    f_flux = max(v["flux_seed_max"] for v in fl.values())
    f_vol = max(v["vol_seed_max"] for v in fl.values())
    f_volmean = max(v["vol_mean_curve"] for v in fl.values())

    fig, axes = plt.subplots(len(anchor_r), 1, figsize=(8, 11), sharex=True)
    for k, (ax, a_req, a_r) in enumerate(zip(axes, anchor_req, anchor_r)):
        valid = d[f"{models[0]}__anchor{a_req:g}__valid"]
        vis = valid & (r >= x_lo) & (r <= x_hi)
        x = r[vis]
        for m in models:
            for path, ls, lw in (("flux", "-", 1.4), ("vol", "--", 0.9)):
                dm = d[f"{m}__anchor{a_req:g}__dM_{path}_seeds"] - \
                    d[f"{m}__anchor{a_req:g}__dM_truth"][None, :]
                diff = dm[:, vis]
                ax.fill_between(x, diff.min(axis=0), diff.max(axis=0),
                                color=colors[m], alpha=0.15, lw=0)
                ax.plot(x, diff.mean(axis=0), ls=ls, color=colors[m],
                        lw=lw,
                        label=f"{m} {path}" if path == "flux" else None)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
        for xe in (2.0, 65.0, 70.0):
            ax.axvline(xe, color="0.4", ls=":", lw=0.8)
        ax.axvline(r_sup, color="0.4", ls="-", lw=0.8)
        rule = rep["anchors"][k]["rule"].split(";")[0]
        ax.set(ylabel=r"$\delta M(r;a)=\Delta M_{\rm model}-\Delta M_{\rm truth}$ [Msun]",
               title=f"anchor a = {a_r:g} kpc ({rule})")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e8)
        # manual symlog window (tuned to this run, shared by the three
        # anchors): visible dM differences span about -4.3e12 .. +4.7e11
        ax.set_ylim(-4.5e12, 5e11)
        # single shared legend on the top panel, placed lower-left where the
        # anchored curves converge to ~0 (upper-left is crossed by curves)
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles += [Line2D([], [], color="0.3", ls="--", lw=0.9),
                        Patch(facecolor="0.5", alpha=0.15, lw=0),
                        Line2D([], [], color="0.4", ls=":", lw=0.8),
                        Line2D([], [], color="0.4", ls="-", lw=0.8)]
            labels += ["volume path (corroboration)",
                       "4-scramble min-max (quadrature repeatability,\n"
                       "NOT a confidence interval)",
                       "Phi support edges 2/65/70 kpc",
                       f"supplementary shell edge {r_sup:g} kpc"]
            ax.legend(handles, labels, fontsize=6.5, loc="lower left")
    axes[-1].set(xlabel="r [kpc] (truth shell outer edges)",
                 xlim=(x_lo, x_hi))
    fig.suptitle("Step 4: anchored absolute mass difference (flux path primary;\n"
                 f"certified floors: flux <= {f_flux:.2%} per scramble, volume "
                 f"<= {f_vol:.1%} single-scramble\nouter region / "
                 f"<= {f_volmean:.2%} 4-scramble mean)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step4_shell(npz_path, json_path, out_png):
    """Step 4 figure 2: per-shell increments dM_i (anchor-independent).
    Top: shell masses, truth vs models (symlog; extrapolation shells inside
    the view are shown, not clipped away).  Bottom: signed relative
    deviation per shell with 4-scramble envelopes and the per-shell
    numerical floors; a deviation under its floor line is numerically
    unresolved."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with open(json_path) as f:
        rep = json.load(f)
    d = np.load(npz_path)
    r = d["r_edges_outer"]
    focus = d["focus_mask"]
    models = [str(x) for x in d["models"]]
    shell_truth = d["shell_truth"]
    r_lo, r_hi = r[focus][0], r[focus][-1]
    x_lo, x_hi = 0.5, 210.0
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                   height_ratios=[1, 1])
    ax1.plot(r, shell_truth, "k-", lw=2, label="truth shell mass")
    for m in models:
        sf = d[f"{m}__shell_flux_seeds"].mean(axis=0)
        sv = d[f"{m}__shell_vol_seeds"].mean(axis=0)
        ax1.plot(r, sf, color=colors[m], lw=1.2, marker=".", ms=3,
                 label=f"{m} flux")
        ax1.plot(r, sv, color=colors[m], lw=0.8, ls="--", alpha=0.8)
    ax1.set_xscale("log")
    ax1.set_yscale("symlog", linthresh=1e10)
    ax1.set(ylabel=r"shell mass $\delta M_i$ [Msun]",
            title="Step 4: per-shell increments (anchor-independent)")
    # manual symlog window (tuned to this run): visible shell masses span
    # about -9e10 .. +1e11, extrapolation shells included, not clipped
    ax1.set_ylim(-3.2e11, 3.2e11)
    ax1.legend(fontsize=7)

    for m in models:
        for path, ls in (("flux", "-"), ("vol", "--")):
            shells = d[f"{m}__shell_{path}_seeds"]
            dev_seeds = shells / shell_truth[None, :] - 1.0
            dev = d[f"{m}__shell_dev_{path}"]
            ax2.fill_between(r, dev_seeds.min(axis=0), dev_seeds.max(axis=0),
                             color=colors[m], alpha=0.12, lw=0)
            ax2.plot(r, dev, ls=ls, color=colors[m], lw=1.2,
                     label=f"{m} {path}" if path == "flux" else None)
    ax2.plot(r, d[f"{models[0]}__shell_floor_flux"], ls=":", color="k",
             lw=1.0)
    ax2.plot(r, -d[f"{models[0]}__shell_floor_flux"], ls=":", color="k",
             lw=1.0)
    ax2.plot(r, d[f"{models[0]}__shell_floor_vol"], ls=":", color="0.55",
             lw=1.0)
    ax2.plot(r, -d[f"{models[0]}__shell_floor_vol"], ls=":", color="0.55",
             lw=1.0)
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
    for xe in (2.0, 65.0, 70.0):
        ax2.axvline(xe, color="0.4", ls=":", lw=0.8)
    ax2.set(xscale="log", xlabel="r [kpc] (truth shell outer edges)",
            ylabel=r"$(\delta M_{i,\rm model}-\delta M_{i,\rm truth})/\delta M_{i,\rm truth}$")
    # symlog y (linthresh 0.05): the common-region structure (~+/-0.1) and
    # the per-shell floors (~3e-3..0.04) must stay readable while the
    # r > 70 kpc extrapolation excursions (~ -5.6..+1.7) remain fully visible
    ax2.set_yscale("symlog", linthresh=0.05)
    # manual symlog window (tuned to this run)
    ax2.set_ylim(-18.0, 6.0)
    handles, labels = ax2.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="--", lw=0.9),
                Patch(facecolor="0.5", alpha=0.12, lw=0),
                Line2D([], [], color="k", ls=":", lw=1.0),
                Line2D([], [], color="0.55", ls=":", lw=1.0)]
    labels += ["volume path (corroboration)",
               "4-scramble min-max (quadrature repeatability,\n"
               "NOT a confidence interval)",
               "per-shell numerical floor (flux)",
               "per-shell numerical floor (volume)"]
    ax2.legend(handles, labels, fontsize=6.5, loc="lower left")
    ax2.set_xlim(x_lo, x_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step5_direction_band(npz_path, json_path, out_png):
    """Step 5 figure 1: the direction distribution of the per-direction
    integrals (p16/p50/p84 + mean), kept strictly SEPARATE from the step-3
    quadrature repeatability (cross-scramble spread of the mean estimator).

    Top: direction distribution -- angular structure of the model field,
    NOT statistical uncertainty (the legacy misreading).
    Bottom: repeatability of the mean estimator only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with open(json_path) as f:
        rep = json.load(f)
    d = np.load(npz_path)
    r = d["r_edges_outer"]
    focus = d["focus_mask"].astype(bool)
    models = [str(x) for x in d["models"]]
    r_lo, r_hi = r[focus][0], r[focus][-1]
    x_lo, x_hi = 0.5, 210.0
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for m in models:
        c = colors[m]
        ax1.fill_between(r, d[f"{m}__band_p16"], d[f"{m}__band_p84"],
                         color=c, alpha=0.15, lw=0)
        ax1.plot(r, d[f"{m}__band_p50"], color=c, lw=0.7, ls="--")
        ax1.plot(r, d[f"{m}__band_mean"], color=c, lw=1.4, label=m)
    ax1.axhline(1.0, color="k", lw=0.8)
    ax1.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
    for xe in (2.0, 65.0, 70.0):
        ax1.axvline(xe, color="0.4", ls=":", lw=0.8)
    ax1.set(ylabel=r"per-direction $\Delta M(r;0.5)/\Delta M_{\rm truth}$",
            title="Step 5: direction distribution of the per-direction\n"
                  "integrals (4 scrambles x 2048 directions)")
    # manual symlog window (tuned to this run): visible per-direction ratio
    # percentiles/mean span about -27.4 .. +26.6 (the extremes sit at the
    # innermost shells where the truth shell mass is tiny); symlog keeps the
    # focus-region structure (band roughly +/-0.8 around 1) readable
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_ylim(-30.0, 30.0)
    handles, labels = ax1.get_legend_handles_labels()
    handles += [Patch(facecolor="0.5", alpha=0.15, lw=0),
                Line2D([], [], color="0.3", ls="--", lw=0.7)]
    labels += ["p16-p84 across directions: DIRECTION DISTRIBUTION (angular\n"
               "structure of the model field), NOT statistical uncertainty",
               "p50 across directions"]
    # lower center is empty for this run: band p16 >= 0.29 over r in [2, 60]
    # (negative dives only at r >= 105, extrapolation shells, right edge)
    ax1.legend(handles, labels, fontsize=6.5, loc="lower center")

    for m in models:
        c = colors[m]
        ax2.fill_between(r, d[f"{m}__band_rep_min"], d[f"{m}__band_rep_max"],
                         color=c, alpha=0.15, lw=0)
        ax2.plot(r, d[f"{m}__band_mean"], color=c, lw=1.4, label=m)
    ax2.axhspan(0.99, 1.01, color="0.5", alpha=0.3, lw=0,
                label="1% engineering band")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0,
                label=f"focus region {r_lo:g}-{r_hi:g} kpc")
    for xe in (2.0, 65.0, 70.0):
        ax2.axvline(xe, color="0.4", ls=":", lw=0.8,
                    label="Phi support edges 2/65/70 kpc"
                    if xe == 2.0 else None)
    ax2.set(xscale="log", xlabel="r [kpc] (truth shell outer edges)",
            ylabel=r"mean estimator $\Delta M/\Delta M_{\rm truth}$")
    # manual linear window (tuned to this run): visible repeatability
    # envelopes span -0.253 .. 1.247 (min at the innermost shells for
    # baseline, negative excursion from rin2's inner-region scatter)
    ax2.set_ylim(-0.35, 1.35)
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.5", alpha=0.15, lw=0))
    labels.append("4-scramble min-max (QUADRATURE REPEATABILITY of the mean\n"
                  "estimator, NOT a confidence interval)")
    ax2.legend(handles, labels, fontsize=6.5, loc="upper right")
    ax2.set_xlim(x_lo, x_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step5_sky_maps(npz_path, json_path, out_png):
    """Step 5 figure 2: model density on the sky at 5/10/25/50 kpc
    (Mollweide, equal-area grid), normalised by each panel's Sobol-direction
    mean.  FIXED colour scale per radius shared by the three models
    (hand-tuned constants below); model-only diagnostic -- no angular truth
    exists to compare against."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(json_path) as f:
        rep = json.load(f)
    d = np.load(npz_path)
    models = [str(x) for x in d["models"]]
    radii = [float(x) for x in d["map_radii"]]
    lat, lon = d["map_lat"], d["map_lon"]
    stats = rep["part_b"]["per_model_fixed_radius"]

    # hand-tuned FIXED colour scale, one for ALL 12 panels (tuned to this
    # run): normalised maps span -7.5 .. +8.3 across models and radii, so a
    # single symmetric symlog scale +/-9 (linear inside +/-0.5) shows every
    # panel without clipping; nothing is saturated away
    from matplotlib.colors import SymLogNorm
    vmax, linthresh = 9.0, 0.5
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

    fig, axes = plt.subplots(len(models), len(radii), figsize=(16, 9),
                             subplot_kw={"projection": "mollweide"})
    ims = [[None] * len(radii) for _ in models]
    for i, m in enumerate(models):
        for j, rr in enumerate(radii):
            ax = axes[i, j]
            norm_map = d[f"{m}__map_rho_r{rr:g}"] / float(
                d[f"{m}__map_meansobol_r{rr:g}"])
            ims[i][j] = ax.pcolormesh(lon, lat, norm_map, cmap="RdBu_r",
                                      norm=norm, shading="nearest",
                                      rasterized=True)
            ax.grid(color="0.6", lw=0.3, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            st = stats[m][f"{rr:g}"]
            a2 = st["angular_power_ratio_A_l_over_A0"]["l2"]["median_over_seeds"]
            if i == 0:
                ax.set_title(f"r = {rr:g} kpc", fontsize=9)
            ax.text(0.02, -0.08,
                    f"neg {st['neg_fraction']:.1%},  "
                    r"$A_2/A_0$"f" = {a2:.3f}",
                    transform=ax.transAxes, fontsize=6.5, color="0.2")
            if j == 0:
                ax.set_ylabel(m, fontsize=10)
    # one colour bar per radius COLUMN (all four show the SAME single fixed
    # scale; they exist so each column can be read off without eye travel)
    for j in range(len(radii)):
        fig.colorbar(ims[-1][j], ax=list(axes[:, j]), shrink=0.85, pad=0.02,
                     fraction=0.04,
                     label=r"$\rho/\langle\rho\rangle_\Omega$ (Sobol mean)")
    fig.suptitle("Step 5: model density on the sky, normalised by the "
                 "directional mean (ONE fixed symlog colour scale "
                 f"+/-{vmax:g}, linear inside +/-{linthresh:g}, for all "
                 "panels; nothing clipped).\nModel-frame axes = star "
                 "principal-axis frame; model-only diagnostic -- no angular "
                 "truth available", fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


def plot_step5_radial_force(npz_path, json_path, out_png):
    """Step 5 figure 3: radial acceleration per direction (p16/p84 band +
    mean) vs the spherical force of the spherically averaged truth profile.
    The truth line is the spherical-equivalent comparison value, NOT an
    angular truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with open(json_path) as f:
        rep = json.load(f)
    d = np.load(npz_path)
    models = [str(x) for x in d["models"]]
    rf = d["force_r_nodes"]
    r_edges = d["r_edges_outer"]
    focus = d["focus_mask"].astype(bool)
    r_lo, r_hi = r_edges[focus][0], r_edges[focus][-1]
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in models:
        c = colors[m]
        ax.fill_between(rf, d[f"{m}__force_p16"], d[f"{m}__force_p84"],
                        color=c, alpha=0.15, lw=0)
        ax.plot(rf, d[f"{m}__force_mean"], color=c, lw=1.4, label=m)
    ax.plot(r_edges, d["force_truth"], "k--", lw=1.2,
            label="truth (spherical field of the spherically\naveraged "
                  "profile)")
    ax.axvspan(r_lo, r_hi, color="gold", alpha=0.15, lw=0)
    for xe in (2.0, 65.0, 70.0):
        ax.axvline(xe, color="0.4", ls=":", lw=0.8)
    ax.set(xscale="log",
           xlabel="r [kpc]",
           ylabel=r"$a_r = -(V^2/L)\,\langle n\cdot\nabla_q\phi\rangle$ "
                  "[(km/s)$^2$/kpc]",
           title="Step 5: radial force -- mean and direction distribution "
                 "(inward negative)")
    ax.set_yscale("symlog", linthresh=1e3)
    # manual symlog window (tuned to this run): visible direction percentiles
    # span -4.1e4 .. +12 (individual directions can point outward at large r);
    # truth spherical force -3.3e4 .. -38.6
    ax.set_ylim(-4.5e4, 2.0e3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.5", alpha=0.15, lw=0))
    labels.append("p16-p84 across directions (direction distribution, NOT "
                  "statistical uncertainty)")
    # upper left is empty for this run: force p84 <= -2.15e3 for r <= 20 and
    # nothing crosses -1e3 before r = 38.6, so the box floats in clear space
    ax.legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.95)
    ax.set_xlim(0.5, 210.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def plot_step5_truth_skymaps(truth_npz, truth_json, model_npz, out_png):
    """Step 5b figure 1: TRUTH (all-source particles) vs model density on
    the sky, normalised by each map's own mean, ONE fixed symlog colour
    scale for every panel.  Truth row: particle shell mass maps on the
    coarser 45x90 equal-area grid (shot-noise limited); model rows: the
    step-5 model maps (180x360).  Truth panels show the shell particle
    count and A_2/A_0 (no negative fraction -- particles have no negative
    density)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    with open(truth_json) as f:
        rep = json.load(f)
    dt = np.load(truth_npz)
    dm = np.load(model_npz)
    radii = [float(x) for x in dt["map_radii"]]
    models = [str(x) for x in dm["models"]]
    shells = rep["shells"]["per_radius"]
    tlat, tlon = dt["truth_grid_lat"], dt["truth_grid_lon"]
    mlat, mlon = dm["map_lat"], dm["map_lon"]
    vmax, linthresh = 9.0, 0.5        # same fixed scale as the step-5 maps
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

    rows = ["TRUTH (particles)"] + models
    fig, axes = plt.subplots(len(rows), len(radii), figsize=(16, 11),
                             subplot_kw={"projection": "mollweide"})
    ims = [[None] * len(radii) for _ in rows]
    for i, row in enumerate(rows):
        for j, rr in enumerate(radii):
            ax = axes[i, j]
            if i == 0:
                norm_map = dt[f"truth_map_rho_r{rr:g}"] / float(
                    dt[f"truth_map_rho_r{rr:g}"].mean())
                lat, lon = tlat, tlon
                st = shells[f"{rr:g}"]
                n_tot = sum(st["n_particles"].values())
                a2 = st["A_l_over_A0"]["l2"]
                a2e = st["A_l_over_A0_bootstrap_std"]["l2"]
                txt = (f"N={n_tot/1e3:.0f}k\n"
                       f"$A_2/A_0={a2:.2f}\\pm{a2e:.2f}$")
            else:
                m = row
                norm_map = dm[f"{m}__map_rho_r{rr:g}"] / float(
                    dm[f"{m}__map_meansobol_r{rr:g}"])
                lat, lon = mlat, mlon
                neg = float((dm[f"{m}__map_rho_r{rr:g}"] < 0).mean())
                a2 = rep["model_comparison"]["per_model"][m][f"{rr:g}"][
                    "A_l_over_A0_model_median"]["l2"]
                txt = f"neg {100*neg:.1f}%\n$A_2/A_0={a2:.2f}$"
            ims[i][j] = ax.pcolormesh(lon, lat, norm_map, cmap="RdBu_r",
                                      norm=norm, shading="nearest",
                                      rasterized=True)
            ax.grid(color="0.6", lw=0.3, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left", color="k",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
            if j == 0:
                ax.set_ylabel(row, fontsize=10)
            if i == 0:
                ax.set_title(f"r = {rr:g} kpc", fontsize=9)
    # one colour bar per radius COLUMN (all identical, fixed scale)
    for j in range(len(radii)):
        fig.colorbar(ims[-1][j], ax=list(axes[:, j]), shrink=0.85, pad=0.02,
                     fraction=0.04,
                     label=r"$\rho/\langle\rho\rangle_\Omega$ (own mean)")
    fig.suptitle("Step 5b: REAL angular structure vs models -- truth from "
                 "all-source particles (gas+DM+formed stars, 45x90 grid,\n"
                 "shot-noise limited) in the measured model frame; models "
                 "from step 5 (180x360); ONE fixed symlog scale "
                 f"$\\pm{vmax:g}$ for all panels", fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


def plot_step5_power_compare(truth_npz, truth_json, out_png):
    """Step 5b figure 2: low-order angular power A_l/A_0 (l = 1..4) at the
    four map radii -- truth (particle bootstrap error bars) vs the three
    models (median over the step-5 scramble seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(truth_json) as f:
        rep = json.load(f)
    dt = np.load(truth_npz)
    radii = [float(x) for x in dt["map_radii"]]
    model_npz = Path(str(truth_npz).replace("angular_truth",
                                            "angular_diagnostics"))
    models = [str(x) for x in np.load(model_npz)["models"]]
    shells = rep["shells"]["per_radius"]
    colors = {"baseline": "C0", "rin2": "C2", "rout65": "C3"}
    els = np.arange(1, 5)

    fig, axes = plt.subplots(1, len(radii), figsize=(14, 3.6), sharey=True)
    for j, rr in enumerate(radii):
        ax = axes[j]
        st = shells[f"{rr:g}"]
        truth = np.array([st["A_l_over_A0"][f"l{l}"] for l in els])
        terr = np.array([st["A_l_over_A0_bootstrap_std"][f"l{l}"]
                         for l in els])
        ax.bar(els - 0.30, truth, width=0.20, color="k", alpha=0.85,
               label="truth (particles)")
        ax.errorbar(els - 0.30, truth, yerr=terr, fmt="none", ecolor="k",
                    capsize=2, lw=0.8)
        for k, m in enumerate(models):
            vals = np.array([
                rep["model_comparison"]["per_model"][m][f"{rr:g}"][
                    "A_l_over_A0_model_median"][f"l{l}"] for l in els])
            ax.bar(els + (-0.10 + 0.20 * k), vals, width=0.20,
                   color=colors[m], alpha=0.85, label=m)
        ax.set_xticks(els)
        ax.set_xticklabels([f"$l={l}$" for l in els])
        ax.set_title(f"r = {rr:g} kpc", fontsize=10)
        if j == 0:
            ax.set_ylabel("$A_l/A_0$")
            ax.legend(fontsize=7, ncol=1, loc="upper right")
    # manual window (tuned to this run): all A_l/A_0 values (truth and
    # models, all radii) lie in 0.029 .. 0.595; headroom to 0.85
    axes[0].set_ylim(0.0, 0.85)
    fig.suptitle("Step 5b: low-order angular power -- real structure "
                 "(particles, bootstrap errors) vs models", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--analytic-json", type=Path, default=None)
    parser.add_argument("--ablation-json", type=Path, default=None)
    parser.add_argument("--step3-npz", type=Path, default=None)
    parser.add_argument("--step3-json", type=Path, default=None)
    parser.add_argument("--step4-npz", type=Path, default=None)
    parser.add_argument("--step4-json", type=Path, default=None)
    parser.add_argument("--step5-npz", type=Path, default=None)
    parser.add_argument("--step5-json", type=Path, default=None)
    parser.add_argument("--step5-truth-npz", type=Path, default=None,
                        help="step-5b particle-truth npz (angular_truth)")
    parser.add_argument("--step5-truth-json", type=Path, default=None,
                        help="step-5b particle-truth json (angular_truth)")
    parser.add_argument("--mass-png", type=Path, default=None,
                        help="output for the step-3 mass/ratio figure")
    parser.add_argument("--closure-png", type=Path, default=None,
                        help="output for the step-3 closure/refinement figure")
    parser.add_argument("--anchor-png", type=Path, default=None,
                        help="output for the step-4 anchored-difference figure")
    parser.add_argument("--shell-png", type=Path, default=None,
                        help="output for the step-4 per-shell figure")
    parser.add_argument("--band-png", type=Path, default=None,
                        help="output for the step-5 direction-band figure")
    parser.add_argument("--skymap-png", type=Path, default=None,
                        help="output for the step-5 sky-map figure")
    parser.add_argument("--force-png", type=Path, default=None,
                        help="output for the step-5 radial-force figure")
    parser.add_argument("--truth-skymap-png", type=Path, default=None,
                        help="output for the step-5b truth-vs-model sky maps")
    parser.add_argument("--power-png", type=Path, default=None,
                        help="output for the step-5b angular-power comparison")
    parser.add_argument("--out-png", type=Path, default=None)
    args = parser.parse_args()
    if args.analytic_json and args.out_png:
        plot_plummer_convergence(args.analytic_json, args.out_png)
    elif args.ablation_json and args.out_png:
        plot_legacy_ablation(args.ablation_json, args.out_png)
    elif args.step3_npz and args.step3_json and (args.mass_png or args.closure_png):
        if args.mass_png:
            plot_step3_mass_ratio(args.step3_npz, args.step3_json,
                                  args.mass_png)
        if args.closure_png:
            plot_step3_closure_convergence(args.step3_npz, args.step3_json,
                                           args.closure_png)
    elif args.step4_npz and args.step4_json and (args.anchor_png or args.shell_png):
        if args.anchor_png:
            plot_step4_anchor(args.step4_npz, args.step4_json,
                              args.anchor_png)
        if args.shell_png:
            plot_step4_shell(args.step4_npz, args.step4_json,
                             args.shell_png)
    elif args.step5_npz and args.step5_json and (
            args.band_png or args.skymap_png or args.force_png):
        if args.band_png:
            plot_step5_direction_band(args.step5_npz, args.step5_json,
                                      args.band_png)
        if args.skymap_png:
            plot_step5_sky_maps(args.step5_npz, args.step5_json,
                                args.skymap_png)
        if args.force_png:
            plot_step5_radial_force(args.step5_npz, args.step5_json,
                                    args.force_png)
    elif args.step5_truth_npz and args.step5_truth_json and (
            args.truth_skymap_png or args.power_png):
        if args.truth_skymap_png:
            model_npz = args.step5_npz or args.step5_truth_npz.parent \
                / "angular_diagnostics.npz"
            plot_step5_truth_skymaps(args.step5_truth_npz,
                                     args.step5_truth_json, model_npz,
                                     args.truth_skymap_png)
        if args.power_png:
            plot_step5_power_compare(args.step5_truth_npz,
                                     args.step5_truth_json, args.power_png)
    else:
        parser.error("provide --analytic-json/--out-png, "
                     "--ablation-json/--out-png, "
                     "--step3-npz/--step3-json with --mass-png/--closure-png, "
                     "--step4-npz/--step4-json with "
                     "--anchor-png/--shell-png, or "
                     "--step5-npz/--step5-json with "
                     "--band-png/--skymap-png/--force-png")


if __name__ == "__main__":
    sys.exit(main())
