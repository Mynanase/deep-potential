#!/usr/bin/env python
"""Plotting for the Halo12 step-6 DF-constraint audit.

CONTRACT (plan): this script only READS persisted arrays/JSON produced by
audit_df_constraints.py.  It must never reload a model or recompute any
quantity shown in a figure.

Axis-limit convention (user decision 2026-09-15): manual set_ylim constants
tuned to this run; each carries a comment with the data range it must
contain.  Re-check them when re-running with a new model.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

RUNS_ORDER = ["baseline", "rin2", "rout65"]
RUN_COLORS = {"baseline": "C0", "rin2": "C1", "rout65": "C2"}


def _floored_rel(diff, ref, floor):
    return np.abs(diff) / np.maximum(np.abs(ref), floor)


def plot_score_audit(npz_path, json_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    meta = json.loads(Path(json_path).read_text())
    runs = [r for r in RUNS_ORDER if f"{r}__dlnf_ad_x64" in z]

    fig, (ax_fd, ax_grp, ax_dec) = plt.subplots(
        3, 1, figsize=(7, 11), layout="constrained")

    # ---- top: FD-vs-autodiff step-size scan (baseline) ----
    steps = np.array(meta["per_run"]["baseline"]["fd_scan"]["steps"])
    err = z["baseline__fd_err"]                      # (n_h, 6)
    comp_names = [r"$\partial_{q_x}$", r"$\partial_{q_y}$", r"$\partial_{q_z}$",
                  r"$\partial_{p_x}$", r"$\partial_{p_y}$", r"$\partial_{p_z}$"]
    for c in range(6):
        ax_fd.loglog(steps, err[:, c], "o-", ms=3, lw=0.8,
                     label=comp_names[c], color=f"C{c}")
    ax_fd.axhline(5e-2, color="k", ls=":", lw=1,
                  label="engineering threshold 5e-2")
    best = meta["per_run"]["baseline"]["fd_scan"]["best_step"]
    ax_fd.axvline(best, color="gray", ls="--", lw=1,
                  label=f"best step h={best:g}")
    ax_fd.set(xlabel="central-FD step size h (code units)",
              ylabel="median |FD - autodiff| / max(|AD|, floor)",
              title="Finite differences vs autodiff (baseline, x64, "
                    f"n={meta['per_run']['baseline']['fd_scan']['n_points']})")
    # data range: full curves 1e-3 .. 1.2 (smoke/full measured); head-room x3
    ax_fd.set_ylim(1e-4, 3.0)   # FD err spans ~5e-4 .. 1.2 incl. truncation wing
    ax_fd.legend(fontsize=7, ncol=2)

    # ---- middle: stored-vs-autodiff error distributions per run/group ----
    x = np.arange(len(runs))
    for gi, (gname, sl) in enumerate([("spatial", slice(0, 3)),
                                      ("velocity", slice(3, 6))]):
        for ri, run in enumerate(runs):
            stored = z[f"{run}__dlnf_stored"][:, sl]
            ad = z[f"{run}__dlnf_ad_x64"][:, sl]
            rel = _floored_rel(ad - stored, stored,
                               np.median(np.abs(stored), axis=0))
            pos = x[ri] + (gi - 0.5) * 0.18
            ax_grp.vlines(pos, np.median(rel), np.percentile(rel, 99),
                          color=RUN_COLORS[run], lw=3, alpha=0.9)
            ax_grp.plot(pos, np.median(rel), "o", ms=4,
                        color=RUN_COLORS[run])
    ax_grp.axhline(1e-3, color="k", ls=":", lw=1, label="gate: median < 1e-3")
    ax_grp.axhline(1e-2, color="k", ls="--", lw=1, label="gate: p99 < 1e-2")
    ax_grp.set_xticks(x, runs)
    ax_grp.set(ylabel="stored vs autodiff(x64), floored rel",
               title="Stored score vs recomputation\n"
                     "(bar: median -> p99; left bar spatial, right velocity)")
    # data range: medians 7e-4..1.4e-3, p99 up to 1.8e-2
    ax_grp.set_ylim(0.0, 2.5e-2)   # p99 max ~1.8e-2; head-room to 2.5e-2
    ax_grp.set_yscale("symlog", linthresh=1e-3)
    ax_grp.legend(fontsize=7)

    # ---- bottom: error decomposition (medians with p99 whiskers) ----
    labels = ["stored(GPU f32)\nvs x64", "f32 CPU vs x64\n(arithmetic)",
              "ODE strict vs\ndefault (x64)"]
    keys = ["stored_vs_autodiff_x64", "f32cpu_vs_x64cpu_arithmetic",
            "ode_strict_vs_default_x64"]
    for ri, run in enumerate(runs):
        for ki, key in enumerate(keys):
            s = meta["per_run"][run][key]
            med = max(s["spatial_median"], s["velocity_median"])
            p99 = max(s["spatial_p99"], s["velocity_p99"])
            pos = ki + (ri - 1) * 0.22
            ax_dec.vlines(pos, med, p99, color=RUN_COLORS[run], lw=3)
            ax_dec.plot(pos, med, "o", ms=4, color=RUN_COLORS[run])
    ax_dec.set_xticks(np.arange(3), labels)
    ax_dec.set(ylabel="floored rel error (worse of spatial/velocity)",
               title="Where the score-chain difference lives: medians -> p99")
    # data range: medians 2e-4..1.4e-3, p99 up to 1.8e-2
    ax_dec.set_ylim(0.0, 2.5e-2)   # same scale as middle panel
    ax_dec.set_yscale("symlog", linthresh=1e-3)

    fig.suptitle("Step 6.1-6.3: DF score audit "
                 f"(n={meta['per_run']['baseline']['n_rows']} stratified "
                 "stored rows per run)", fontsize=11)
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")
    plt.close(fig)


def plot_cbe(npz_path, json_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    meta = json.loads(Path(json_path).read_text())

    r = z["r_kpc"]
    radii = np.unique(r)
    med_absR = [np.median(np.abs(z["R"][r == rr])) for rr in radii]
    p84_absR = [np.percentile(np.abs(z["R"][r == rr]), 84) for rr in radii]
    med_t1 = [np.median(np.abs(z["term1"][r == rr])) for rr in radii]
    med_t2 = [np.median(np.abs(z["term2"][r == rr])) for rr in radii]
    med_rel = [np.median(z["relR"][r == rr]) for rr in radii]
    p84_rel = [np.percentile(z["relR"][r == rr], 84) for rr in radii]
    med_lnf = [np.median(z["lnf"][r == rr]) for rr in radii]

    fig, (ax_abs, ax_rel, ax_sc) = plt.subplots(
        3, 1, figsize=(7, 11), layout="constrained")

    ax_abs.loglog(radii, med_t1, "--", color="C0", lw=1, label="|term1| median")
    ax_abs.loglog(radii, med_t2, "--", color="C3", lw=1, label="|term2| median")
    ax_abs.loglog(radii, med_absR, "o-", color="C1", lw=1.5, ms=3,
                  label="|R| median")
    ax_abs.fill_between(radii, med_absR, p84_absR, color="C1", alpha=0.2, lw=0)
    dr = meta["stored_training_points"]["abs_R_stored_minus_fresh_median"]
    ax_abs.axhline(dr, color="gray", ls=":", lw=1.2,
                   label=f"score-storage noise on R (median {dr:.1e})")
    ax_abs.set(xlabel="r [kpc]", ylabel="code units",
               title="Step 6.4: CBE residual R = p.d_q lnF - (d_q phi).d_p lnF "
                     f"(baseline, n={meta['n_points']})")
    # data range: |R| median ~0.4..3, terms ~1..10, noise line ~1e-2
    ax_abs.set_ylim(3e-3, 60.0)   # terms reach ~20; dR line ~1e-2; head-room x3
    ax_abs.legend(fontsize=8)

    ax_rel.loglog(radii, med_rel, "o-", color="C1", lw=1.5, ms=3,
                  label="relR median")
    ax_rel.fill_between(radii, med_rel, p84_rel, color="C1", alpha=0.2, lw=0,
                        label="p84")
    ax_rel.set(xlabel="r [kpc]",
               ylabel="|R| / (|t1| + |t2| + 1.0)",
               title="Normalized CBE residual (epsilon = 1.0 pre-declared)")
    # data range: medians ~0.03..0.3, p84 to ~0.7
    ax_rel.set_ylim(1e-2, 2.0)    # p84 max ~0.7; head-room x3
    ax_rel.legend(fontsize=8)

    ax_sc.set_yscale("symlog", linthresh=1.0)
    ax_sc.scatter(np.abs(z["stored_R"]), np.abs(z["stored_R_fresh"]),
                  s=4, alpha=0.4, color="C0")
    lims = [0.0, max(np.abs(z["stored_R"]).max(),
                     np.abs(z["stored_R_fresh"]).max())]
    ax_sc.plot([0, lims[1]], [0, lims[1]], "k--", lw=0.8, label="y = x")
    st = meta["stored_training_points"]
    ax_sc.set(xlabel="|R| with stored (f32 GPU) scores",
              ylabel="|R| with fresh x64 scores",
              title="R on stored training points (n="
                    f"{st['n_rows']}): dR median "
                    f"{st['abs_R_stored_minus_fresh_median']:.1e}")
    # data range: |R| 0..~31.6 on stored training points
    ax_sc.set_xlim(-2.0, 36.0)    # max |R| ~31.6; head-room to 36
    ax_sc.set_ylim(-2.0, 60.0)    # symlog linthresh=1; data to ~31.6
    ax_sc.legend(fontsize=8)

    fig.suptitle("Step 6.4: CBE diagnostics "
                 f"(DF weight fraction inside 30 kpc = "
                 f"{meta['overall']['df_weight_fraction_inside_30kpc']:.3f})",
                 fontsize=11)
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")
    plt.close(fig)


def plot_local_force(npz_path, json_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    meta = json.loads(Path(json_path).read_text())

    r = z["pos_r_kpc"]
    radii = np.unique(r)
    tiers = [k for k in meta["per_tier"]]

    fig, (ax_res, ax_cond, ax_g) = plt.subplots(
        3, 1, figsize=(7, 11), layout="constrained", sharex=True)

    for ti, tier in enumerate(tiers):
        m = meta["per_tier"][tier]["n_velocities"]
        resid = z[f"{tier}__resid_rel"]
        cond = z[f"{tier}__cond"]
        g_hat = z[f"{tier}__g_hat"]
        g_phi = z[f"{tier}__g_phi"]
        ok = ~np.isnan(g_hat).any(axis=1)

        med_r = [np.median(resid[r == rr]) for rr in radii]
        p84_r = [np.percentile(resid[r == rr], 84) for rr in radii]
        ax_res.loglog(radii, med_r, "o-", ms=3, lw=1.2, color=f"C{ti}",
                      label=f"M={m} velocities (median)")
        ax_res.fill_between(radii, med_r, p84_r, color=f"C{ti}", alpha=0.15,
                            lw=0, label=f"M={m} p84" if ti == 0 else None)

        ax_cond.loglog(r[ok], cond[ok], "o", ms=3, color=f"C{ti}",
                       label=f"M={m}")
        rel = np.linalg.norm(g_hat[ok] - g_phi[ok], axis=1) / np.maximum(
            np.linalg.norm(g_phi[ok], axis=1), 1e-30)
        med_g = [np.median(rel[r[ok] == rr]) for rr in radii]
        p84_g = [np.percentile(rel[r[ok] == rr], 84) for rr in radii]
        ax_g.loglog(radii, med_g, "o-", ms=3, lw=1.2, color=f"C{ti}",
                    label=f"M={m} (median)")
        ax_g.fill_between(radii, med_g, p84_g, color=f"C{ti}", alpha=0.15,
                          lw=0, label=f"M={m} p84" if ti == 0 else None)

    ax_res.axhline(1.0, color="k", ls=":", lw=1, label="all of b explained")
    ax_res.set(ylabel="min_g ||A g - b|| / ||b||",
               title="Step 6.5: local force constraint residual\n"
                     "(SVD least squares; rank-3 positions)")
    # data range: per-radius medians ~0.10..0.35, p84 band to ~0.62
    ax_res.set_ylim(3e-2, 3.0)    # p84 max ~0.62; head-room x5
    ax_res.legend(fontsize=7)

    ax_cond.set(ylabel="condition number s1/s3",
                title="Constraint conditioning (all 64 positions rank 3;\n"
                      "COND_MAX = 1e4 far above the axis)")
    # data range: cond 1.13..4.41 across all positions and tiers
    ax_cond.set_ylim(1.0, 10.0)   # data 1.13..4.41; head-room x2 both ends

    stab = meta["cross_tier_stability"]
    ax_g.axhline(stab["g_rel_change_median"], color="gray", ls=":",
                 lw=1.2, label=f"cross-tier stability median "
                 f"({stab['g_rel_change_median']:.2f})")
    ax_g.set(xlabel="r [kpc]", ylabel="|g_hat - grad phi| / |grad phi|",
             title="Step 6.6: best local force vs trained Phi gradient\n"
                   "(well-conditioned positions)")
    # data range: per-radius medians ~0.05..0.5, p84 band to ~1.2
    ax_g.set_ylim(2e-2, 5.0)      # p84 max ~1.2; head-room x4
    ax_g.legend(fontsize=7)

    n_dirs = meta["per_tier"][tiers[0]]["n_positions"] // len(radii)
    fig.suptitle("Step 6.5-6.6: SVD local force constraints "
                 f"({len(radii)} radii x {n_dirs} directions, baseline)",
                 fontsize=11)
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")
    plt.close(fig)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                             formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audit-dir", type=Path,
                        default=REPO / "runs/halo12-mass-audit-20260914T0453")
    parser.add_argument("--score-png", type=Path, default=None)
    parser.add_argument("--cbe-png", type=Path, default=None)
    parser.add_argument("--force-png", type=Path, default=None)
    args = parser.parse_args()

    step6 = args.audit_dir / "step6"
    if args.score_png:
        plot_score_audit(step6 / "score_audit.npz",
                         step6 / "score_audit.json", args.score_png)
    if args.cbe_png:
        plot_cbe(step6 / "cbe_diagnostics.npz",
                 step6 / "cbe_diagnostics.json", args.cbe_png)
    if args.force_png:
        plot_local_force(step6 / "local_force_constraints.npz",
                         step6 / "local_force_constraints.json",
                         args.force_png)


if __name__ == "__main__":
    main()
