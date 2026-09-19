#!/usr/bin/env python
"""Potential diagnostics for the three adjudicated w1024 clean+smooth Phis.

One figure, one claim: S1 diverges from base/s11 only in the outer region.
(a) sphere-averaged inward acceleration, (b) enclosed-mass error vs the
simulation truth (read from the adjudication run's truth_compare.json - no
model reload, honouring the plot_enclosed_mass contract), (c) signed density
sphere mean with per-direction min-max band, (d) axis-vs-sphere anisotropy
of the acceleration.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    sobol_directions, make_radial_nodes, rho_from_phi, grad_phi_dot_n_batch)
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODEL_STYLE = {
    "base": ("blue", "base (seed-0)"),
    "s11": ("orange", "s11 (reshuffle)"),
    "S1": ("red", "S1 (stratified)"),
}


def load_phi_f32(run_dir):
    import jax
    import fit_all
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        model = fit_all.load_potential(Path(run_dir) / "models" / "Phi",
                                       checkpoint_index=-1)
        return model.phi_model
    finally:
        jax.config.update("jax_enable_x64", prev)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="label=run_dir (run_dir contains models/Phi)")
    ap.add_argument("--truth-compare", type=Path, required=True,
                    help="truth_compare.json written by compare_phi_truth.py")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/s1-potential"))
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--r-inner", type=float, default=1.0)
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-nodes", type=int, default=48)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    r_nodes = np.asarray(make_radial_nodes(args.r_inner, args.r_outer,
                                           args.n_nodes - 1), dtype=float)
    axis_hat = np.eye(3)
    results = {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        t0 = time.time()
        g_sph = np.empty(r_nodes.size)
        rho_mean = np.empty(r_nodes.size)
        rho_min = np.empty(r_nodes.size)
        rho_max = np.empty(r_nodes.size)
        for j, r in enumerate(r_nodes):
            dn = np.asarray(grad_phi_dot_n_batch(phi, dirs * (r / L_KPC), dirs))
            g_sph[j] = float(dn.mean()) * V_KMS ** 2 / L_KPC
            rho = np.asarray(rho_from_phi(phi, dirs * (r / L_KPC), L_KPC, V_KMS))
            rho_mean[j], rho_min[j], rho_max[j] = rho.mean(), rho.min(), rho.max()
        g_axis = np.array([
            [float(np.asarray(grad_phi_dot_n_batch(
                phi, (axis_hat[i] * (r / L_KPC))[None, :],
                axis_hat[i][None, :]))[0]) * V_KMS ** 2 / L_KPC
             for i in range(3)] for r in r_nodes])
        with np.errstate(divide="ignore", invalid="ignore"):
            aniso = np.max(np.abs(g_axis / g_sph[:, None] - 1.0), axis=1) * 100.0
        aniso[~np.isfinite(aniso)] = np.nan
        results[label] = dict(g_sph=g_sph, rho_mean=rho_mean, rho_min=rho_min,
                              rho_max=rho_max, aniso=aniso)
        print(f"[{label}] {r_nodes.size} nodes x {args.n_dirs} dirs in "
              f"{time.time() - t0:.0f}s; g_sph(70kpc)={g_sph[-1]:.1f} "
              f"(km/s)^2/kpc; rho_mean(1kpc)={rho_mean[0]:.3e} Msun/kpc^3")

    with open(args.truth_compare) as f:
        tc = json.load(f)
    r_edges = np.asarray(tc["r_edges"], dtype=float)
    dM_true = np.asarray(tc["dM_true"], dtype=float)

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, 2, width=ofs.WIDE, ratio=0.5)
    (axa, axb), (axc, axd) = axes
    for spec in args.model:
        label = spec.split("=", 1)[0]
        color, _ = MODEL_STYLE.get(label, (None, label))
        res = results[label]
        axa.plot(r_nodes, res["g_sph"], color=ofs.PALETTE.get(color, color), lw=1.4)
        rel = (np.asarray(tc["models"][label]["dM_flux"]) - dM_true) / np.abs(dM_true) * 100.0
        axb.plot(r_edges, rel, color=ofs.PALETTE.get(color, color), lw=1.4)
        axc.plot(r_nodes, res["rho_mean"], color=ofs.PALETTE.get(color, color), lw=1.4)
        axc.fill_between(r_nodes, res["rho_min"], res["rho_max"],
                         color=ofs.PALETTE.get(color, color), alpha=0.12, lw=0)
        axd.plot(r_nodes, res["aniso"], color=ofs.PALETTE.get(color, color), lw=1.4)
    axb.axhline(0.0, color=ofs.BASELINE, lw=0.8)
    axb.axvspan(50.0, 70.0, color=ofs.MUTED, alpha=0.4, lw=0)
    axc.axhline(0.0, color=ofs.BASELINE, lw=0.8)
    rho_abs_max = max(np.nanmax(np.abs(results[l]["rho_mean"])) for l in results)
    axc.set_yscale("symlog", linthresh=max(1.0, rho_abs_max * 1e-3))
    for ax in (axa, axb, axc, axd):
        ax.set_xscale("log")
        ax.set_xlim(args.r_inner, args.r_outer)
        ax.set_xlabel("r [kpc]")
    axa.set_ylabel(r"inward accel. $\langle g_r\rangle_\Omega$ "
                   r"[$\mathrm{km^2\,s^{-2}\,kpc^{-1}}$]")
    axb.set_ylabel("enclosed-mass error [%]")
    axc.set_ylabel(r"$\langle\rho\rangle_\Omega$ "
                   r"[$M_\odot\,\mathrm{kpc^{-3}}$]")
    axd.set_ylabel("axis anisotropy of $g_r$ [%]")
    ofs.panel_labels([axa, axb, axc, axd])
    handles = [axa.plot([], [], color=ofs.PALETTE[c], lw=1.6)[0]
               for c in ("blue", "orange", "red")]
    fig.legend(handles, [MODEL_STYLE[l][1] for l in ("base", "s11", "S1")],
               loc="outside lower center", ncol=3, frameon=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "s1_potential_diagnostics.npz",
             r_nodes=r_nodes, r_edges=r_edges,
             **{f"{l}_{k}": v for l in results for k, v in results[l].items()})
    out = ofs.save(fig, str(args.output_dir / "s1-potential-diagnostics"))
    for p in out:
        print("saved", p)
    print("POTENTIAL_DIAG_FIGURES_DONE")


if __name__ == "__main__":
    sys.exit(main())
