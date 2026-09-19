#!/usr/bin/env python
"""Log-color 2D density maps, truth comparison, and residuals for three Phis.

Row 1: log10 rho on the y=0 meridional (x-z) slice - (a) spherical truth from
the 60-shell total-mass profile, (b-d) models from the potential Laplacian,
shared viridis scale; cells with non-positive model density are masked grey
(the negative-Laplacian diagnostic, with the in-domain fraction printed).
Row 2: (e) radial log-log profiles of the sphere-averaged density vs truth,
(f-h) residuals log10(rho_model/rho_true) on the same slice, RdBu_r
symmetric limits. Valid domain r in [1.09, 70] kpc.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    load_truth, sobol_directions, make_radial_nodes, rho_from_phi)
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "s11", "S1")
MODEL_TEXT = {"base": "base (seed-0)", "s11": "s11 (reshuffle)", "S1": "S1 (stratified)"}
COLS = ("truth", "base", "s11", "S1")
COL_TEXT = {"truth": "truth (total matter)"}


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


def rho_grid(phi, xs, zs, batch=32768):
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    q = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]) / L_KPC
    out = np.empty(q.shape[0], dtype=np.float64)
    for i in range(0, q.shape[0], batch):
        out[i:i + batch] = np.asarray(rho_from_phi(phi, q[i:i + batch], L_KPC, V_KMS))
    return out.reshape(xx.shape)


def rho_sphere_mean(phi, r_nodes, dirs):
    out = np.empty(r_nodes.size)
    for j, r in enumerate(r_nodes):
        out[j] = float(np.mean(np.asarray(
            rho_from_phi(phi, dirs * (r / L_KPC), L_KPC, V_KMS))))
    return out


def truth_density(truth, r_fine):
    """Shell-averaged truth density at shell centres, interpolated in log-log."""
    r_c = np.asarray(truth["r_center"], dtype=float)
    dm = np.asarray(truth["M_shell_total"], dtype=float)
    r_e = np.asarray(truth["r_edges"], dtype=float)
    vol = 4.0 / 3.0 * np.pi * (r_e[1:] ** 3 - r_e[:-1] ** 3)
    rho_c = dm / vol
    return np.exp(np.interp(np.log(r_fine), np.log(r_c), np.log(rho_c))), r_c, rho_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/s1-density-2d"))
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-sphere-nodes", type=int, default=48)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    truth = load_truth(args.truth)
    edges = np.asarray(truth["r_edges"], dtype=float)
    r_anchor = float(edges[edges >= 1.0][0])
    r_nodes = np.asarray(make_radial_nodes(r_anchor, args.r_outer,
                                           args.n_sphere_nodes - 1), dtype=float)
    rho_true_nodes, r_c, rho_c = truth_density(truth, r_nodes)
    print(f"truth: anchor {r_anchor:.4f} kpc; rho_true(1.09)={rho_true_nodes[0]:.3e}, "
          f"rho_true(70)={rho_true_nodes[-1]:.3e} Msun/kpc^3")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    xs = np.linspace(-args.r_outer, args.r_outer, args.n_grid)
    grids, profs, negfrac = {}, {}, {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        t0 = time.time()
        g = rho_grid(phi, xs, xs)
        rr = np.hypot(np.meshgrid(xs, xs, indexing="xy")[0],
                      np.meshgrid(xs, xs, indexing="xy")[1])
        in_dom = (rr >= r_anchor) & (rr <= args.r_outer)
        negfrac[label] = float(np.mean(g[in_dom] <= 0.0))
        grids[label] = g
        profs[label] = rho_sphere_mean(phi, r_nodes, dirs)
        print(f"[{label}] grid {args.n_grid}^2 + {args.n_sphere_nodes}x{args.n_dirs} "
              f"sphere nodes in {time.time() - t0:.0f}s; rho_mean(1.09)="
              f"{profs[label][0]:.3e}, rho_mean(70)={profs[label][-1]:.3e}; "
              f"non-positive density cells in domain: {negfrac[label]*100:.1f}%")

    rr = np.hypot(np.meshgrid(xs, xs, indexing="xy")[0],
                  np.meshgrid(xs, xs, indexing="xy")[1])
    in_dom = (rr >= r_anchor) & (rr <= args.r_outer)
    rho_true_grid = truth_density(truth, rr.ravel())[0].reshape(rr.shape)

    row1 = [np.log10(rho_true_grid)]
    row1 += [np.where((in_dom) & (grids[l] > 0), np.log10(np.abs(grids[l])), np.nan)
             for l in MODELS]
    finite1 = np.concatenate([r[np.isfinite(r)] for r in row1])
    vmin, vmax = np.percentile(finite1, [0.5, 99.5])
    resid = {l: np.log10(np.where((in_dom) & (grids[l] > 0), grids[l], np.nan)
                         / rho_true_grid) for l in MODELS}
    rabs = np.nanpercentile(np.abs(np.stack(list(resid.values()))), 98)
    print(f"shared scales: log10 rho in [{vmin:.2f}, {vmax:.2f}]; "
          f"log-ratio residual +/-{rabs:.2f} dex (98th pct)")

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, 4, width=ofs.WIDE, ratio=0.58)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)
    im0 = axes[0, 0].pcolormesh(xs, xs, row1[0], cmap=cmap_seq, vmin=vmin,
                                vmax=vmax, rasterized=True, shading="auto")
    ims = [im0]
    for j, l in enumerate(MODELS):
        im = axes[0, j + 1].pcolormesh(xs, xs, row1[j + 1], cmap=cmap_seq,
                                       vmin=vmin, vmax=vmax, rasterized=True,
                                       shading="auto")
        ims.append(im)
    im_res = None
    for j, l in enumerate(MODELS):
        im_res = axes[1, j + 1].pcolormesh(xs, xs, resid[l], cmap=cmap_div,
                                          vmin=-rabs, vmax=rabs,
                                          rasterized=True, shading="auto")
    axp = axes[1, 0]
    axp.loglog(r_nodes, rho_true_nodes, color="0.2", lw=1.6, label="truth")
    for l in MODELS:
        color = {"base": "blue", "s11": "orange", "S1": "red"}[l]
        axp.loglog(r_nodes, profs[l], color=ofs.PALETTE[color], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\langle\rho\rangle_\Omega$ "
                   r"[$M_\odot\,\mathrm{kpc^{-3}}$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower left")
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            if not ax is axp:
                ax.set_aspect("equal")
                ax.set_xlabel("x [kpc]")
                ax.set_ylabel("z [kpc]")
            else:
                ax.set_aspect("auto")
    for j, name in enumerate(COLS):
        axes[0, j].annotate(COL_TEXT.get(name, MODEL_TEXT.get(name, name)),
                            xy=(0, 1), xycoords="axes fraction", xytext=(0, 14),
                            textcoords="offset points", fontsize=7,
                            color="#333333")
    ofs.panel_labels(list(axes.ravel()))
    cb1 = fig.colorbar(ims[0], ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    cb1.set_label(r"$\log_{10}\rho$ [$M_\odot\,\mathrm{kpc^{-3}}$]")
    cb2 = fig.colorbar(im_res, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
    cb2.set_label(r"$\log_{10}(\rho/\rho_{\rm true})$ [dex]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "s1_density_2d.npz",
             x_kpc=xs, r_grid=rr, in_domain=in_dom,
             r_nodes=r_nodes, rho_true_nodes=rho_true_nodes,
             rho_true_grid=rho_true_grid,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("rho", grids[l]), ("prof", profs[l]),
                             ("resid_dex", resid[l]), ("negfrac", negfrac[l]))})
    for p in ofs.save(fig, str(args.output_dir / "s1-density-2d")):
        print("saved", p)
    print("DENSITY_2D_DONE")


if __name__ == "__main__":
    sys.exit(main())
