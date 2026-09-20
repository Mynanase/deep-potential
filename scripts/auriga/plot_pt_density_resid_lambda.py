#!/usr/bin/env python
"""Density maps and log residuals vs the committed particle truth.

Reads the particle-truth node product data/auriga/halo12_particle_truth_grids.h5
ONLY for the truth: the 96^3 mass histogram midplane slab (genuinely
non-spherical; no particle asset, no smoothing).  Model densities come from
the potential Laplacian evaluated at the same cell centres.

Layout (2x4): row 1 - truth log rho plus three model log densities on the
y=0 midplane, shared sequential scale (non-positive model cells masked
grey, in-domain fraction annotated); row 2 - radial rho profile with the
60-shell cross-check and model sphere means (e), then log10(rho_m/rho_t)
residual maps (f-h) on a shared symmetric scale.  Valid domain r in
[1.0906, 70] kpc.  Truth density is a cell average (histogram); the model
value is a point evaluation at the cell centre - a second-order
discretization mismatch at 1.5625 kpc cells, stated rather than hidden.
Three models: base, innerA (lambda=1), lambda10 (lambda=10) - colors follow
the lambda pt-adjudication figure.
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    sobol_directions, make_radial_nodes, load_truth, rho_from_phi)
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
R_ANCHOR = 1.0906
MODEL_TEXT = {
    "base": "base (seed-0)",
    "innerA": "innerA (lambda=1)",
    "lambda10": "lambda=10",
}
COLOR = {"base": "blue", "innerA": "purple",
         "lambda10": "orange"}


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


def annotate_name(ax, name):
    ax.annotate(name, xy=(0.02, 0.97), xycoords="axes fraction", ha="left",
                va="top", fontsize=7, color="#333333", backgroundcolor="white")


def rho_grid(phi, xs, zs, y_const, batch=32768):
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    q = np.column_stack([xx.ravel(), np.full(xx.size, y_const), zz.ravel()]) / L_KPC
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path, required=True,
                    help="halo12_particle_truth_grids.h5 (node product)")
    ap.add_argument("--model", action="append", required=True,
                    help="label=run_dir; pass exactly three")
    ap.add_argument("--truth", type=Path, default=None,
                    help="60-shell truth hdf5 (profile cross-check only)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()

    specs = dict(spec.split("=", 1) for spec in args.model)
    if len(specs) != 3:
        raise SystemExit(f"need exactly three models, got {sorted(specs)}")
    models = tuple(specs)

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    t0 = time.time()
    with h5py.File(args.grids, "r") as f:
        assert f.attrs["schema"] == "dpjax.particle-truth-grids.v1"
        rho3d = f["density/rho3d"][:]
        edges3d = np.asarray(f["density/rho3d_edges_kpc"][:])
        rho_r = np.asarray(f["density/rho_r"][:], dtype=np.float64)
        re_r = np.asarray(f["density/rho_r_edges_kpc"][:], dtype=np.float64)
        n_particles = int(f.attrs["n_particles"])
    ex3 = edges3d[0]
    c3 = 0.5 * (ex3[:-1] + ex3[1:])
    jy = int(np.argmin(np.abs(c3)))
    y_c = float(c3[jy])
    rho_true_mid = rho3d[:, jy, :].T.astype(np.float64)  # rows z, cols x
    xx3, zz3 = np.meshgrid(c3, c3, indexing="xy")
    rr3 = np.hypot(xx3, zz3)
    in_dom = (rr3 >= R_ANCHOR) & (rr3 <= args.r_outer)
    valid3 = in_dom & (rho_true_mid > 0)
    print(f"grids product read in {time.time()-t0:.2f}s "
          f"(n={n_particles}); midplane y={y_c:.3f} kpc, {c3.size}^2 cells; "
          f"truth rho range in domain "
          f"[{rho_true_mid[in_dom].min():.3e}, {rho_true_mid[in_dom].max():.3e}]")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    r_nodes_rho = np.asarray(make_radial_nodes(
        R_ANCHOR, args.r_outer, 47), dtype=float)
    rho_models, profs, negfrac = {}, {}, {}
    for label in models:
        phi = load_phi_f32(specs[label])
        t0 = time.time()
        g = rho_grid(phi, c3, c3, y_c)
        negfrac[label] = float(np.mean(g[in_dom] <= 0.0))
        rho_models[label] = g
        profs[label] = rho_sphere_mean(phi, r_nodes_rho, dirs)
        print(f"[{label}] 96^2 midplane + 48x{args.n_dirs} sphere nodes in "
              f"{time.time()-t0:.0f}s; non-positive cells in domain "
              f"{negfrac[label]*100:.1f}%; <rho>(1.09)={profs[label][0]:.3e}")

    logtrue = np.where(valid3, np.log10(rho_true_mid), np.nan)
    logmodels = {l: np.where(valid3 & (rho_models[l] > 0),
                             np.log10(rho_models[l]), np.nan) for l in models}
    allvals = np.concatenate([logtrue.ravel()]
                             + [logmodels[l].ravel() for l in models])
    vmin = float(np.nanpercentile(allvals, 0.5))
    vmax = float(np.nanpercentile(allvals, 99.5))
    dres = float(np.nanpercentile(np.abs(np.stack(
        [logmodels[l] - logtrue for l in models])), 98))
    print(f"shared scales: log10 rho in [{vmin:.2f}, {vmax:.2f}]; "
          f"log-ratio residual +/-{dres:.2f} dex (98th pct)")

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, 4, width=ofs.WIDE, ratio=0.58)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)
    ims = [axes[0, 0].pcolormesh(c3, c3, logtrue, cmap=cmap_seq, vmin=vmin,
                                 vmax=vmax, rasterized=True, shading="auto")]
    for j, l in enumerate(models):
        ims.append(axes[0, j + 1].pcolormesh(
            c3, c3, logmodels[l], cmap=cmap_seq, vmin=vmin, vmax=vmax,
            rasterized=True, shading="auto"))
    axp = axes[1, 0]
    centers_r = 0.5 * (re_r[1:] + re_r[:-1])
    axp.loglog(centers_r, rho_r, color="0.2", lw=1.6,
               label="truth (histogram)")
    if args.truth is not None and Path(args.truth).is_file():
        truth = load_truth(args.truth)
        r_c = np.asarray(truth["r_center"], dtype=float)
        r_e = np.asarray(truth["r_edges"], dtype=float)
        rho_c = np.asarray(truth["M_shell_total"], dtype=float) / \
            (4.0 / 3.0 * np.pi * (r_e[1:] ** 3 - r_e[:-1] ** 3))
        m60 = (r_c >= 1.0) & (r_c <= args.r_outer)
        axp.plot(r_c[m60], rho_c[m60], "x", color="0.5", ms=4, mew=1.0,
                 label="60-shell truth")
    for l in models:
        axp.loglog(r_nodes_rho, profs[l], color=ofs.PALETTE[COLOR[l]], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower left")
    im_res = None
    for j, l in enumerate(models):
        im_res = axes[1, j + 1].pcolormesh(
            c3, c3, logmodels[l] - logtrue, cmap=cmap_div, vmin=-dres,
            vmax=dres, rasterized=True, shading="auto")
        axes[1, j + 1].annotate(f"neg cells {negfrac[l]*100:.1f}%",
                                xy=(0.02, 0.05), xycoords="axes fraction",
                                fontsize=6.5, color="#333333",
                                backgroundcolor="white")
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            if ax is axp:
                continue
            ax.set_aspect("equal")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("z [kpc]")
    annotate_name(axes[0, 0], "truth (particles)")
    for j, l in enumerate(models):
        annotate_name(axes[0, j + 1], MODEL_TEXT.get(l, l))
    ofs.panel_labels(list(axes.ravel()))
    cb1 = fig.colorbar(ims[0], ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    cb1.set_label(r"$\log_{10}\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    cb2 = fig.colorbar(im_res, ax=axes[1, 1:].tolist(), fraction=0.026, pad=0.02)
    cb2.set_label(r"$\log_{10}(\rho_m/\rho_t)$")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / f"{args.output_dir.name}.npz",
             c_kpc=c3, y_mid_kpc=y_c, log_true=logtrue, dres=dres,
             rho_r_centers=centers_r, rho_r=rho_r,
             **{f"{l}_{k}": v for l in models
                for k, v in (("logmodel", logmodels[l]), ("negfrac", negfrac[l]),
                             ("prof", profs[l]))})
    stem = str(args.output_dir / args.output_dir.name)
    for p in ofs.save(fig, stem):
        print("saved", p)
    fig.savefig(stem + "_preview.png", dpi=150)
    print("PT_DENSITY_RESID_VARIANT_DONE")


if __name__ == "__main__":
    sys.exit(main())
