#!/usr/bin/env python
"""2D potential / density slices: models, particle truth, and residuals.

The generic counterpart of the upstream ``plot_2d_slice_pot`` /
``plot_2d_slices_rho`` benchmark figures (potential_benchmarking.py), as a
standalone post-hoc CLI.  Two data kinds participate:

  models  trained Phi checkpoints, evaluated through the same autodiff as
          training (units recovered from the --input h5 attrs);
  truth   the committed particle-truth grid product (--truth, optional).

Without --truth this renders model-only slices (the upstream intent; the
smoke path).  With --truth each figure gains the truth panel and the
residual row, aligned by the sphere-mean constant c = <Phi_model - Phi_true>
over the shared radial nodes (zero convention Phi_sphere_mean(r_outer) = 0).

Figures (per quantity, plane x-z at y=0 — the only plane the grid product
carries; xy/yz are available in model-only mode):
  2d_slice_phi[_truth]   row 0: truth + models, shared robust scale;
                         row 1: radial sphere-mean profile (truth band from
                         per-direction spread) + pointwise residual maps,
                         symmetric diverging scale.
  2d_slice_rho[_truth]   same layout in log10 rho; truth is the 96^3
                         midplane cell of the mass histogram (cell average),
                         model rho is the Laplacian evaluated at the same
                         cell centres (point value) — a second-order
                         discretisation mismatch, stated rather than hidden.

Data-level conventions follow the upstream project: figures go to --fig-dir
(png by default, --fig-fmt to override), array products are persisted as
npz alongside for later reading without model reloads.

Run from the repo root (CPU is fine; models are f32):
  python scripts/plot_potential_2d.py --input data/auriga/halo12.h5 \
      --model base=runs/w1024full --truth data/auriga/halo12_particle_truth_grids.h5 \
      --fig-dir runs/w1024full/plots
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
HERE = REPO / "scripts"
AURIGA = HERE / "auriga"
for p in (str(HERE), str(AURIGA)):
    if p not in sys.path:
        sys.path.insert(0, p)

from validate_enclosed_mass import (  # noqa: E402
    sobol_directions, make_radial_nodes, rho_from_phi)
import orx_figstyle as ofs  # noqa: E402


def load_units(input_fname):
    with h5py.File(input_fname, "r") as f:
        attrs = dict(f.attrs)
    return (float(attrs["length_scale_kpc"]),
            float(attrs["velocity_scale_kms"]))


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


def batch_eval(func, q, batch=65536):
    import jax.numpy as jnp
    out = np.empty(q.shape[0], dtype=np.float64)
    for i in range(0, q.shape[0], batch):
        out[i:i + batch] = np.asarray(func(jnp.asarray(q[i:i + batch])))
    return out


def parse_models(specs):
    out = []
    for spec in specs:
        label, _, run_dir = spec.partition("=")
        out.append((label.strip(), Path(run_dir.strip())))
    return out


def plane_coords(plane, xs):
    """Meshgrid + code-unit query points for 'xz' (y=0), 'xy' (z=0), 'yz' (x=0)."""
    import numpy as np
    if plane == "xz":
        d1, d2, fixed, ax = xs, xs, 1, ("x [kpc]", "z [kpc]")
    elif plane == "xy":
        d1, d2, fixed, ax = xs, xs, 2, ("x [kpc]", "y [kpc]")
    elif plane == "yz":
        d1, d2, fixed, ax = xs, xs, 0, ("y [kpc]", "z [kpc]")
    else:
        raise SystemExit(f"unknown plane {plane!r}")
    a, b = np.meshgrid(d1, d2, indexing="xy")
    q = np.zeros((a.size, 3))
    order = [0, 1, 2]
    order.pop(fixed)
    q[:, order[0]] = a.ravel()
    q[:, order[1]] = b.ravel()
    return a, b, q, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="training/eta h5 (attrs carry the physical units)")
    ap.add_argument("--model", action="append", default=[],
                    metavar="NAME=RUN_DIR",
                    help="trained potential run dir; repeatable")
    ap.add_argument("--truth", type=Path, default=None,
                    help="particle-truth grid product h5 (optional)")
    ap.add_argument("--quantity", choices=("phi", "rho", "both"),
                    default="both" if True else "phi")
    ap.add_argument("--plane", choices=("xz", "xy", "yz"), default="xz",
                    help="slice plane; truth comparison only supports xz "
                         "(the grid product carries y=0 only)")
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--fig-fmt", default="png",
                    help="figure format, upstream convention is png")
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--r-inner", type=float, default=0.0,
                    help="mask r < this in maps (truth mode uses the grids "
                         "anchor 1.0906 regardless)")
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-sphere-nodes", type=int, default=24)
    ap.add_argument("--save-npz", action="store_true", default=True)
    args = ap.parse_args()

    models = parse_models(args.model)
    if not models and args.truth is None:
        ap.error("need at least one --model or a --truth product")
    if args.truth is not None and args.plane != "xz":
        ap.error("--truth comparison only supports --plane xz "
                 "(the grid product carries the y=0 slice only)")

    import jax
    import matplotlib as mpl
    L, V = load_units(args.input)
    print(f"units from {args.input}: L={L} kpc, V={V} km/s")

    t_start = time.time()
    truth = None
    r_anchor = args.r_inner
    if args.truth is not None:
        with h5py.File(args.truth, "r") as f:
            assert f.attrs["schema"] == "dpjax.particle-truth-grids.v1", \
                f"{args.truth} is not a particle-truth grid product"
            truth = {
                "phi_slice": np.asarray(f["potential/phi_slice"][:], dtype=np.float64),
                "xs": np.asarray(f["potential/x_kpc"][:], dtype=np.float64),
                "r_nodes": np.asarray(f["potential/r_nodes"][:], dtype=np.float64),
                "phi_prof": np.asarray(f["potential/phi_sphere_mean"][:], dtype=np.float64),
                "phi_dirs": np.asarray(f["potential/phi_dirs"][:], dtype=np.float64),
                "rho3d": np.asarray(f["density/rho3d"][:], dtype=np.float64),
                "rho3d_edges": np.asarray(f["density/rho3d_edges_kpc"][:], dtype=np.float64),
                "rho_r": np.asarray(f["density/rho_r"][:], dtype=np.float64),
                "rho_r_edges": np.asarray(f["density/rho_r_edges_kpc"][:], dtype=np.float64),
            }
        r_anchor = float(truth["r_nodes"][0])
        print(f"truth grids read in {time.time()-t_start:.2f}s "
              f"(r_anchor={r_anchor:.4f} kpc)")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    phis = {label: load_phi_f32(rd) for label, rd in models}

    ofs.use_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, stem):
        for fmt in args.fig_fmt.split(","):
            path = args.fig_dir / f"{stem}.{fmt.strip()}"
            fig.savefig(path, dpi=200 if fmt.strip() == "png" else None)
            print("saved", path)

    quantities = ("phi", "rho") if args.quantity == "both" else (args.quantity,)
    for quantity in quantities:
        suffix = "_truth" if truth is not None else ""
        if quantity == "phi":
            _figure_phi(args, models, phis, truth, dirs, L, V, r_anchor,
                        save_fig, suffix)
        else:
            _figure_rho(args, models, phis, truth, dirs, L, V, r_anchor,
                        save_fig, suffix)
    print(f"PLOT_POTENTIAL_2D_DONE total {time.time()-t_start:.1f}s")


# --------------------------------------------------------------------------
# phi figure
# --------------------------------------------------------------------------

def _figure_phi(args, models, phis, truth, dirs, L, V, r_anchor, save_fig, suffix):
    import jax
    import matplotlib as mpl

    # model evaluation grid: truth coordinates when present (exact alignment)
    if truth is not None:
        xs = truth["xs"]
        r_nodes = truth["r_nodes"]
    else:
        xs = np.linspace(-args.r_outer, args.r_outer, args.n_grid)
        r_nodes = np.asarray(make_radial_nodes(
            max(args.r_inner, 1e-3), args.r_outer, args.n_sphere_nodes - 1),
            dtype=float)
    a, b, q, (xlab, ylab) = plane_coords(args.plane, xs)
    rr = np.hypot(a, b)

    slices, profs, consts = {}, {}, {}
    for label, _ in models:
        t0 = time.time()
        phi = phis[label]
        grid = batch_eval(jax.vmap(phi), q / L).reshape(a.shape) * V ** 2
        nodes = np.empty(r_nodes.size)
        for j, r in enumerate(r_nodes):
            nodes[j] = float(np.mean(np.asarray(
                jax.vmap(phi)(jax.numpy.asarray(dirs * (r / L)))))) * V ** 2
        if truth is not None:
            c = float(np.mean(nodes - truth["phi_prof"]))
        else:
            # self-convention: sphere mean at r_outer is the zero
            c = float(nodes[-1])
        slices[label] = grid - c
        profs[label] = nodes - c
        consts[label] = c
        print(f"[phi {label}] slice+profile {time.time()-t0:.1f}s; c={c:.1f}")

    if truth is not None:
        true_map = truth["phi_slice"]
        prof_true = truth["phi_prof"]
        dirs_true = truth["phi_dirs"]
        lo, hi = np.nanpercentile(truth["phi_dirs"], [15.85, 84.15], axis=1)
    else:
        true_map, prof_true, dirs_true, lo, hi = (None,) * 5

    valid = (rr >= (r_anchor if truth is not None else args.r_inner)) & \
            (rr <= args.r_outer)
    n = len(models)
    ncols = n + (1 if truth is not None else 0)
    if ncols == 0:
        return

    if truth is not None:
        vmin = min([np.nanmin(np.where(valid, slices[l], np.nan)) for l, _ in models]
                   + [np.nanmin(np.where(valid, true_map, np.nan))])
        vmax = max([np.nanmax(np.where(valid, slices[l], np.nan)) for l, _ in models]
                   + [np.nanmax(np.where(valid, true_map, np.nan))])
    else:
        vmin = min(np.nanmin(np.where(valid, slices[l], np.nan)) for l, _ in models)
        vmax = max(np.nanmax(np.where(valid, slices[l], np.nan)) for l, _ in models)
    print(f"shared Phi scale [{vmin:.1f}, {vmax:.1f}] (km/s)^2")

    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)

    rows = 2
    fig, axes = ofs.figure_grid(rows, max(ncols, 2), width=ofs.WIDE, ratio=0.58)
    axes = np.atleast_2d(axes)
    col = 0
    im0 = None
    if truth is not None:
        im0 = axes[0, col].pcolormesh(xs, xs, np.where(valid, true_map, np.nan),
                                      cmap=cmap_seq, vmin=vmin, vmax=vmax,
                                      rasterized=True, shading="auto")
        axes[0, col].set_title("truth (particles)", fontsize=8)
        col += 1
    for label, _ in models:
        axes[0, col].pcolormesh(xs, xs, np.where(valid, slices[label], np.nan),
                                cmap=cmap_seq, vmin=vmin, vmax=vmax,
                                rasterized=True, shading="auto")
        axes[0, col].set_title(label, fontsize=8)
        col += 1
    for c_ in range(col, axes.shape[1]):
        axes[0, c_].set_visible(False)

    axp = axes[1, 0]
    if truth is not None:
        axp.fill_between(r_nodes, lo, hi, color="0.85", label="truth 68% dirs")
        axp.plot(r_nodes, prof_true, color="0.2", lw=1.6, label="truth (particles)")
    for j, (label, _) in enumerate(models):
        axp.plot(r_nodes, profs[label], color=ofs.CYCLE[j % len(ofs.CYCLE)], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\langle\Phi\rangle_\Omega$ [$(\mathrm{km/s})^2$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower right")

    if truth is not None:
        resid = {l: slices[l] - true_map for l, _ in models}
        rabs = None
        if resid:
            rabs = float(np.nanpercentile(np.abs(np.stack(
                [np.where(valid, resid[l], np.nan) for l, _ in models])), 98))
            print(f"pointwise residual +-p98 = {rabs:.1f} (km/s)^2")
        im_res = None
        for j, (label, _) in enumerate(models):
            im_res = axes[1, j + 1].pcolormesh(
                xs, xs, np.where(valid, resid[label], np.nan),
                cmap=cmap_div, vmin=-rabs, vmax=rabs,
                rasterized=True, shading="auto")
        if im_res is not None:
            cb2 = fig.colorbar(im_res, ax=axes[1, 1:].tolist(),
                               fraction=0.026, pad=0.02)
            cb2.set_label(r"$\Phi_m - \Phi_{\rm true}$")
        for c_ in range(len(models) + 1, axes.shape[1]):
            axes[1, c_].set_visible(False)
    else:
        for c_ in range(1, axes.shape[1]):
            axes[1, c_].set_visible(False)

    for i in range(rows):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if ax is axp:
                continue
            ax.set_aspect("equal")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
    ofs.panel_labels(list(axes.ravel()))
    if im0 is not None:
        cb1 = fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
        cb1.set_label(r"$\Phi - c$ [$(\mathrm{km/s})^2$]")
    save_fig(fig, f"2d_slice_phi{suffix}")
    if args.save_npz:
        np.savez(args.fig_dir / f"2d_slice_phi{suffix}.npz",
                 x_kpc=xs, r_nodes=r_nodes,
                 **({} if truth is None else
                    dict(phi_true=true_map, phi_true_prof=prof_true,
                         phi_true_lo=lo, phi_true_hi=hi)),
                 **{f"{l}_{k}": v for l, _ in models
                    for k, v in (("phi", slices[l]), ("c", consts[l]),
                                 ("prof", profs[l]))})


# --------------------------------------------------------------------------
# rho figure
# --------------------------------------------------------------------------

def _figure_rho(args, models, phis, truth, dirs, L, V, r_anchor, save_fig, suffix):
    import jax
    import matplotlib as mpl

    if truth is not None:
        ex = truth["rho3d_edges"][0]
        c3 = 0.5 * (ex[:-1] + ex[1:])
        jy = int(np.argmin(np.abs(c3)))
        y_c = float(c3[jy])
        rho_true_mid = truth["rho3d"][:, jy, :].T  # rows z, cols x
        xx3, zz3 = np.meshgrid(c3, c3, indexing="xy")
        rr = np.hypot(xx3, zz3)
        xs = c3
    else:
        xs = np.linspace(-args.r_outer, args.r_outer, min(args.n_grid, 128))
        a, b, q, _ = plane_coords(args.plane, xs)
        rr = np.hypot(a, b)

    valid = (rr >= (r_anchor if truth is not None else args.r_inner)) & \
            (rr <= args.r_outer)

    rho_models, negfrac = {}, {}
    for label, _ in models:
        phi = phis[label]
        if truth is not None:
            xx, zz = np.meshgrid(c3, c3, indexing="xy")
            qq = np.column_stack([xx.ravel(), np.full(xx.size, y_c),
                                  zz.ravel()]) / L
        else:
            qq = q / L
        out = np.empty(qq.shape[0])
        for i in range(0, qq.shape[0], 32768):
            out[i:i + 32768] = np.asarray(
                rho_from_phi(phi, qq[i:i + 32768], L, V))
        if truth is not None:
            g = out.reshape(xx.shape)
        else:
            g = out.reshape(a.shape)
        negfrac[label] = float(np.mean(g[valid] <= 0.0))
        rho_models[label] = g
        print(f"[rho {label}] Laplacian rho on {g.shape[0]}^2 plane; "
              f"non-positive cells in domain: {negfrac[label]*100:.1f}%")

    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)

    if truth is not None:
        valid3 = valid & (rho_true_mid > 0)
        logtrue = np.where(valid3, np.log10(rho_true_mid), np.nan)
        logmodels = {l: np.where(valid & (rho_models[l] > 0),
                                 np.log10(rho_models[l]), np.nan)
                     for l, _ in models}
        allvals = np.concatenate([logtrue.ravel()]
                                 + [logmodels[l].ravel() for l, _ in models])
        vmin, vmax = np.nanpercentile(allvals, 0.5), np.nanpercentile(allvals, 99.5)
        dres = None
        if models:
            dres = float(np.nanpercentile(np.abs(np.stack(
                [logmodels[l] - logtrue for l, _ in models])), 98))
    else:
        logmodels = {l: np.where(valid & (rho_models[l] > 0),
                                 np.log10(rho_models[l]), np.nan)
                     for l, _ in models}
        allvals = np.concatenate([logmodels[l].ravel() for l, _ in models])
        vmin, vmax = np.nanpercentile(allvals, 0.5), np.nanpercentile(allvals, 99.5)
        logtrue, dres = None, None

    n = len(models)
    ncols = n + (1 if truth is not None else 0)
    fig, axes = ofs.figure_grid(2, max(ncols, 2), width=ofs.WIDE, ratio=0.58)
    axes = np.atleast_2d(axes)
    col = 0
    im0 = None
    if truth is not None:
        im0 = axes[0, col].pcolormesh(xs, xs, logtrue, cmap=cmap_seq,
                                      vmin=vmin, vmax=vmax, rasterized=True,
                                      shading="auto")
        axes[0, col].set_title("truth (histogram)", fontsize=8)
        col += 1
    for label, _ in models:
        axes[0, col].pcolormesh(xs, xs, logmodels[label], cmap=cmap_seq,
                                vmin=vmin, vmax=vmax, rasterized=True,
                                shading="auto")
        axes[0, col].set_title(label, fontsize=8)
        col += 1
    for c_ in range(col, axes.shape[1]):
        axes[0, c_].set_visible(False)

    axp = axes[1, 0]
    if truth is not None:
        centers_r = 0.5 * (truth["rho_r_edges"][1:] + truth["rho_r_edges"][:-1])
        axp.loglog(centers_r, truth["rho_r"], color="0.2", lw=1.6,
                   label="truth (histogram)")
    r_nodes_rho = np.asarray(make_radial_nodes(
        max(r_anchor, 0.5), args.r_outer, 47), dtype=float)
    for j, (label, _) in enumerate(models):
        prof = np.empty(r_nodes_rho.size)
        for k, r in enumerate(r_nodes_rho):
            prof[k] = float(np.mean(np.asarray(
                rho_from_phi(phis[label], dirs * (r / L), L, V))))
        axp.loglog(r_nodes_rho, prof, color=ofs.CYCLE[j % len(ofs.CYCLE)], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower left")

    if truth is not None:
        im_res = None
        for j, (label, _) in enumerate(models):
            im_res = axes[1, j + 1].pcolormesh(
                xs, xs, logmodels[label] - logtrue, cmap=cmap_div,
                vmin=-dres, vmax=dres, rasterized=True, shading="auto")
            axes[1, j + 1].annotate(
                f"neg cells {negfrac[label]*100:.1f}%",
                xy=(0.02, 0.05), xycoords="axes fraction", fontsize=6.5,
                color="#333333", backgroundcolor="white")
        if im_res is not None:
            cb2 = fig.colorbar(im_res, ax=axes[1, 1:].tolist(),
                               fraction=0.026, pad=0.02)
            cb2.set_label(r"$\log_{10}(\rho_m/\rho_t)$")
        for c_ in range(len(models) + 1, axes.shape[1]):
            axes[1, c_].set_visible(False)
    else:
        for c_ in range(1, axes.shape[1]):
            axes[1, c_].set_visible(False)

    for i in range(2):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if ax is axp:
                continue
            ax.set_aspect("equal")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("z [kpc]")
    ofs.panel_labels(list(axes.ravel()))
    if im0 is not None:
        cb1 = fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
        cb1.set_label(r"$\log_{10}\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    save_fig(fig, f"2d_slice_rho{suffix}")
    if args.save_npz:
        np.savez(args.fig_dir / f"2d_slice_rho{suffix}.npz",
                 x_kpc=xs, r_nodes=r_nodes_rho,
                 **({} if truth is None else
                    dict(log_rho_true=logtrue)),
                 **{f"{l}_{k}": v for l, _ in models
                    for k, v in (("log_rho", logmodels[l]),
                                 ("negfrac", negfrac[l]))})


if __name__ == "__main__":
    sys.exit(main())
