#!/usr/bin/env python
"""Three 2D figures from the committed particle-truth grid product.

Reads data/auriga/halo12_particle_truth_grids.h5 ONLY (no particle asset,
no GPU direct summation). Models are evaluated from their checkpoints.

Figures (all in the stellar principal-axis frame, meridional y=0 slice):
  A pt-density-2d      - truth rho from the 96^3 mass histogram (midplane
                        slab bin) vs model Laplacian densities, radial rho_r
                        profile with the 60-shell cross-check, log residuals.
  S pt-density-slab    - truth surface density Sigma(x,z), |y|<=0.5 kpc slab
                        histogram at 480^2 (the zero-processing map).
  B pt-potential-slices- truth Phi slice + aligned models + radial profiles,
                        truth angular fluctuation sigma_Omega, sphere-mean
                        residuals, pointwise residual histograms.
  C pt-potential-resid - pointwise Phi_m - Phi_true maps, shared scale.

Truth density is a cell average (histogram); model density is evaluated at
the same cell centres - a point value. At 1.5625 kpc cells this discretization
mismatch is second order and is stated here rather than hidden.
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
MODELS = ("base", "s11", "S1")
MODEL_TEXT = {"base": "base (seed-0)", "s11": "s11 (reshuffle)", "S1": "S1 (stratified)"}
COLOR = {"base": "blue", "s11": "orange", "S1": "red"}
R_ANCHOR = 1.0906


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


def annotate_name(ax, name):
    ax.annotate(name, xy=(0.02, 0.97), xycoords="axes fraction", ha="left",
                va="top", fontsize=7, color="#333333", backgroundcolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--truth", type=Path, default=None,
                    help="60-shell truth hdf5 (reference points only)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/particle-truth-2d"))
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    t_start = time.time()
    with h5py.File(args.grids, "r") as f:
        assert f.attrs["schema"] == "dpjax.particle-truth-grids.v1"
        phi_slice = f["potential/phi_slice"][:]
        xs_phi = f["potential/x_kpc"][:]
        r_nodes = f["potential/r_nodes"][:]
        phi_prof = f["potential/phi_sphere_mean"][:]
        phi_dirs = f["potential/phi_dirs"][:].astype(np.float64)
        rho3d = f["density/rho3d"][:]
        edges3 = f["density/rho3d_edges_kpc"][:]
        rho_r = f["density/rho_r"][:]
        re_r = f["density/rho_r_edges_kpc"][:]
        sigma_slab = f["density/sigma_slab"][:]
        esx = f["density/sigma_slab_edges_x_kpc"][:]
        esz = f["density/sigma_slab_edges_z_kpc"][:]
        n_particles = int(f.attrs["n_particles"])
    print(f"grids product read in {time.time()-t_start:.2f}s "
          f"(n_particles attr {n_particles}); no particle asset touched")

    # --- model evaluation -------------------------------------------------
    phis, slices, consts, profs = {}, {}, {}, {}
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        phis[label] = phi
        xx, zz = np.meshgrid(xs_phi, xs_phi, indexing="xy")
        q = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]) / L_KPC
        grid = batch_eval(jax.vmap(phi), q).reshape(xx.shape) * V_KMS ** 2
        import jax.numpy as jnp
        val = jax.vmap(phi)
        nodes = np.empty(r_nodes.size)
        for j, r in enumerate(r_nodes):
            nodes[j] = float(np.mean(np.asarray(val(
                jnp.asarray(dirs * (r / L_KPC)))))) * V_KMS ** 2
        c = float(np.mean(nodes - phi_prof))
        slices[label] = grid - c
        consts[label] = c
        profs[label] = nodes - c
        rms = float(np.sqrt(np.mean((nodes - c - phi_prof) ** 2)))
        print(f"[{label}] slice+profile in {time.time()-t0:.0f}s; c={c:.1f} "
              f"(handoff 2.710e5/2.347e5/2.173e5); sphere-mean RMS vs truth="
              f"{rms:.1f} (handoff 2907/3046/3022)")

    valid_phi = (np.hypot(*np.meshgrid(xs_phi, xs_phi, indexing="xy")) >= R_ANCHOR) & \
                (np.hypot(*np.meshgrid(xs_phi, xs_phi, indexing="xy")) <= args.r_outer)
    resid = {l: slices[l] - phi_slice for l in MODELS}
    rabs = float(np.nanpercentile(np.abs(np.stack(
        [np.where(valid_phi, resid[l], np.nan) for l in MODELS])), 98))
    print(f"pointwise residual +-p98 = {rabs:.1f} (km/s)^2 (handoff 7374.6)")

    # --- density: truth midplane from the histogram product ----------------
    c3 = 0.5 * (edges3[:-1] + edges3[1:])
    jy = int(np.argmin(np.abs(c3)))
    y_c = float(c3[jy])
    rho_true_mid = rho3d[:, jy, :].T.astype(np.float64)  # rows z, cols x
    xx3, zz3 = np.meshgrid(c3, c3, indexing="xy")
    rr3 = np.hypot(xx3, zz3)
    valid3 = (rr3 >= R_ANCHOR) & (rr3 <= args.r_outer) & (rho_true_mid > 0)
    rho_models, negfrac = {}, {}
    for spec in args.model:
        label, _ = spec.split("=", 1)
        g = rho_grid(phis[label] if label in phis else load_phi_f32(spec.split("=", 1)[1]),
                     c3, c3, y_c)
        in_dom = (rr3 >= R_ANCHOR) & (rr3 <= args.r_outer)
        negfrac[label] = float(np.mean(g[in_dom] <= 0.0))
        rho_models[label] = g
        print(f"[{label}] Laplacian rho on 96^2 midplane; non-positive cells "
              f"in domain: {negfrac[label]*100:.1f}%")

    ofs.use_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================== figure A: density ========================
    logtrue = np.where(valid3, np.log10(rho_true_mid), np.nan)
    logmodels = {l: np.where(valid3 & (rho_models[l] > 0),
                             np.log10(rho_models[l]), np.nan) for l in MODELS}
    allvals = np.concatenate([logtrue.ravel()]
                             + [logmodels[l].ravel() for l in MODELS])
    vmin, vmax = np.nanpercentile(allvals, 0.5), np.nanpercentile(allvals, 99.5)
    dres = np.nanpercentile(np.abs(np.stack(
        [logmodels[l] - logtrue for l in MODELS])), 98)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)
    fig, axes = ofs.figure_grid(2, 4, width=ofs.WIDE, ratio=0.58)
    ims = [axes[0, 0].pcolormesh(c3, c3, logtrue, cmap=cmap_seq, vmin=vmin,
                                 vmax=vmax, rasterized=True, shading="auto")]
    for j, l in enumerate(MODELS):
        ims.append(axes[0, j + 1].pcolormesh(c3, c3, logmodels[l], cmap=cmap_seq,
                                             vmin=vmin, vmax=vmax, rasterized=True,
                                             shading="auto"))
    axp = axes[1, 0]
    centers_r = 0.5 * (re_r[1:] + re_r[:-1])
    axp.loglog(centers_r, rho_r, color="0.2", lw=1.6, label="truth (histogram)")
    if args.truth is not None and Path(args.truth).is_file():
        truth = load_truth(args.truth)
        r_c = np.asarray(truth["r_center"], dtype=float)
        r_e = np.asarray(truth["r_edges"], dtype=float)
        rho_c = np.asarray(truth["M_shell_total"], dtype=float) / \
            (4.0 / 3.0 * np.pi * (r_e[1:] ** 3 - r_e[:-1] ** 3))
        m60 = (r_c >= 1.0) & (r_c <= args.r_outer)
        axp.plot(r_c[m60], rho_c[m60], "x", color="0.5", ms=4, mew=1.0,
                 label="60-shell truth")
    r_nodes_rho = np.asarray(make_radial_nodes(R_ANCHOR, args.r_outer, 47), dtype=float)
    for l in MODELS:
        axp.loglog(r_nodes_rho, rho_sphere_mean(phis[l], r_nodes_rho, dirs),
                   color=ofs.PALETTE[COLOR[l]], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower left")
    for j, l in enumerate(MODELS):
        im_res = axes[1, j + 1].pcolormesh(c3, c3, logmodels[l] - logtrue,
                                           cmap=cmap_div, vmin=-dres, vmax=dres,
                                           rasterized=True, shading="auto")
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            if ax is axp:
                continue
            ax.set_aspect("equal")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("z [kpc]")
    for j, name in enumerate(["truth (particles)"] + [MODEL_TEXT[l] for l in MODELS]):
        annotate_name(axes[0, j], name)
    for j, l in enumerate(MODELS):
        axes[1, j + 1].annotate(f"neg cells {negfrac[l]*100:.1f}%",
                                xy=(0.02, 0.05), xycoords="axes fraction",
                                fontsize=6.5, color="#333333",
                                backgroundcolor="white")
    ofs.panel_labels(list(axes.ravel()))
    cb1 = fig.colorbar(ims[0], ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    cb1.set_label(r"$\log_{10}\rho$ [$M_\odot/\mathrm{kpc}^3$]")
    cb2 = fig.colorbar(im_res, ax=axes[1, 1:].tolist(), fraction=0.026, pad=0.02)
    cb2.set_label(r"$\log_{10}(\rho_m/\rho_t)$")
    np.savez(args.output_dir / "pt_density_2d.npz",
             c_kpc=c3, y_mid_kpc=y_c, log_true=logtrue,
             **{f"{l}_logmodel": logmodels[l] for l in MODELS},
             **{f"{l}_negfrac": negfrac[l] for l in MODELS},
             rho_r_centers=centers_r, rho_r=rho_r, dres=dres)
    for p in ofs.save(fig, str(args.output_dir / "pt-density-2d")):
        print("saved", p)

    # ===================== figure S: slab surface density ==================
    figS, axS = ofs.figure(width=4.6, ratio=1.02)
    logS = np.where(sigma_slab.T > 0, np.log10(sigma_slab.T), np.nan)
    imS = axS.pcolormesh(esx, esz, logS, cmap=cmap_seq, rasterized=True,
                         shading="auto")
    axS.set_aspect("equal")
    axS.set_xlabel("x [kpc]")
    axS.set_ylabel("z [kpc]")
    axS.annotate(r"$\Sigma$, slab $|y|\leq0.5$ kpc", xy=(0.02, 0.97),
                 xycoords="axes fraction", ha="left", va="top", fontsize=7,
                 color="#333333", backgroundcolor="white")
    cbS = figS.colorbar(imS, ax=axS, fraction=0.046, pad=0.03)
    cbS.set_label(r"$\log_{10}\Sigma$ [$M_\odot/\mathrm{kpc}^2$]")
    for p in ofs.save(figS, str(args.output_dir / "pt-density-slab")):
        print("saved", p)

    # =================== figure B: potential slices ========================
    figB, axesB = ofs.figure_grid(2, 4, width=ofs.WIDE, ratio=0.58)
    vminB = min(np.nanmin(np.where(valid_phi, s, np.nan))
                for s in list(slices.values()) + [phi_slice])
    vmaxB = max(np.nanmax(np.where(valid_phi, s, np.nan))
                for s in list(slices.values()) + [phi_slice])
    imsB = [axesB[0, 0].pcolormesh(xs_phi, xs_phi,
                                   np.where(valid_phi, phi_slice, np.nan),
                                   cmap=cmap_seq, vmin=vminB, vmax=vmaxB,
                                   rasterized=True, shading="auto")]
    for j, l in enumerate(MODELS):
        imsB.append(axesB[0, j + 1].pcolormesh(
            xs_phi, xs_phi, np.where(valid_phi, slices[l], np.nan), cmap=cmap_seq,
            vmin=vminB, vmax=vmaxB, rasterized=True, shading="auto"))
    axpB = axesB[1, 0]
    axpB.plot(r_nodes, phi_prof, color="0.2", lw=1.6, label="truth (particles)")
    for l in MODELS:
        axpB.plot(r_nodes, profs[l], color=ofs.PALETTE[COLOR[l]], lw=1.2)
    axpB.set_xlabel("r [kpc]")
    axpB.set_ylabel(r"$\langle\Phi\rangle_\Omega$ [$(\mathrm{km/s})^2$]")
    axpB.legend(frameon=False, fontsize=6.5, loc="lower right")

    axS2 = axesB[1, 1]
    sig_r = phi_dirs.std(axis=1) / np.abs(phi_dirs.mean(axis=1))
    axS2.plot(r_nodes, sig_r * 100, color="0.2", lw=1.4)
    axS2.set_xlabel("r [kpc]")
    axS2.set_ylabel(r"$\sigma_\Omega/|\langle\Phi\rangle|$ [%]")
    axS2.annotate("truth angular\nfluctuation", xy=(0.62, 0.9),
                  xycoords="axes fraction", ha="center", fontsize=6.5,
                  color="#333333")

    axD = axesB[1, 2]
    for l in MODELS:
        axD.plot(r_nodes, profs[l] - phi_prof, color=ofs.PALETTE[COLOR[l]], lw=1.2)
    axD.axhline(0, color="0.2", lw=0.8)
    axD.set_xlabel("r [kpc]")
    axD.set_ylabel(r"$\Delta\langle\Phi\rangle_\Omega$")

    axH = axesB[1, 3]
    bins = np.linspace(-2 * rabs, 2 * rabs, 81)
    for l in MODELS:
        vals = np.where(valid_phi, resid[l], np.nan).ravel()
        axH.hist(vals[~np.isnan(vals)], bins=bins, histtype="step", lw=1.2,
                 color=ofs.PALETTE[COLOR[l]],
                 label=f"{l}: RMS {np.sqrt(np.nanmean(np.where(valid_phi, resid[l], np.nan)**2)):.0f}")
    axH.axvline(0, color="0.2", lw=0.8)
    axH.set_xlabel(r"$\Phi_m-\Phi_{\rm true}$ [$(\mathrm{km/s})^2$]")
    axH.set_ylabel("cells")
    axH.legend(frameon=False, fontsize=5.5, loc="upper left")
    for i in range(2):
        for j in range(4):
            ax = axesB[i, j]
            if ax in (axpB, axS2, axD, axH):
                continue
            ax.set_aspect("equal")
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("z [kpc]")
    for j, name in enumerate(["truth (particles)"] + [MODEL_TEXT[l] for l in MODELS]):
        annotate_name(axesB[0, j], name)
    ofs.panel_labels(list(axesB.ravel()))
    cbB = figB.colorbar(imsB[0], ax=axesB[0, :].tolist(), fraction=0.02, pad=0.02)
    cbB.set_label(r"$\Phi-c$ [$(\mathrm{km/s})^2$]")
    np.savez(args.output_dir / "pt_potential_2d.npz",
             x_kpc=xs_phi, valid=valid_phi, r_nodes=r_nodes,
             phi_true_prof=phi_prof, phi_true_grid=phi_slice,
             sigma_omega_rel=sig_r,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("phi", slices[l]), ("c", consts[l]),
                             ("prof", profs[l]))})
    for p in ofs.save(figB, str(args.output_dir / "pt-potential-slices")):
        print("saved", p)

    # =================== figure C: residual maps ===========================
    figC, axesC = ofs.figure_grid(1, 3, width=ofs.WIDE, ratio=0.38)
    imC = None
    for j, l in enumerate(MODELS):
        imC = axesC[j].pcolormesh(xs_phi, xs_phi,
                                  np.where(valid_phi, resid[l], np.nan),
                                  cmap=cmap_div, vmin=-rabs, vmax=rabs,
                                  rasterized=True, shading="auto")
        axesC[j].set_aspect("equal")
        axesC[j].set_xlabel("x [kpc]")
        axesC[j].set_ylabel("z [kpc]")
        rms = float(np.sqrt(np.nanmean(np.where(valid_phi, resid[l], np.nan) ** 2)))
        annotate_name(axesC[j], f"{MODEL_TEXT[l]}  RMS {rms:.0f}")
    ofs.panel_labels(list(axesC.ravel()))
    cbC = figC.colorbar(imC, ax=axesC.tolist(), fraction=0.03, pad=0.02)
    cbC.set_label(r"$\Phi_m-\Phi_{\rm true}$ [$(\mathrm{km/s})^2$]")
    for p in ofs.save(figC, str(args.output_dir / "pt-potential-resid")):
        print("saved", p)

    print(f"TOTAL WALL {time.time()-t_start:.1f}s (product-read fast path)")
    print("PT_FIGURES_DONE")


if __name__ == "__main__":
    sys.exit(main())
