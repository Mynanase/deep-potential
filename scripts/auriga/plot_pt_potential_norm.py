#!/usr/bin/env python
"""Merged potential figure with vc2-normalized residuals.

Replaces the raw dPhi residual maps and the standalone resid figure.
Normalization: dPhi / vc2(r) with vc2(r) = r |d<Phi_true>_Omega/dr| =
G M_true(<r)/r (monopole theorem; the sphere-mean profile IS the l=0
component). Unlike epsilon = dPhi/|Phi_sim(r)| this is finite everywhere:
the r=70 kpc zero convention makes the last profile node exactly zero and
the map has a Phi zero-crossing contour near the outer boundary.

One 3-row figure (GridSpec 3x12):
  row 1  truth + 3 aligned model Phi maps, shared scale;
  row 2  3 dPhi/vc2 residual maps + the pointwise residual histogram
         (normalized values, same quantity as the maps);
  row 3  <Phi>_Omega profiles, truth sigma_Omega, Delta<Phi>_Omega, side
         by side.
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

from validate_enclosed_mass import G_KPC_KMS2_MSUN, sobol_directions  # noqa: E402
from plot_particle_truth_2d import load_phi_f32, batch_eval, annotate_name  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "s11", "S1")
MODEL_TEXT = {"base": "base (seed-0)", "s11": "s11 (reshuffle)", "S1": "S1 (stratified)"}
COLOR = {"base": "blue", "s11": "orange", "S1": "red"}
R_ANCHOR = 1.0906


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/pt-potential-norm"))
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    t_start = time.time()
    with h5py.File(args.grids, "r") as f:
        phi_slice = f["potential/phi_slice"][:]
        xs = f["potential/x_kpc"][:]
        r_nodes = f["potential/r_nodes"][:]
        phi_prof = f["potential/phi_sphere_mean"][:]
        phi_dirs = f["potential/phi_dirs"][:].astype(np.float64)
        rho_r = f["density/rho_r"][:]
        re_r = f["density/rho_r_edges_kpc"][:]
    print(f"grids product read in {time.time()-t_start:.2f}s")

    print("=== vc2 from the truth monopole profile ===")
    u = np.log(r_nodes)
    vc2 = np.abs(np.gradient(phi_prof, u))  # r|dPhi/dr| = G M(<r)/r
    centers = 0.5 * (re_r[1:] + re_r[:-1])
    m_cum_r = np.cumsum(rho_r * 4.0 / 3.0 * np.pi * (re_r[1:] ** 3 - re_r[:-1] ** 3))
    for r in (2.0, 8.0, 20.0, 50.0):
        i = int(np.argmin(np.abs(r_nodes - r)))
        m_mono = vc2[i] * r_nodes[i] / G_KPC_KMS2_MSUN
        m_cum = float(np.interp(r_nodes[i], centers, m_cum_r))
        print(f"vc2({r_nodes[i]:.2f})={vc2[i]:.4e} (km/s)^2, vc={np.sqrt(vc2[i]):.1f} km/s; "
              f"M_mono={m_mono:.3e} Msun vs rho_r cum (misses r<0.5) {m_cum:.3e}, "
              f"ratio {m_mono/m_cum:.4f}")
    print(f"zero-convention check: profile node values at r="
          f"{r_nodes[-1]:.1f}/{r_nodes[-2]:.1f} kpc = "
          f"{phi_prof[-1]:.3e}/{phi_prof[-2]:.3e} -> per-bin |Phi| relative residual "
          "undefined at the last node")
    xx, zz = np.meshgrid(xs, xs, indexing="xy")
    rr = np.hypot(xx, zz)
    valid = (rr >= R_ANCHOR) & (rr <= args.r_outer)
    tiny = int(np.sum(valid & (np.abs(phi_slice) < 0.01 * abs(phi_prof[0]))))
    print(f"map cells in domain with |Phi_true| < 1% |Phi(1.09)|: {tiny} "
          "(zero-crossing band near r=70)")

    print("=== models (aligned to the truth sphere-mean profile) ===")
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    slices, consts, profs = {}, {}, {}
    import jax.numpy as jnp
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        q = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]) / L_KPC
        grid = batch_eval(jax.vmap(phi), q).reshape(xx.shape) * V_KMS ** 2
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
        print(f"[{label}] c={c:.1f}; sphere-mean RMS vs truth={rms:.1f} "
              f"(handoff 2907/3046/3022) in {time.time()-t0:.0f}s")

    vc2_map = np.interp(rr, r_nodes, vc2)
    resid = {l: slices[l] - phi_slice for l in MODELS}
    resid_n = {l: resid[l] / vc2_map for l in MODELS}
    nabs = float(np.nanpercentile(np.abs(np.stack(
        [np.where(valid, resid_n[l], np.nan) for l in MODELS])), 98))
    rms_n = {l: float(np.sqrt(np.nanmean(np.where(valid, resid_n[l], np.nan) ** 2)))
             for l in MODELS}
    print("normalized residual scale: +-{:.4f} (p98, dimensionless; {:.1f}%)".format(
        nabs, 100 * nabs)
        + "; RMS " + "/".join(f"{l} {100*rms_n[l]:.1f}%" for l in MODELS))

    ofs.use_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)

    fig = plt.figure(figsize=(ofs.WIDE, 5.9))
    gs = fig.add_gridspec(3, 12, height_ratios=(1.0, 1.0, 0.78),
                          hspace=0.55, wspace=0.45)
    ax_maps = [fig.add_subplot(gs[0, 3 * j:3 * j + 3]) for j in range(4)]
    ax_hist = fig.add_subplot(gs[1, 0:3])
    ax_res = [fig.add_subplot(gs[1, 3 * (j + 1):3 * (j + 1) + 3]) for j in range(3)]
    ax_prof = fig.add_subplot(gs[2, 0:4])
    ax_sig = fig.add_subplot(gs[2, 4:8])
    ax_dprof = fig.add_subplot(gs[2, 8:12])

    vmin = min(np.nanmin(np.where(valid, s, np.nan))
               for s in list(slices.values()) + [phi_slice])
    vmax = max(np.nanmax(np.where(valid, s, np.nan))
               for s in list(slices.values()) + [phi_slice])
    im0 = ax_maps[0].pcolormesh(xs, xs, np.where(valid, phi_slice, np.nan),
                                cmap=cmap_seq, vmin=vmin, vmax=vmax,
                                rasterized=True, shading="auto")
    for j, l in enumerate(MODELS):
        ax_maps[j + 1].pcolormesh(xs, xs, np.where(valid, slices[l], np.nan),
                                  cmap=cmap_seq, vmin=vmin, vmax=vmax,
                                  rasterized=True, shading="auto")
    for j, name in enumerate(["truth (particles)"] + [MODEL_TEXT[l] for l in MODELS]):
        annotate_name(ax_maps[j], name)
    im_res = None
    for j, l in enumerate(MODELS):
        im_res = ax_res[j].pcolormesh(xs, xs, np.where(valid, resid_n[l], np.nan),
                                      cmap=cmap_div, vmin=-nabs, vmax=nabs,
                                      rasterized=True, shading="auto")
        annotate_name(ax_res[j], f"RMS {100*rms_n[l]:.1f}%")
    bins = np.linspace(-2 * nabs, 2 * nabs, 81)
    for l in MODELS:
        vals = np.where(valid, resid_n[l], np.nan).ravel()
        ax_hist.hist(vals[~np.isnan(vals)], bins=bins, histtype="step", lw=1.1,
                     color=ofs.PALETTE[COLOR[l]], label=f"{l} {100*rms_n[l]:.1f}%")
    ax_hist.axvline(0, color="0.2", lw=0.8)
    ax_hist.set_xlabel(r"$\Delta\Phi/v_c^2$")
    ax_hist.set_ylabel("cells")
    ax_hist.tick_params(labelsize=5.5)
    ax_hist.legend(frameon=False, fontsize=5.5, loc="upper left")

    ax_prof.plot(r_nodes, phi_prof, color="0.2", lw=1.6, label="truth (particles)")
    for l in MODELS:
        ax_prof.plot(r_nodes, profs[l], color=ofs.PALETTE[COLOR[l]], lw=1.2)
    ax_prof.set_xlabel("r [kpc]")
    ax_prof.set_ylabel(r"$\langle\Phi\rangle_\Omega$ [$(\mathrm{km/s})^2$]")
    ax_prof.legend(frameon=False, fontsize=6.5, loc="upper left")
    sig_r = phi_dirs.std(axis=1) / np.abs(phi_dirs.mean(axis=1))
    ax_sig.plot(r_nodes, 100 * sig_r, color="0.2", lw=1.4)
    ax_sig.set_xlabel("r [kpc]")
    ax_sig.set_ylabel(r"$\sigma_\Omega/|\langle\Phi\rangle|$ [%]")
    for l in MODELS:
        ax_dprof.plot(r_nodes, profs[l] - phi_prof, color=ofs.PALETTE[COLOR[l]], lw=1.2)
    ax_dprof.axhline(0, color="0.2", lw=0.8)
    ax_dprof.set_xlabel("r [kpc]")
    ax_dprof.set_ylabel(r"$\Delta\langle\Phi\rangle_\Omega$ [$(\mathrm{km/s})^2$]")

    for ax in list(ax_maps) + list(ax_res):
        ax.set_aspect("equal")
        ax.set_xlabel("x [kpc]", fontsize=6)
        ax.set_ylabel("z [kpc]", fontsize=6)
        ax.tick_params(labelsize=6)
    ofs.panel_labels(list(ax_maps) + [ax_hist] + list(ax_res)
                     + [ax_prof, ax_sig, ax_dprof])
    cb1 = fig.colorbar(im0, ax=ax_maps, fraction=0.02, pad=0.02)
    cb1.set_label(r"$\Phi-c$ [$(\mathrm{km/s})^2$]")
    cb1.ax.tick_params(labelsize=5.5)
    cb2 = fig.colorbar(im_res, ax=[ax_hist] + ax_res, fraction=0.02, pad=0.02)
    cb2.set_label(r"$\Delta\Phi/v_c^2(r)$")
    cb2.ax.tick_params(labelsize=5.5)

    np.savez(args.output_dir / "pt_potential_norm.npz",
             x_kpc=xs, valid=valid, r_nodes=r_nodes, vc2=vc2,
             phi_true_prof=phi_prof, phi_true_grid=phi_slice,
             n_scale=nabs,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("phi", slices[l]), ("c", consts[l]),
                             ("prof", profs[l]), ("resid_n", resid_n[l]))})
    for p in ofs.save(fig, str(args.output_dir / "pt-potential-slices")):
        print("saved", p)
    print(f"TOTAL WALL {time.time()-t_start:.1f}s")
    print("PT_NORM_DONE")


if __name__ == "__main__":
    sys.exit(main())
