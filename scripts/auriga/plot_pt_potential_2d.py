#!/usr/bin/env python
"""2D potential slices vs the PARTICLE total-matter truth.

Row 1: (a) truth potential map from direct particle summation (genuinely
non-spherical), (b-d) aligned model potentials. Row 2: (e) radial sphere-mean
profiles vs truth, (f-h) pointwise residuals. Zero convention: the truth
sphere mean at r=70 kpc is zero; model constants least-squares aligned to the
truth sphere-mean profile. Masked outside r in [1.09, 70] kpc.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import sobol_directions, make_radial_nodes  # noqa: E402
from particle_truth import load_particles, phi_direct  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "s11", "S1")
MODEL_TEXT = {"base": "base (seed-0)", "s11": "s11 (reshuffle)", "S1": "S1 (stratified)"}
COLOR = {"base": "blue", "s11": "orange", "S1": "red"}


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
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--asset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/pt-potential-2d"))
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-grid", type=int, default=240)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-sphere-nodes", type=int, default=24)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    xyz_p, m_p = load_particles(args.asset)
    print(f"particles: n={m_p.size}, M={m_p.sum():.3e} Msun")

    print('--- unit calibration ---')
    from particle_truth import G_KPC_KMS2_MSUN as GCAL
    one_xyz = np.array([[10.0, 0.0, 0.0]])
    one_m = np.array([1.0e5])
    v = float(phi_direct(one_xyz, one_m, np.array([[0.0, 0.0, 0.0]]))[0])
    print('single particle m=1e5 at 10 kpc, query origin:', v,
          'expected', -GCAL * 1e5 / 10.0)
    rng = np.random.default_rng(1)
    sel = rng.choice(m_p.size, size=200000, replace=False)
    q0 = np.array([[5.0, 0.0, 0.0]])
    d = np.linalg.norm(xyz_p[sel] - q0, axis=1)
    s_numpy = float(np.sum(m_p[sel] / d) * 4.30091e-6)
    print('numpy subsample 2e5 particles, sum m/d * G (partial, not full):', s_numpy)
    print('full phi_direct at (5,0,0):', float(phi_direct(xyz_p, m_p, q0)[0]))
    print('--- end calibration ---')

    edges = np.asarray([1.0, 1.0906, 70.0])
    r_anchor = 1.0906
    r_nodes = np.asarray(make_radial_nodes(r_anchor, args.r_outer,
                                           args.n_sphere_nodes - 1), dtype=float)
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)

    print("=== truth potential: sphere means then slice ===")
    t0 = time.time()
    q_sph = (dirs[None, :, :] * r_nodes[:, None, None]).reshape(-1, 3)
    phi_true_nodes = phi_direct(xyz_p, m_p, q_sph).reshape(r_nodes.size, -1)
    phi_true_70 = phi_direct(xyz_p, m_p, dirs * args.r_outer).mean()
    phi_true_nodes = phi_true_nodes - phi_true_70
    phi_true_prof = phi_true_nodes.mean(axis=1)
    print(f"sphere means on {r_nodes.size}x{args.n_dirs} dirs in {time.time()-t0:.0f}s; "
          f"Phi_true(1.09)={phi_true_prof[0]:.1f} (expect ~ -1.77e5); sigma_Omega at r=5/20 kpc = "
          + "/".join("%.2e" % v for v in
                     (phi_true_nodes[np.argmin(np.abs(r_nodes-r))].std()
                      / abs(phi_true_nodes[np.argmin(np.abs(r_nodes-r))].mean())
                      for r in (5.0, 20.0))))

    xs = np.linspace(-args.r_outer, args.r_outer, args.n_grid)
    xx, zz = np.meshgrid(xs, xs, indexing="xy")
    rr = np.hypot(xx, zz)
    q_slice = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])
    t0 = time.time()
    phi_true_grid = (phi_direct(xyz_p, m_p, q_slice).reshape(xx.shape)
                     - phi_true_70)
    print(f"slice {args.n_grid}^2 in {time.time()-t0:.0f}s")

    slices, consts, profs = {}, {}, {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        t0 = time.time()
        import jax.numpy as jnp
        val = jax.vmap(phi)
        q = q_slice / L_KPC
        grid = np.empty(q.shape[0])
        for i in range(0, q.shape[0], 65536):
            grid[i:i + 65536] = np.asarray(val(jnp.asarray(q[i:i + 65536])))
        grid = grid.reshape(xx.shape) * V_KMS ** 2
        nodes = np.empty(r_nodes.size)
        for j, r in enumerate(r_nodes):
            nodes[j] = float(np.mean(np.asarray(val(
                jnp.asarray(dirs * (r / L_KPC)))))) * V_KMS ** 2
        c = float(np.mean(nodes - phi_true_prof))
        slices[label] = grid - c
        consts[label] = c
        profs[label] = nodes - c
        print(f"[{label}] slice+profile in {time.time()-t0:.0f}s; c={c:.1f}; "
              f"sphere-mean RMS vs truth={float(np.sqrt(np.mean((nodes - c - phi_true_prof) ** 2))):.1f} (km/s)^2")

    valid = (rr >= r_anchor) & (rr <= args.r_outer)
    phi_true_map = np.where(valid, phi_true_grid, np.nan)
    rabs = np.nanpercentile(np.abs(np.stack(
        [np.where(valid, slices[l] - phi_true_grid, np.nan) for l in MODELS])), 98)
    vmin = min(np.nanmin(np.where(valid, s, np.nan)) for s in
               list(slices.values()) + [phi_true_map])
    vmax = max(np.nanmax(np.where(valid, s, np.nan)) for s in
               list(slices.values()) + [phi_true_map])
    print(f"shared Phi scale [{vmin:.1f}, {vmax:.1f}]; residual +-{rabs:.1f} (98th pct)")

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, 4, width=ofs.WIDE, ratio=0.58)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)
    ims = [axes[0, 0].pcolormesh(xs, xs, phi_true_map, cmap=cmap_seq,
                                 vmin=vmin, vmax=vmax, rasterized=True,
                                 shading="auto")]
    for j, l in enumerate(MODELS):
        m = np.where(valid, slices[l], np.nan)
        ims.append(axes[0, j + 1].pcolormesh(xs, xs, m, cmap=cmap_seq,
                                             vmin=vmin, vmax=vmax,
                                             rasterized=True, shading="auto"))
    axp = axes[1, 0]
    axp.plot(r_nodes, phi_true_prof, color="0.2", lw=1.6, label="truth (particles)")
    for l in MODELS:
        axp.plot(r_nodes, profs[l], color=ofs.PALETTE[COLOR[l]], lw=1.2)
    axp.set_xlabel("r [kpc]")
    axp.set_ylabel(r"$\langle\Phi\rangle_\Omega$ [$(\mathrm{km/s})^2$]")
    axp.legend(frameon=False, fontsize=6.5, loc="lower right")
    im_res = None
    for j, l in enumerate(MODELS):
        res = np.where(valid, slices[l] - phi_true_grid, np.nan)
        im_res = axes[1, j + 1].pcolormesh(xs, xs, res, cmap=cmap_div,
                                           vmin=-rabs, vmax=rabs,
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
        axes[0, j].annotate(name, xy=(0.02, 0.97), xycoords="axes fraction",
                            ha="left", va="top", fontsize=7, color="#333333",
                            backgroundcolor="white")
    ofs.panel_labels(list(axes.ravel()))
    cb1 = fig.colorbar(ims[0], ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
    cb1.set_label(r"$\Phi - c$ [$(\mathrm{km/s})^2$]")
    cb2 = fig.colorbar(im_res, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
    cb2.set_label(r"$\Phi_m - \Phi_{\rm true}$ [$(\mathrm{km/s})^2$]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "pt_potential_2d.npz",
             x_kpc=xs, r_grid=rr, valid=valid, r_nodes=r_nodes,
             phi_true_prof=phi_true_prof, phi_true_dirs=phi_true_nodes, phi_true_grid=phi_true_grid,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("phi", slices[l]), ("c", consts[l]),
                             ("prof", profs[l]))})
    for p in ofs.save(fig, str(args.output_dir / "pt-potential-2d")):
        print("saved", p)
    print("PT_POTENTIAL_2D_DONE")


if __name__ == "__main__":
    sys.exit(main())
