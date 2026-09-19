#!/usr/bin/env python
"""Meridional 2D potential maps and truth residuals for three Phis.

Row 1: aligned model potential Phi(R,z) on the y=0 slice (x-z plane),
shared viridis scale. Row 2: residual against the spherical truth potential
obtained by integrating the 60-shell total-mass profile
(Phi_true(r) = int_r^rout G M(<s)/s^2 ds, zero point at the outer truth
edge), RdBu_r symmetric limits. Each model's additive constant is fixed by
least squares on the sphere-averaged Phi over the valid radial range; the
zero convention is stated in the caption, not hidden in the plot.
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
    G_KPC_KMS2_MSUN, load_truth, sobol_directions, make_radial_nodes)
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "s11", "S1")
MODEL_TEXT = {"base": "base (seed-0)", "s11": "s11 (reshuffle)", "S1": "S1 (stratified)"}


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


def phi_grid(phi, xs, zs):
    """Phi on the (x, z) grid, physical [(km/s)^2]; batched vmap."""
    import jax
    import jax.numpy as jnp
    xx, zz = np.meshgrid(xs, zs, indexing="xy")  # rows = z, cols = x
    q = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]) / L_KPC
    val = jax.vmap(phi)
    out = np.empty(q.shape[0], dtype=np.float64)
    for i in range(0, q.shape[0], 65536):
        out[i:i + 65536] = np.asarray(val(jnp.asarray(q[i:i + 65536])))
    return out.reshape(xx.shape) * V_KMS ** 2, xx * 0 + np.hypot(xx, zz)


def phi_sphere_mean(phi, r_nodes, dirs):
    import jax
    import jax.numpy as jnp
    val = jax.vmap(phi)
    out = np.empty(r_nodes.size)
    for j, r in enumerate(r_nodes):
        q = dirs * (r / L_KPC)
        out[j] = float(np.mean(np.asarray(val(jnp.asarray(q))))) * V_KMS ** 2
    return out


def truth_potential(truth, r_fine):
    """Phi_true(r) [(km/s)^2], zero at the outer truth edge.

    M_cum_total[i] binds to r_edges[i+1] (validate_enclosed_mass convention).
    log-log interpolation of M(<s), then cumulative trapezoid of G M/s^2.
    """
    edges = np.asarray(truth["r_edges"], dtype=float)
    m_cum = np.asarray(truth["M_cum_total"], dtype=float)
    r_bind = edges[1:]
    if r_bind.size != m_cum.size:
        raise ValueError("truth convention mismatch: M_cum does not bind to r_edges[1:]")

    s = np.geomspace(edges[1], edges[-1], 4000)
    m_of_s = np.exp(np.interp(np.log(s), np.log(r_bind), np.log(m_cum)))
    g_int = G_KPC_KMS2_MSUN * m_of_s / s ** 2
    phi_outer = 0.0
    cum = np.concatenate([[phi_outer],
                          np.cumsum((g_int[:-1] + g_int[1:]) * np.diff(s) / 2.0)])[::-1]
    phi_true_s = phi_outer + cum
    return np.interp(r_fine, s, phi_true_s), float(edges[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/s1-potential-2d"))
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

    truth = load_truth(args.truth)
    edges = np.asarray(truth["r_edges"], dtype=float)
    r_anchor = float(edges[edges >= 1.0][0])
    if edges[-1] < args.r_outer - 1e-6:
        raise ValueError(f"truth outer edge {edges[-1]} < r-outer {args.r_outer}")
    r_nodes = np.asarray(make_radial_nodes(r_anchor, args.r_outer,
                                           args.n_sphere_nodes - 1), dtype=float)
    phi_true_nodes, r_out_edge = truth_potential(truth, r_nodes)
    print(f"truth: anchor {r_anchor:.4f} kpc, outer edge {r_out_edge:.2f} kpc, "
          f"Phi_true({r_anchor:.2f}) = {phi_true_nodes[0]:.1f} (km/s)^2")

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    xs = np.linspace(-args.r_outer, args.r_outer, args.n_grid)
    zs = xs.copy()
    XX, RR = None, None
    slices, residuals, consts = {}, {}, {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        t0 = time.time()
        grid, rr = phi_grid(phi, xs, zs)
        if RR is None:
            XX = np.meshgrid(xs, zs, indexing="xy")[0]
            RR = rr
        mean_nodes = phi_sphere_mean(phi, r_nodes, dirs)
        c = float(np.mean(mean_nodes - phi_true_nodes))
        slices[label] = grid - c
        residuals[label] = grid - c - np.interp(rr, r_nodes, phi_true_nodes)
        consts[label] = c
        rms = float(np.sqrt(np.mean((mean_nodes - c - phi_true_nodes) ** 2)))
        print(f"[{label}] grid {args.n_grid}^2 + {args.n_sphere_nodes}x{args.n_dirs} "
              f"sphere nodes in {time.time() - t0:.0f}s; const c={c:.1f}, "
              f"sphere-mean residual RMS={rms:.1f} (km/s)^2")

    valid = (RR >= r_anchor) & (RR <= args.r_outer)
    vlim = max(float(np.nanmax(np.where(valid, s, np.nan))) for s in slices.values())
    vmin = min(float(np.nanmin(np.where(valid, s, np.nan))) for s in slices.values())
    rabs = np.nanpercentile(np.abs(np.where(valid, np.stack(list(residuals.values())), np.nan)),
                            98)
    print(f"shared scales: Phi in [{vmin:.1f}, {vlim:.1f}]; "
          f"residual +/-{rabs:.1f} (98th pct of |res|)")

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, len(args.model), width=ofs.WIDE, ratio=0.62)
    cmap_seq = mpl.colormaps[ofs.SEQUENTIAL].with_extremes(bad=ofs.MUTED)
    cmap_div = mpl.colormaps[ofs.DIVERGING].with_extremes(bad=ofs.MUTED)
    ims = [[], []]
    for col, label in enumerate(MODELS):
        m = np.where(valid, slices[label], np.nan)
        im = axes[0, col].pcolormesh(xs, zs, m, cmap=cmap_seq, vmin=vmin,
                                     vmax=vlim, rasterized=True,
                                     shading="auto")
        ims[0].append(im)
        rmap = np.where(valid, residuals[label], np.nan)
        im2 = axes[1, col].pcolormesh(xs, zs, rmap, cmap=cmap_div, vmin=-rabs,
                                      vmax=rabs, rasterized=True, shading="auto")
        ims[1].append(im2)
    for row in (0, 1):
        for col in range(len(MODELS)):
            ax = axes[row, col]
            ax.set_aspect("equal")
            ax.set_xlabel("x [kpc]" if row == 1 else "")
            if col == 0:
                ax.set_ylabel("z [kpc]")
    for col, label in enumerate(MODELS):
        axes[0, col].annotate(MODEL_TEXT[label], xy=(0, 1),
                              xycoords="axes fraction", xytext=(0, 14),
                              textcoords="offset points", fontsize=7.5,
                              color="#333333")
    ofs.panel_labels([axes[0, 0], axes[0, 1], axes[0, 2],
                      axes[1, 0], axes[1, 1], axes[1, 2]])
    cb1 = fig.colorbar(ims[0][0], ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    cb1.set_label(r"$\Phi - c_m$ [$(\mathrm{km/s})^2$]")
    cb2 = fig.colorbar(ims[1][0], ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)
    cb2.set_label(r"$\Phi - c_m - \Phi_{\rm true}$ [$(\mathrm{km/s})^2$]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "s1_potential_2d.npz",
             x_kpc=xs, z_kpc=zs, r_grid=RR, phi_true_nodes=phi_true_nodes,
             r_nodes=r_nodes, valid=valid,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("phi", slices[l]), ("res", residuals[l]),
                             ("c", consts[l]))})
    for p in ofs.save(fig, str(args.output_dir / "s1-potential-2d")):
        print("saved", p)
    print("POTENTIAL_2D_DONE")


if __name__ == "__main__":
    sys.exit(main())
