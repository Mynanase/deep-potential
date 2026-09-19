#!/usr/bin/env python
"""Angular structure of the model density: fluctuation profile, multipoles,
Mollweide maps.

The truth profile is spherical, so every l>=1 angular moment of the model
density (potential Laplacian) is a deviation from it. Coherent low-l power
(l=2, l=4) would look like halo shape; broadband power up to the fit limit
is network noise. (a) sigma_Omega/<rho> vs r; (b,c) multipole power
fractions at r=2 and 5 kpc; (d-f) Mollweide maps of rho/<rho>-1 at r=5 kpc.
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
COLOR = {"base": "blue", "s11": "orange", "S1": "red"}
LMAX = 10


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


def truth_density(truth, r_fine):
    r_c = np.asarray(truth["r_center"], dtype=float)
    dm = np.asarray(truth["M_shell_total"], dtype=float)
    r_e = np.asarray(truth["r_edges"], dtype=float)
    vol = 4.0 / 3.0 * np.pi * (r_e[1:] ** 3 - r_e[:-1] ** 3)
    rho_c = dm / vol
    return np.exp(np.interp(np.log(r_fine), np.log(r_c), np.log(rho_c)))


def sph_design(dirs):
    """Complex design matrix of Y_lm, columns ordered l=0..LMAX, m=-l..l."""
    from scipy.special import sph_harm
    z = dirs[:, 2]
    theta_pol = np.arccos(np.clip(z, -1.0, 1.0))
    phi_azi = np.arctan2(dirs[:, 1], dirs[:, 0])
    cols, meta = [], []
    for l in range(LMAX + 1):
        for m in range(-l, l + 1):
            cols.append(sph_harm(m, l, phi_azi, theta_pol))
            meta.append((l, m))
    return np.stack(cols, axis=1), meta


def multipole_power(f, A, meta):
    """Power per l (fraction of total l>=1 fluctuation power, percent)."""
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    p = np.abs(c) ** 2
    ls = np.array([l for l, _ in meta])
    per_l = np.array([p[ls == l].sum() for l in range(LMAX + 1)])
    tot = per_l[1:].sum()
    frac = 100.0 * per_l[1:] / tot if tot > 0 else np.full(LMAX, np.nan)
    return frac, per_l


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/s1-density-angular"))
    ap.add_argument("--r-outer", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    ap.add_argument("--n-nodes", type=int, default=48)
    ap.add_argument("--map-radius", type=float, default=5.0)
    ap.add_argument("--spec-radii", nargs="+", type=float, default=[2.0, 5.0])
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib as mpl
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")

    truth = load_truth(args.truth)
    edges = np.asarray(truth["r_edges"], dtype=float)
    r_anchor = float(edges[edges >= 1.0][0])
    r_nodes = np.asarray(make_radial_nodes(
        r_anchor, args.r_outer, args.n_nodes - 1,
        boundaries=sorted(set(args.spec_radii + [args.map_radius]))), dtype=float)
    rho_true = truth_density(truth, r_nodes)

    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    A, meta = sph_design(dirs)
    prof = {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        t0 = time.time()
        vals = np.empty((r_nodes.size, dirs.shape[0]))
        for j, r in enumerate(r_nodes):
            vals[j] = np.asarray(rho_from_phi(phi, dirs * (r / L_KPC), L_KPC, V_KMS))
        prof[label] = vals
        print(f"[{label}] {r_nodes.size} nodes x {dirs.shape[0]} dirs in "
              f"{time.time() - t0:.0f}s")
        for rr in args.spec_radii + [10.0]:
            j = int(np.argmin(np.abs(r_nodes - rr)))
            rho = vals[j]
            mu = float(rho.mean())
            f = rho / mu - 1.0
            frac, _ = multipole_power(f, A, meta)
            print(f"  r={r_nodes[j]:.2f}: mean={mu:.3e} (truth {rho_true[j]:.3e}, "
                  f"offset {100 * (mu / rho_true[j] - 1):+.1f}%), "
                  f"sigma/mean={float(rho.std() / mu) * 100:.1f}%, "
                  f"l=1 {frac[0]:.1f}% l=2 {frac[1]:.1f}% l=3 {frac[2]:.1f}% "
                  f"broadband(l>=5) {frac[4:].sum():.1f}%")

    ofs.use_style()
    fig, axes = ofs.figure_grid(2, 3, width=ofs.WIDE, ratio=0.62)
    axa, axb, axc = axes[0]
    for label in MODELS:
        mu = prof[label].mean(axis=1)
        sig = prof[label].std(axis=1)
        ok = mu > 0
        axa.plot(r_nodes[ok], sig[ok] / mu[ok] * 100,
                 color=ofs.PALETTE[COLOR[label]], lw=1.4)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(r_anchor, args.r_outer)
    axa.set_xlabel("r [kpc]")
    axa.set_ylabel(r"$\sigma_\Omega/\langle\rho\rangle$ [%]")

    for ax, rr in zip((axb, axc), args.spec_radii):
        j = int(np.argmin(np.abs(r_nodes - rr)))
        ls = np.arange(1, LMAX + 1)
        for label in MODELS:
            rho = prof[label][j]
            f = rho / rho.mean() - 1.0
            frac, _ = multipole_power(f, A, meta)
            ax.plot(ls, np.sqrt(frac / 100.0), "o-", ms=3,
                    color=ofs.PALETTE[COLOR[label]], lw=1.2)
        ax.set_xlabel("multipole l")
        ax.set_ylabel(r"$\sqrt{C_l/\sum C}$")
        ax.set_ylim(0, None)
        ax.annotate(f"r = {r_nodes[j]:.1f} kpc", xy=(0.97, 0.95),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=7.5, color="#333333")

    jm = int(np.argmin(np.abs(r_nodes - args.map_radius)))
    fmap = {}
    for label in MODELS:
        rho = prof[label][jm]
        fmap[label] = rho / rho.mean() - 1.0
    clim = float(np.percentile(np.abs(np.stack(list(fmap.values()))), 99))
    lon = np.arctan2(dirs[:, 1], dirs[:, 0])
    lat = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    for k, label in enumerate(MODELS):
        ax = axes[1, k]
        sc = ax.scatter(lon, lat, c=fmap[label], s=1.5, cmap=ofs.DIVERGING,
                        vmin=-clim, vmax=clim, rasterized=True)
        ax.set_xlabel("lon [rad]")
        ax.set_ylabel("lat [rad]")
        ax.annotate(MODEL_TEXT[label] + f"  r={r_nodes[jm]:.1f} kpc",
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 10),
                    textcoords="offset points", fontsize=7, color="#333333")
    cb = fig.colorbar(sc, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)
    cb.set_label(r"$\rho/\langle\rho\rangle_\Omega - 1$")
    handles = [axa.plot([], [], color=ofs.PALETTE[COLOR[l]], lw=1.6)[0]
               for l in MODELS]
    fig.legend(handles, [MODEL_TEXT[l] for l in MODELS],
               loc="outside lower center", ncol=3, frameon=False)
    ofs.panel_labels(list(axes.ravel()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "s1_density_angular.npz",
             r_nodes=r_nodes, rho_true_nodes=rho_true, lmax=LMAX,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("rho_dirs", prof[l]),)})
    for p in ofs.save(fig, str(args.output_dir / "s1-density-angular")):
        print("saved", p)
    print("DENSITY_ANGULAR_DONE")


if __name__ == "__main__":
    sys.exit(main())

