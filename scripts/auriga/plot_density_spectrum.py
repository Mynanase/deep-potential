#!/usr/bin/env python
"""Radial and angular oscillation spectra of model density vs particle truth.

Diagnostic for the negative-density / high-frequency-oscillation study. No
retraining; evaluates frozen Phi checkpoints and the particle-truth density
histogram on common probes:

  * Radial residual spectra: on 512 Sobol directions x 128 uniform radial
    nodes (1.09-70 kpc), rho = Laplacian(Phi) * rho_scale. The angular mean
    per radius is removed; the residual is windowed (Hann) and Fourier
    transformed along r. Reports the high-frequency energy fraction
    f_hi(lambda_c) = sum_{lambda < lambda_c} P / sum_{lambda > 0} P at
    wavelength cutoffs (default 10 / 5 / 2.5 kpc) and the residual variance.
  * Angular structure: on shells (default r = 20 / 45 / 60 kpc) with 4096
    Fibonacci directions, the angular autocorrelation C(theta) of the
    density residual from random pairs; the patch correlation angle
    theta_half (C = 0.5 crossing) and the transverse patch scale r*theta.
  * Pair calibration for the osc-pair penalty: mean squared rho first and
    second differences across point pairs separated by delta in {2,4,6} kpc on the
    radius-balanced prior grid (r = R*u, n default 8192, R = 70 kpc), with
    the matching arcsinh negative-density penalty for scale.

Truth convention: rho3d of the committed particle-truth grid product
(96^3 cells, 1.5625 kpc, cell-average total-matter histogram, star frame),
trilinearly interpolated at the probe points. Its spectrum is valid up to
the cell Nyquist wavelength ~3.1 kpc; f_hi at 2.5 kpc for truth is reported
but flagged as cell-limited.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from plot_particle_truth_2d import load_phi_f32  # noqa: E402
from validate_enclosed_mass import G_KPC_KMS2_MSUN, rho_from_phi  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
RHO_SCALE = V_KMS ** 2 / (4.0 * np.pi * G_KPC_KMS2_MSUN * L_KPC ** 2)
COLOR = {"base": "blue", "S1": "red", "gridprior": "green",
         "innerA": "purple", "lambda10": "orange", "truth": "k"}


def sobol_dirs(n, seed):
    from scipy.stats import qmc
    sob = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sob.random(n)
    mu = 2.0 * uv[:, 0] - 1.0
    az = 2.0 * np.pi * uv[:, 1]
    s = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
    return np.column_stack([s * np.cos(az), s * np.sin(az), mu])


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    z = 1.0 - 2.0 * i / n
    s = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


def radial_spectrum(delta, dr):
    """Mean periodogram of ray residuals (rows = rays), Hann-windowed."""
    n = delta.shape[1]
    win = np.hanning(n)
    x = delta * win[None, :]
    p = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freq = np.fft.rfftfreq(n, d=dr)
    keep = freq > 0
    return freq[keep], p[:, keep].mean(axis=0)


def f_hi(freq, power, cutoff_kpc):
    lam = 1.0 / np.maximum(freq, 1e-30)
    tot = float(power.sum())
    if tot <= 0:
        return 0.0
    return float(power[lam < cutoff_kpc].sum() / tot)


def theta_half_crossing(theta_bins, corr):
    """First angular separation where the autocorrelation drops below 0.5."""
    for i in range(len(theta_bins) - 1):
        if corr[i] >= 0.5 > corr[i + 1]:
            t0, t1 = theta_bins[i], theta_bins[i + 1]
            c0, c1 = corr[i], corr[i + 1]
            return float(t0 + (c0 - 0.5) * (t1 - t0) / max(c0 - c1, 1e-12))
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--model", action="append", required=True,
                    help="label=run_dir, repeatable")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/density-spectrum"))
    ap.add_argument("--r-min", type=float, default=1.09)
    ap.add_argument("--r-max", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=512)
    ap.add_argument("--n-radial", type=int, default=128)
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[10.0, 5.0, 2.5])
    ap.add_argument("--shells", type=float, nargs="+",
                    default=[20.0, 45.0, 60.0])
    ap.add_argument("--n-fib", type=int, default=4096)
    ap.add_argument("--n-pairs", type=int, default=2_000_000)
    ap.add_argument("--pair-grid-n", type=int, default=8192)
    ap.add_argument("--pair-radius-kpc", type=float, default=70.0)
    ap.add_argument("--pair-deltas", type=float, nargs="+",
                    default=[2.0, 4.0, 6.0])
    ap.add_argument("--eval-batch", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"CONFIG n_dirs={args.n_dirs} n_radial={args.n_radial} "
          f"r=[{args.r_min},{args.r_max}] kpc cutoffs={args.cutoffs} kpc "
          f"shells={args.shells} kpc seed={args.seed}")
    t0_all = time.time()

    # ---- truth histogram field -------------------------------------------
    with h5py.File(args.grids, "r") as f:
        rho3d = np.asarray(f["density/rho3d"][:], dtype=float)
        edges3d = np.asarray(f["density/rho3d_edges_kpc"][:], dtype=float)
    cell = float(np.abs(np.diff(edges3d[0])).max())
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges3d]
    print(f"truth rho3d {rho3d.shape} cell {cell:.4f} kpc "
          f"(Nyquist wavelength {2*cell:.2f} kpc)")
    from scipy.interpolate import RegularGridInterpolator
    truth_interp = RegularGridInterpolator(
        centers, rho3d, method="linear", bounds_error=False, fill_value=None)

    def eval_batched(phi, q_kpc):
        out = np.empty(q_kpc.shape[0])
        for i in range(0, q_kpc.shape[0], args.eval_batch):
            out[i:i + args.eval_batch] = np.asarray(rho_from_phi(
                phi, q_kpc[i:i + args.eval_batch] / L_KPC, L_KPC, V_KMS))
        return out

    # ---- common radial rays ----------------------------------------------
    r = np.linspace(args.r_min, args.r_max, args.n_radial)
    dr = float(r[1] - r[0])
    dirs = sobol_dirs(args.n_dirs, args.seed)
    rays = (r[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
    truth_rays = truth_interp(rays).reshape(args.n_dirs, args.n_radial)

    labels = [spec.split("=", 1)[0] for spec in args.model]
    model_dirs = dict(s.split("=", 1) for s in args.model)
    series = {"truth": truth_rays}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        phi = load_phi_f32(run_dir)
        series[label] = eval_batched(phi, rays).reshape(
            args.n_dirs, args.n_radial)

    print("=== radial residual spectra (angular mean removed per radius) ===")
    print(f"{'label':>10}"
          + "".join(f"  f_hi<{c:>4.1f}kpc" for c in args.cutoffs)
          + "  resid.var  peak-lam")
    spectra, resid_var, fhi_rows = {}, {}, {}
    freq_ref = None
    for label, mat in series.items():
        delta = mat - mat.mean(axis=0, keepdims=True)
        freq, power = radial_spectrum(delta, dr)
        freq_ref = freq
        spectra[label] = power
        resid_var[label] = float(np.mean(delta ** 2))
        fhi_rows[label] = {f"{c:g}": f_hi(freq, power, c)
                           for c in args.cutoffs}
        peak = float(1.0 / freq[int(np.argmax(power))])
        row = "".join(f"  {fhi_rows[label][f'{c:g}']:9.1%}"
                      for c in args.cutoffs)
        print(f"{label:>10}{row}  {resid_var[label]:9.3e}  {peak:6.1f}kpc")
    cell_limited = min(args.cutoffs) < 2 * cell
    if cell_limited:
        print(f"note: truth f_hi < {min(args.cutoffs):g} kpc is cell-limited "
              f"(histogram Nyquist {2*cell:.1f} kpc); read cutoffs >= "
              f"{2*cell:.1f} kpc as the truth reference")

    # ---- angular structure on shells --------------------------------------
    fib = fib_dirs(args.n_fib)
    rng = np.random.default_rng(args.seed)
    ip = rng.integers(0, args.n_fib, args.n_pairs)
    jp = rng.integers(0, args.n_fib, args.n_pairs)
    ang = np.arccos(np.clip((fib[ip] * fib[jp]).sum(axis=1), -1.0, 1.0))
    n_bins = 32
    theta_bins = np.linspace(0.0, np.pi, n_bins + 1)
    which = np.clip(np.digitize(ang, theta_bins) - 1, 0, n_bins - 1)
    theta_c = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    ang_rows = {}
    print("=== angular patch structure (C(theta) half-crossing) ===")
    print(f"{'label':>10}"
          + "".join(f"  r={s:>4.1f}kpc theta_half  patch" for s in args.shells))
    for label in ["truth"] + labels:
        ang_rows[label] = {}
        row = ""
        for s in args.shells:
            pts = s * fib
            if label == "truth":
                rho_s = truth_interp(pts)
            else:
                phi = load_phi_f32(model_dirs[label])
                rho_s = eval_batched(phi, pts)
            delta = rho_s - rho_s.mean()
            var = float(np.mean(delta ** 2))
            prod = delta[ip] * delta[jp]
            corr = np.array([prod[which == b].mean() / max(var, 1e-30)
                             for b in range(n_bins)])
            th = theta_half_crossing(theta_c, corr)
            ang_rows[label][f"{s:g}"] = dict(
                theta_half_rad=th, patch_kpc=float(s * th),
                corr=corr.tolist())
            row += f"  {np.degrees(th):8.1f}deg {s*th:6.1f}kpc"
        print(f"{label:>10}{row}")

    # ---- osc-pair calibration on the radius-balanced grid ----------------
    n_g = args.pair_grid_n
    u_r = rng.random(n_g)
    r_g = args.pair_radius_kpc * u_r  # radius-balanced (linear CDF)
    g = rng.normal(size=(n_g, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    q_a = r_g[:, None] * g
    calib = {}
    print("=== osc-pair calibration (radius-balanced grid, "
          f"n={n_g}, R={args.pair_radius_kpc:g} kpc) ===")
    for label in labels:
        phi = load_phi_f32(model_dirs[label])
        rho_a = eval_batched(phi, q_a)
        pen = float(np.mean(np.arcsinh(np.maximum(
            -(rho_a / RHO_SCALE), 0.0))))
        calib[label] = dict(prior_neg=pen)
        for d0 in args.pair_deltas:
            u = rng.normal(size=(n_g, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            rho_b = eval_batched(phi, q_a + d0 * u)
            msd = float(np.mean((rho_b - rho_a) ** 2))
            calib[label][f"msd_{d0:g}kpc"] = msd
        for d0 in args.pair_deltas:
            u = rng.normal(size=(n_g, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            rho_p = eval_batched(phi, q_a + d0 * u)
            rho_m = eval_batched(phi, q_a - d0 * u)
            calib[label][f"msd2_{d0:g}kpc"] = float(
                np.mean((rho_p - 2.0 * rho_a + rho_m) ** 2))
        row = "".join(f"  d={d:g}: {calib[label][f'msd_{d:g}kpc']:8.2e}"
                      f"/{calib[label][f'msd2_{d:g}kpc']:8.2e}"
                      for d in args.pair_deltas)
        print(f"{label:>10}  prior_neg={pen:.4f}{row}")
    # Truth reference on the same probes (cell-average histogram field).
    rho_a_t = truth_interp(q_a)
    calib["truth"] = {}
    for d0 in args.pair_deltas:
        u = rng.normal(size=(n_g, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rho_b = truth_interp(q_a + d0 * u)
        rho_p = truth_interp(q_a + d0 * u)
        rho_m = truth_interp(q_a - d0 * u)
        calib["truth"][f"msd_{d0:g}kpc"] = float(np.mean((rho_b - rho_a_t) ** 2))
        calib["truth"][f"msd2_{d0:g}kpc"] = float(
            np.mean((rho_p - 2.0 * rho_a_t + rho_m) ** 2))
    print(f"{'truth':>10}  (histogram field reference)"
          + "".join(f"  d={d:g}: {calib['truth'][f'msd_{d:g}kpc']:8.2e}"
                    f"/{calib['truth'][f'msd2_{d:g}kpc']:8.2e}"
                    for d in args.pair_deltas))
    print("columns are msd1/msd2 = mean squared first/second difference")
    if "innerA" in calib and calib["innerA"].get("msd2_4kpc", 0) > 0:
        print("suggested osc_weight eta, second-difference form "
              "(eta*msd2_lap(innerA,d=4kpc)=1.0): "
              f"{RHO_SCALE ** 2 / calib['innerA']['msd2_4kpc']:.3e}")

    # ---- figure ------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ofs.use_style()
    fig, axes = ofs.figure_grid(1, 2, width=ofs.TEXT)
    for label in ["truth"] + labels:
        axes[0].loglog(freq_ref, spectra[label],
                       color=ofs.PALETTE[COLOR[label]],
                       lw=1.3 if label == "truth" else 0.9,
                       ls="--" if label == "truth" else "-",
                       label="truth (histogram)" if label == "truth" else label)
    axes[0].set_xlabel(r"Wavenumber $k$ [kpc$^{-1}$]")
    axes[0].set_ylabel(r"Mean radial power of $\delta\rho$")
    for c in args.cutoffs:
        axes[0].axvline(2 * np.pi / c, color="0.5", lw=0.5, alpha=0.5)
    axes[0].legend(fontsize=6, frameon=False)
    for label in ["truth"] + labels:
        corr = np.asarray(ang_rows[label][f"{args.shells[0]:g}"]["corr"])
        axes[1].plot(np.degrees(theta_c), corr,
                     color=ofs.PALETTE[COLOR[label]],
                     ls="--" if label == "truth" else "-",
                     lw=1.3 if label == "truth" else 0.9)
    axes[1].axhline(0.5, color="0.5", lw=0.5, alpha=0.5)
    axes[1].set_xlabel(r"Angular separation $\theta$ [deg]")
    axes[1].set_ylabel(r"$C(\theta)$ of $\delta\rho$")
    axes[1].set_ylim(-0.2, 1.0)
    ofs.panel_labels(axes)
    base = args.output_dir / "density-spectrum"
    ofs.save(fig, str(base))

    out = dict(
        config=dict(n_dirs=args.n_dirs, n_radial=args.n_radial, dr_kpc=dr,
                    r_min=args.r_min, r_max=args.r_max,
                    cutoffs_kpc=args.cutoffs, shells_kpc=args.shells,
                    seed=args.seed, truth_cell_kpc=cell),
        f_hi=fhi_rows, resid_var=resid_var, angular=ang_rows, calib=calib)
    json.dump(out, open(args.output_dir / "density_spectrum.json", "w"),
              indent=2)
    np.savez_compressed(
        args.output_dir / "density_spectrum.npz",
        freq=freq_ref,
        spectra=np.array([spectra[l] for l in ["truth"] + labels]),
        labels=np.array(["truth"] + labels),
        theta_deg=np.degrees(theta_c))
    print(f"TOTAL WALL {time.time()-t0_all:.1f}s")
    print("DENSITY_SPECTRUM_DONE")
    print("=== OUTPUTS ===")
    print(f"{base}.pdf")
    print(f"{base}.svg")
    print(f"{args.output_dir}/density_spectrum.json")
    print(f"{args.output_dir}/density_spectrum.npz")


if __name__ == "__main__":
    sys.exit(main())
