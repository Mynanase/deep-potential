#!/usr/bin/env python
"""Resolution convergence of the particle-truth density spectrum.

Pre-registered design (node particle-truth-spectrum-resolution-convergence-6):
the phase-1 density-spectrum adjudication reads high-frequency fractions off
the 96^3 particle-truth histogram, whose cell Nyquist wavelength
(2 * 1.5625 = 3.125 kpc) already flagged the old 2.5 kpc cutoff as
cell-limited.  This study quantifies where that spectrum is actually
resolved: identical particle set (sha256-verified against the committed
truth-grid product lineage), identical [-75, 75]^3 kpc NGP mass histogram at
N^3 in {64, 96, 128, 192}, one spectrum pipeline identical to the phase-1
diagnostic (512 scrambled-Sobol directions seed 20260921, 128 radial nodes
1.09-70 kpc, trilinear interpolation, angular mean removed per radius, Hann
window, mean rfft periodogram).

Convergence criterion (fixed in advance): a log-lambda bin is converged when
the median over its frequencies of |P_N2 - P_N1| / P_N2 < rel_tol (default
10%) for the adjacent gate pairs (96, 128) and (128, 192); the coarse pair
(64, 96) is reported but never gates.  lambda_reliable is the smallest bin
centre whose bin and every larger bin are converged, and which stays at or
beyond the Nyquist safety margin nyquist_factor * lambda_Nyq(finest grid)
(default 2 x 1.5625 = 3.125 kpc; the 1.5x variant is recorded as
sensitivity).

Shot-noise check: a seeded random 50/50 particle split builds A/B grids at
96^3 and 192^3 and runs the identical pipeline on the difference field
rho_A - rho_B.  A + B and A - B carry the same noise variance while the
halo signal cancels in the difference, so P_AB(k) / P_full(k) estimates the
noise fraction of the full-field power; values > 0.5 mark noise-dominated
scales.

Compute layer only (numpy/scipy/h5py, CPU): persists npz + json for the plot
script, never reads models.  Lineage contract: the asset sha256 must equal
the truth-grid product's recorded source_sha256 - a mismatch is an error,
never a silent different-particle run.
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

from truth_products import load_particles, sha256_file  # noqa: E402


def sobol_dirs(n, seed):
    """Scrambled-Sobol unit directions (identical generator to the phase-1
    density-spectrum diagnostic, so the ray set matches run e4d4db5d)."""
    from scipy.stats import qmc
    sob = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sob.random(n)
    mu = 2.0 * uv[:, 0] - 1.0
    az = 2.0 * np.pi * uv[:, 1]
    s = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
    return np.column_stack([s * np.cos(az), s * np.sin(az), mu])


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


def build_rho3d(xyz, m, n, extent):
    """NGP mass histogram of the production truth product: rho = H / cell^3.

    Identical estimator to truth_products.cmd_build_grids (float32 cast
    included); resolution is the only variable.
    """
    H, edges = np.histogramdd(xyz, bins=n, range=[(-extent, extent)] * 3, weights=m)
    cell = 2.0 * extent / n
    rho = (H / cell ** 3).astype(np.float32)
    closure = float(np.asarray(H, dtype=np.float64).sum() / m.sum())
    fill = float(np.mean(H > 0))
    return rho, edges, cell, closure, fill


def periodogram_of_grid(rho, edges, rays, n_dirs, n_radial, dr):
    """Identical pipeline for every field: trilinear interpolation on the
    common rays, angular mean removed per radius, Hann-windowed periodogram."""
    from scipy.interpolate import RegularGridInterpolator
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    interp = RegularGridInterpolator(centers, rho.astype(np.float64),
                                     method="linear", bounds_error=False,
                                     fill_value=None)
    mat = interp(rays).reshape(n_dirs, n_radial)
    delta = mat - mat.mean(axis=0, keepdims=True)
    freq, power = radial_spectrum(delta, dr)
    return float(np.mean(delta ** 2)), freq, power


def bin_median(lam, values, edges):
    """Median of per-frequency values in each log-lambda bin (nan if empty)."""
    which = np.digitize(lam, edges) - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = ok & (which == b)
        if sel.any():
            out[b] = float(np.median(values[sel]))
    return out


def reliable_bin_center(centers, converged, guard_kpc):
    """Pre-registered rule: smallest bin centre whose bin and every larger
    bin are converged, pushed out to the Nyquist guard if needed.
    Returns (lambda_reliable, lambda_empirical), nan if no converged tail."""
    for i in range(len(centers)):
        if bool(np.all(converged[i:])):
            empirical = float(centers[i])
            return max(empirical, float(guard_kpc)), empirical
    return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", type=Path, required=True,
                    help="star-frame particle h5 (must match truth-grid lineage)")
    ap.add_argument("--grids", type=Path, required=True,
                    help="committed particle-truth grid product (lineage source)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("runs/orx/resolution-convergence"))
    ap.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128, 192])
    ap.add_argument("--ab-resolutions", type=int, nargs="+", default=[96, 192])
    ap.add_argument("--extent", type=float, default=75.0,
                    help="histogram half-extent [kpc] (production value)")
    ap.add_argument("--r-min", type=float, default=1.09)
    ap.add_argument("--r-max", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=512)
    ap.add_argument("--n-radial", type=int, default=128)
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[10.0, 5.0, 4.0, 3.0, 2.5])
    ap.add_argument("--bin-edges", type=float, nargs="+",
                    default=[1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 25, 40, 80])
    ap.add_argument("--rel-tol", type=float, default=0.10)
    ap.add_argument("--nyquist-factor", type=float, default=2.0)
    ap.add_argument("--noise-thresh", type=float, default=0.5)
    ap.add_argument("--sobol-seed", type=int, default=20260921)
    ap.add_argument("--split-seed", type=int, default=20260924)
    args = ap.parse_args()
    t0 = time.time()
    print(f"CONFIG resolutions={args.resolutions} ab_resolutions={args.ab_resolutions} extent={args.extent:g} kpc")
    print(f"CONFIG rays n_dirs={args.n_dirs} n_radial={args.n_radial} r=[{args.r_min:g},{args.r_max:g}] kpc sobol_seed={args.sobol_seed}")
    print(f"CONFIG criterion rel_tol={args.rel_tol:.0%} bin-median |P_fine-P_coarse|/P_fine, gate pairs exclude the coarsest; guard {args.nyquist_factor:g}x lambda_Nyq(finest)")
    print(f"CONFIG lambda bins {list(args.bin_edges)}; split seed {args.split_seed}")

    # ---- lineage: same particles as the frozen production truth ----------
    recorded = None
    with h5py.File(args.grids, "r") as f:
        attr_sets = [dict(f.attrs)]
        f.visititems(lambda name, obj: attr_sets.append(dict(obj.attrs))
                     if isinstance(obj, h5py.Group) else None)
        for attrs in attr_sets:
            if "source_sha256" in attrs:
                recorded = attrs["source_sha256"]
    if recorded is None:
        raise SystemExit("lineage error: grids product records no source_sha256")
    if isinstance(recorded, bytes):
        recorded = recorded.decode()
    asset_sha = sha256_file(args.asset)
    print(f"lineage: asset sha256 {asset_sha[:20]}; grids recorded {str(recorded)[:20]}")
    if asset_sha != recorded:
        raise SystemExit("lineage error: asset sha256 != grids source_sha256; refusing a different particle set")

    xyz, m = load_particles(args.asset)
    print(f"particles: n={m.size} M={m.sum():.6e} Msun (all types, r<=75 kpc, star frame)")

    # ---- common probes (identical to the phase-1 diagnostic) -------------
    r = np.linspace(args.r_min, args.r_max, args.n_radial)
    dr = float(r[1] - r[0])
    dirs = sobol_dirs(args.n_dirs, args.sobol_seed)
    rays = (r[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
    print(f"probes: {args.n_dirs} rays x {args.n_radial} nodes, dr={dr:.4f} kpc, ray Nyquist {2*dr:.3f} kpc")

    print("=== full-field grids ===")
    spectra, grid_info = {}, {}
    freq_ref = None
    for n in args.resolutions:
        t0g = time.time()
        rho, edges, cell, closure, fill = build_rho3d(xyz, m, n, args.extent)
        resid_var, freq, power = periodogram_of_grid(rho, edges, rays, args.n_dirs, args.n_radial, dr)
        freq_ref = freq
        spectra[n] = power
        grid_info[n] = dict(cell_kpc=cell, lambda_nyq_kpc=2 * cell,
                            guard_kpc=args.nyquist_factor * 2 * cell,
                            guard15_kpc=1.5 * 2 * cell, filled_frac=fill,
                            mass_closure=closure, resid_var=resid_var)
        print(f"N={n:3d}^3 cell {cell:.4f} kpc lambda_Nyq {2*cell:.3f} guard2 {2*2*cell:.3f} guard1.5 {1.5*2*cell:.3f} kpc; "
              f"filled {fill*100:5.1f}%; closure {closure:+.2e}; resid.var {resid_var:.3e}; {time.time()-t0g:.1f}s")
    lam = 1.0 / freq_ref

    print("=== f_hi TABLE (cumulative high-frequency fraction) ===")
    print(f"{'grid':>8}" + "".join(f"  f_hi<{c:>4g}" for c in args.cutoffs))
    fhi_rows = {}
    for n in args.resolutions:
        fhi_rows[n] = {f"{c:g}": f_hi(freq_ref, spectra[n], c) for c in args.cutoffs}
        print(f"{str(n) + '^3':>8}" + "".join(f"  {fhi_rows[n][f'{c:g}']:7.1%}" for c in args.cutoffs))

    # ---- convergence: adjacent pairs, bin-median relative difference -----
    bin_edges = np.asarray(args.bin_edges, dtype=float)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    adjacent = list(zip(args.resolutions[:-1], args.resolutions[1:]))
    gate_pairs = adjacent[1:]
    print(f"=== CONVERGENCE (median |P_N2-P_N1|/P_N2 per log-lambda bin; tol {args.rel_tol:.0%}) ===")
    header = "".join(f"{c:8.2f}" for c in bin_centers)
    print(f"{'pair':>12}  bin centres [kpc] {header}")
    rel_binned, conv_binned = {}, {}
    for n1, n2 in adjacent:
        rel = np.abs(spectra[n2] - spectra[n1]) / spectra[n2]
        rel_binned[(n1, n2)] = bin_median(lam, rel, bin_edges)
        conv_binned[(n1, n2)] = rel_binned[(n1, n2)] < args.rel_tol
        row = "".join("       -" if np.isnan(v) else f" {v:7.1%}" for v in rel_binned[(n1, n2)])
        gate = " (gate)" if (n1, n2) in gate_pairs else ""
        print(f"{str(n1) + '-' + str(n2):>12}{gate:>8}{row}")
    gate_conv = np.all([conv_binned[p] for p in gate_pairs], axis=0)
    row = "".join("     ok " if v else "   FAIL " for v in gate_conv)
    print(f"{'gate':>20}{row}")

    guard = args.nyquist_factor * grid_info[max(args.resolutions)]["lambda_nyq_kpc"]
    guard15 = 1.5 * grid_info[max(args.resolutions)]["lambda_nyq_kpc"]
    lam_reliable, lam_empirical = reliable_bin_center(bin_centers, gate_conv, guard)
    lam_reliable15, _ = reliable_bin_center(bin_centers, gate_conv, guard15)
    basis = "guard-limited" if lam_empirical <= guard else "convergence-limited"
    if np.isnan(lam_reliable):
        print("WARNING: no converged tail under the pre-registered criterion")
    print("=== f_hi CONVERGENCE (|f_hi(N2)-f_hi(N1)|/f_hi(N2)) ===")
    fhi_conv = {}
    for n1, n2 in adjacent:
        fhi_conv[f"{n1}-{n2}"] = {f"{c:g}": abs(fhi_rows[n2][f"{c:g}"] - fhi_rows[n1][f"{c:g}"]) / fhi_rows[n2][f"{c:g}"]
                                  for c in args.cutoffs}
        print(f"{str(n1) + '-' + str(n2):>12}" + "".join(f"  {fhi_conv[str(n1) + '-' + str(n2)][f'{c:g}']:7.1%}" for c in args.cutoffs))

    # ---- shot-noise check: seeded A/B split, identical pipeline ----------
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(m.size)
    half = m.size // 2
    iA, iB = perm[:half], perm[half:]
    print(f"=== SHOT-NOISE A/B SPLIT (seed {args.split_seed}, n_A={iA.size}, n_B={iB.size}) ===")
    noise_binned, ab_agree_binned = {}, {}
    for n in args.ab_resolutions:
        rhoA, edgesA, _, _, _ = build_rho3d(xyz[iA], m[iA], n, args.extent)
        rhoB, edgesB, _, _, _ = build_rho3d(xyz[iB], m[iB], n, args.extent)
        _, _, P_A = periodogram_of_grid(rhoA, edgesA, rays, args.n_dirs, args.n_radial, dr)
        _, _, P_B = periodogram_of_grid(rhoB, edgesB, rays, args.n_dirs, args.n_radial, dr)
        from scipy.interpolate import RegularGridInterpolator
        centersA = [0.5 * (e[:-1] + e[1:]) for e in edgesA]
        centersB = [0.5 * (e[:-1] + e[1:]) for e in edgesB]
        matA = RegularGridInterpolator(centersA, rhoA.astype(np.float64), method="linear",
                                       bounds_error=False, fill_value=None)(rays)
        matB = RegularGridInterpolator(centersB, rhoB.astype(np.float64), method="linear",
                                       bounds_error=False, fill_value=None)(rays)
        deltaD = (matA - matB).reshape(args.n_dirs, args.n_radial)
        deltaD = deltaD - deltaD.mean(axis=0, keepdims=True)
        _, P_ABdiff = radial_spectrum(deltaD, dr)
        noise_binned[n] = bin_median(lam, P_ABdiff / spectra[n], bin_edges)
        ab_agree_binned[n] = bin_median(lam, np.abs(P_A - P_B) / P_A, bin_edges)
        rowN = "".join("       -" if np.isnan(v) else f" {v:7.1%}" for v in noise_binned[n])
        print(f"{str(n) + '^3':>8} P_AB/P_full {rowN}")
    print(f"noise fraction > {args.noise_thresh:.0%} marks noise-dominated scales; A/B spectral agreement (median |P_A-P_B|/P_A):")
    for n in args.ab_resolutions:
        rowA = "".join("       -" if np.isnan(v) else f" {v:7.1%}" for v in ab_agree_binned[n])
        print(f"{str(n) + '^3':>8} agree     {rowA}")
    cutoff_noise = {}
    for c in args.cutoffs:
        b = int(np.clip(np.digitize(c, bin_edges) - 1, 0, len(bin_centers) - 1))
        cutoff_noise[f"{c:g}"] = {str(n): None if np.isnan(noise_binned[n][b]) else float(noise_binned[n][b])
                                  for n in args.ab_resolutions}
        vals = "  ".join(f"{n}^3 {cutoff_noise[f'{c:g}'][str(n)]:6.1%}" for n in args.ab_resolutions)
        print(f"lambda={c:>4g} kpc  noise fraction  {vals}")

    # ---- persist ----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "resolution_convergence.npz",
        freq=freq_ref, lam=lam, bin_edges=bin_edges, bin_centers=bin_centers,
        **{f"P{n}": spectra[n] for n in args.resolutions},
        **{f"rel_{n1}_{n2}": rel_binned[(n1, n2)] for n1, n2 in adjacent},
        gate_conv=gate_conv,
        **{f"noise_{n}": noise_binned[n] for n in args.ab_resolutions},
        **{f"abagree_{n}": ab_agree_binned[n] for n in args.ab_resolutions})
    summary = dict(
        config=dict(resolutions=args.resolutions, ab_resolutions=args.ab_resolutions,
                    extent_kpc=args.extent, n_dirs=args.n_dirs, n_radial=args.n_radial,
                    r_min_kpc=args.r_min, r_max_kpc=args.r_max, dr_kpc=dr,
                    cutoffs_kpc=args.cutoffs, bin_edges=list(args.bin_edges),
                    rel_tol=args.rel_tol, nyquist_factor=args.nyquist_factor,
                    noise_thresh=args.noise_thresh, sobol_seed=args.sobol_seed,
                    split_seed=args.split_seed),
        lineage=dict(asset=str(args.asset), asset_sha256=asset_sha,
                     grids=str(args.grids), grids_source_sha256=str(recorded), match=True),
        grids={str(n): grid_info[n] for n in args.resolutions},
        f_hi={str(n): fhi_rows[n] for n in args.resolutions},
        f_hi_convergence=fhi_conv,
        convergence={f"{n1}-{n2}": dict(
            rel_binned=[None if np.isnan(v) else float(v) for v in rel_binned[(n1, n2)]],
            converged=[bool(v) for v in conv_binned[(n1, n2)]],
            gate=bool((n1, n2) in gate_pairs)) for n1, n2 in adjacent},
        shot_noise={str(n): dict(
            noise_ratio_binned=[None if np.isnan(v) else float(v) for v in noise_binned[n]],
            ab_agree_binned=[None if np.isnan(v) else float(v) for v in ab_agree_binned[n]],
            noise_at_cutoffs=cutoff_noise) for n in args.ab_resolutions},
        conclusion=dict(lambda_reliable_kpc=lam_reliable,
                        lambda_empirical_kpc=lam_empirical,
                        guard_kpc=guard, guard15_kpc=guard15,
                        lambda_reliable_15x_kpc=lam_reliable15,
                        basis=basis))
    json.dump(summary, open(args.output_dir / "resolution_convergence.json", "w"), indent=2)

    print("=== CONCLUSION ===")
    print(f"lambda_conv_empirical = {lam_empirical:.3f} kpc (smallest converged bin centre, gate pairs, tol {args.rel_tol:.0%})")
    print(f"nyquist guard {args.nyquist_factor:g}x(finest grid) = {guard:.3f} kpc (1.5x variant {guard15:.3f} kpc)")
    print(f"LAMBDA_RELIABLE = {lam_reliable:.3f} kpc ({basis}; 1.5x-guard variant {lam_reliable15:.3f} kpc)")
    print(f"TOTAL WALL {time.time()-t0:.1f}s")
    print("RESOLUTION_CONVERGENCE_DONE")
    print("=== OUTPUTS ===")
    print(args.output_dir / "resolution_convergence.npz")
    print(args.output_dir / "resolution_convergence.json")


if __name__ == "__main__":
    main()
