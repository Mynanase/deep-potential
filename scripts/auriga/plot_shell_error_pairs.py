#!/usr/bin/env python
"""Paired shell errors vs the particle-truth histogram field.

Upgrade of the scheme-1 shell adjudication (node 4b21457d): per shell of the
60-shell particle truth (edges 1.0906..~67.3 kpc), report the SIGNED and the
ABSOLUTE shell error of each model density against the particle-truth density,
and their ratio - the cancellation factor:

  e(r)  = int_shell (rho_model - rho_true) dV      [signed shell-mass error]
  E(r)  = int_shell |rho_model - rho_true| dV      [absolute shell error]
  kappa(r) = E(r) / |e(r)|                         kappa=1 one-sided error,
                                                    kappa>>1 large cancellation

Truth conventions (stated, not hidden):
  * rho_true field = density/rho3d of the committed grid product
    (96^3 = 1.5625 kpc cells, total-matter mass histogram, star frame).
  * e uses the EXACT per-shell particle mass M_shell_total of the 60-shell
    truth product: e = M_model(shell) - M_shell_total, with M_model from
    Sobol(2048 dirs) x Gauss-Legendre(48 radial nodes in r^3) quadrature of
    the smooth model Laplacian density. The same quadrature applied to the
    histogram field gives an independent estimate e_mc (validation).
  * E integrates |rho_model - rho3d| pointwise on the same quadrature;
    the model is a point value, the truth a cell average (second-order
    discretization mismatch, same caveat as plot_particle_truth_2d.py).

Poisson floors (truth is a particle realization):
  * sigma_e(shell) = sqrt(sum_i M_shell,i * m_i) - exact for equal-mass
    particles per type; even a perfect model of the smooth field shows
    |e| ~ sigma_e.
  * E_floor(shell) = sqrt(2/pi) * sum_cells f_cell * sqrt(sum_i n_cell,i m_i^2)
    with per-cell per-type counts n from the source particle asset (sha256
    verified against the grid product attrs); f_cell from the quadrature.
    A perfect smooth model shows E >= E_floor. Without the asset an
    approximate floor (shell-level type mixture) is used and marked.

Models: the five frozen Phis of node 4b21457d (base, S1, gridprior, innerA,
innerB). No retraining. Figures: (a) M_shell log-log truth vs models;
(b) e/M and E/M with the Poisson floors; (c) kappa(r); plus per-model
small multiples.
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from validate_enclosed_mass import (  # noqa: E402
    G_KPC_KMS2_MSUN, sobol_directions, rho_from_phi)
from plot_particle_truth_2d import load_phi_f32  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "S1", "gridprior", "innerA", "innerB")
COLOR = {"base": "blue", "S1": "red", "gridprior": "green",
         "innerA": "purple", "innerB": "orange"}
TYPES = ("PartType0", "PartType1", "PartType4")


def sha256_of(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--truth60", type=Path, required=True,
                    help="60-shell particle truth hdf5 (exact M_shell per type)")
    ap.add_argument("--particles", type=Path, default=None,
                    help="source particle asset (exact per-cell E floor)")
    ap.add_argument("--model", action="append", required=True,
                    help="label=run_dir, repeatable")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/shell-error-pairs"))
    ap.add_argument("--r-min", type=float, default=1.0906)
    ap.add_argument("--r-max", type=float, default=70.0)
    ap.add_argument("--n-dirs", type=int, default=2048)
    ap.add_argument("--n-radial", type=int, default=48)
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32768,
                    help="points per model Laplacian call (memory bound)")
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"CONFIG n_dirs={args.n_dirs} n_radial={args.n_radial} "
          f"n_blocks={args.n_blocks} sobol_seed={args.sobol_seed} "
          f"r=[{args.r_min},{args.r_max}] kpc")
    if args.n_dirs % args.n_blocks:
        ap.error("--n-dirs must be divisible by --n-blocks")
    t_start = time.time()

    # ---- truth shell set and exact per-type shell masses -----------------
    with h5py.File(args.truth60, "r") as f:
        r_edges_all = np.asarray(f["r_edges"][:], dtype=float)
        m60 = np.asarray(f["M_shell_total"][:], dtype=float)
        m60_cum = np.asarray(f["M_cum_total"][:], dtype=float)
        m60_type = {t: np.asarray(f[f"{t}/M_shell"][:], dtype=float)
                    for t in TYPES}
    i0 = int(np.argmin(np.abs(r_edges_all - args.r_min)))
    if not np.isclose(r_edges_all[i0], args.r_min, rtol=1e-4):
        ap.error(f"r_min {args.r_min} not a truth edge "
                 f"(nearest {r_edges_all[i0]:.6f})")
    i1 = int(np.flatnonzero(r_edges_all <= args.r_max)[-1])
    r_edges = r_edges_all[i0:i1 + 1]
    m_true = m60[i0:i1]                       # exact particle shell masses
    m_true_type = {t: m60_type[t][i0:i1] for t in TYPES}
    n_shell = r_edges.size - 1
    r_lo, r_hi = r_edges[:-1], r_edges[1:]
    r_c = np.sqrt(r_lo * r_hi)
    v_sh = 4.0 * np.pi / 3.0 * (r_hi ** 3 - r_lo ** 3)
    cum_check = m60_cum[i1 - 1] - m60_cum[i0 - 1]
    assert np.isclose(cum_check, m_true.sum(), rtol=1e-6), \
        "M_shell_total inconsistent with M_cum_total"
    print(f"shells: {n_shell} from {r_edges[0]:.4f} to {r_edges[-1]:.2f} kpc "
          f"(60-shell product, edge ratio {r_edges[1]/r_edges[0]:.4f}); "
          f"M_true total {m_true.sum():.4e} Msun")

    # ---- grid product: histogram field, per-type particle masses ---------
    with h5py.File(args.grids, "r") as f:
        ga = dict(f.attrs)
        rho3d = np.asarray(f["density/rho3d"][:], dtype=float)
        edges3d = np.asarray(f["density/rho3d_edges_kpc"][:], dtype=float)
        r_nodes = np.asarray(f["potential/r_nodes"][:], dtype=float)
        phi_prof = np.asarray(f["potential/phi_sphere_mean"][:], dtype=float)
    m_part = {t: float(ga[f"{t}_mass_msun"]) / int(ga[f"{t}_n"])
              for t in TYPES}
    h_cell = float(np.abs(np.diff(edges3d[0])).max())
    v_cell = h_cell ** 3
    print(f"rho3d {rho3d.shape} cell {h_cell:.4f} kpc; per-particle masses "
          + ", ".join(f"{t}={m_part[t]:.3e}" for t in TYPES))

    # ---- per-cell sigma_rho for the E floor ------------------------------
    edge_list = [edges3d[a] for a in range(3)]
    if args.particles is not None and Path(args.particles).is_file():
        got = sha256_of(args.particles)
        want = str(ga.get("source_sha256", ""))
        if got != want:
            ap.error(f"particle asset sha256 mismatch: {got} != {want}")
        var_cell = np.zeros_like(rho3d)
        rho_from_counts = np.zeros_like(rho3d)
        m2_shell = np.zeros(n_shell)
        with h5py.File(args.particles, "r") as f:
            for t in TYPES:
                xyz = np.column_stack([
                    np.asarray(f[f"{t}/x"][:], dtype=float),
                    np.asarray(f[f"{t}/y"][:], dtype=float),
                    np.asarray(f[f"{t}/z"][:], dtype=float)])
                m_i = np.asarray(f[f"{t}/mass"][:], dtype=float)
                cnt, _ = np.histogramdd(xyz, bins=edge_list)
                assert int(np.rint(cnt.sum())) == int(ga[f"{t}_n"]), \
                    f"particle count mismatch for {t}"
                m2_cell, _ = np.histogramdd(xyz, bins=edge_list,
                                            weights=m_i ** 2)
                m_cell, _ = np.histogramdd(xyz, bins=edge_list, weights=m_i)
                var_cell += m2_cell
                rho_from_counts += m_cell / v_cell
                r_i = np.linalg.norm(xyz, axis=1)
                m2_shell += np.histogram(r_i, bins=r_edges,
                                         weights=m_i ** 2)[0]
        rel = np.abs(rho_from_counts - rho3d) / np.maximum(rho3d, 1e-30)
        print(f"particle asset sha256 OK; counts->rho3d max rel diff "
              f"{rel.max():.2e} (median {np.median(rel):.2e})")
        sigma_rho_cell = np.sqrt(var_cell) / v_cell
        sigma_e = np.sqrt(m2_shell)      # exact: sum of m^2 per shell
        floor_mode = "exact per-cell per-type counts"
    else:
        # approximate: shell-level number mixture spread over cell mass
        num_frac = {t: m_true_type[t] / m_part[t] for t in TYPES}
        tot_num = sum(num_frac.values())
        p_num = {t: num_frac[t] / tot_num for t in TYPES}
        cx = edges3d[0][:-1] + 0.5 * h_cell
        cy = edges3d[1][:-1] + 0.5 * h_cell
        cz = edges3d[2][:-1] + 0.5 * h_cell
        r_cell = np.sqrt(np.add.outer(np.add.outer(cx ** 2, cy ** 2), cz ** 2))
        sigma_rho_cell = np.zeros_like(rho3d)
        for j in range(n_shell):
            m2_over_m1 = sum(p_num[t] * m_part[t] ** 2 for t in TYPES) / \
                sum(p_num[t] * m_part[t] for t in TYPES)
            sel = (r_cell >= r_lo[j]) & (r_cell < r_hi[j])
            sigma_rho_cell[sel] = np.sqrt(rho3d[sel] * m2_over_m1 / v_cell)
        floor_mode = ("APPROXIMATE shell-level type mixture "
                      "(particle asset not provided)")
        print("WARNING: particle asset missing - approximate E floor")
        sigma_e = np.sqrt(sum(m_true_type[t] * m_part[t] for t in TYPES))
    n_eff = m_true ** 2 / sigma_e ** 2
    print(f"E-floor mode: {floor_mode}")

    # ---- quadrature points per shell ------------------------------------
    dirs = sobol_directions(args.n_dirs, args.sobol_seed)
    u_g, w_g = np.polynomial.legendre.leggauss(args.n_radial)
    u_g = 0.5 * (u_g + 1.0)
    w_g = 0.5 * w_g
    assert np.isclose(w_g.sum(), 1.0)
    r_g = (r_lo[:, None] ** 3 + u_g[None, :]
           * (r_hi[:, None] ** 3 - r_lo[:, None] ** 3)) ** (1.0 / 3.0)

    # truth field + cell sigma at all points, per shell (n_shell, G, N)
    rho_t_pts = np.empty((n_shell, args.n_radial, args.n_dirs))
    sig_pts = np.empty_like(rho_t_pts)
    for j in range(n_shell):
        pts = r_g[j][:, None, None] * dirs[None, :, :]   # (G, N, 3)
        idx = [np.clip(np.searchsorted(edge_list[a], pts[..., a],
                                       side="right") - 1, 0,
                       rho3d.shape[a] - 1) for a in range(3)]
        rho_t_pts[j] = rho3d[idx[0], idx[1], idx[2]]
        sig_pts[j] = sigma_rho_cell[idx[0], idx[1], idx[2]]
    m_true_mc = v_sh * np.einsum("g,sg->s", w_g, rho_t_pts.mean(axis=2))
    rel_mc = np.abs(m_true_mc / m_true - 1.0)
    e_floor = np.sqrt(2.0 / np.pi) * v_sh * np.einsum(
        "g,sg->s", w_g, sig_pts.mean(axis=2))
    print("MC truth shell mass vs exact M_shell_total: median "
          f"{100*np.median(rel_mc):.2f}% max {100*rel_mc.max():.2f}% "
          f"(at r={r_c[np.argmax(rel_mc)]:.2f} kpc) - cell quantization + "
          "quadrature noise of the histogram field")

    # monopole cross-check of the shell set (parent node did the same)
    vc2 = np.abs(np.gradient(phi_prof, np.log(r_nodes)))
    m_mono = vc2 * r_nodes / G_KPC_KMS2_MSUN
    dm_mono = m_mono - m_mono[0]
    cum_true = np.cumsum(m_true)
    ours = np.interp(r_hi, r_nodes, dm_mono)
    rel_mono = np.abs(ours / cum_true - 1.0)
    print(f"monopole dM vs particle-shell cumsum: median "
          f"{100*np.median(rel_mono):.2f}% max {100*rel_mono.max():.2f}% "
          f"(at r={r_hi[np.argmax(rel_mono)]:.2f} kpc)")

    # ---- per-model evaluation -------------------------------------------
    B = args.n_blocks
    D = args.n_dirs // B
    res = {}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        m_model = np.empty(n_shell)
        e_mc = np.empty(n_shell)
        e_mc_se = np.empty(n_shell)
        e_abs = np.empty(n_shell)
        e_abs_se = np.empty(n_shell)
        for j in range(n_shell):
            q = (r_g[j][:, None, None] * dirs[None, :, :]) / L_KPC
            flat_q = q.reshape(-1, 3)
            rho_flat = np.empty(flat_q.shape[0])
            for i in range(0, flat_q.shape[0], args.eval_batch):
                rho_flat[i:i + args.eval_batch] = np.asarray(rho_from_phi(
                    phi, flat_q[i:i + args.eval_batch], L_KPC, V_KMS))
            rho_m = rho_flat.reshape(args.n_radial, args.n_dirs)
            m_model[j] = v_sh[j] * float(
                np.einsum("g,g->", w_g, rho_m.mean(axis=1)))
            delta = rho_m - rho_t_pts[j]
            per_dir = w_g[:, None] * delta               # (G, N)
            e_mc[j] = v_sh[j] * float(per_dir.sum(axis=0).mean())
            blocks = per_dir.reshape(args.n_radial, B, D).mean(axis=2)
            e_mc_se[j] = v_sh[j] * float(
                blocks.sum(axis=0).std(ddof=1) / np.sqrt(B))
            per_dir_a = np.abs(per_dir)
            e_abs[j] = v_sh[j] * float(per_dir_a.sum(axis=0).mean())
            blocks_a = per_dir_a.reshape(args.n_radial, B, D).mean(axis=2)
            e_abs_se[j] = v_sh[j] * float(
                blocks_a.sum(axis=0).std(ddof=1) / np.sqrt(B))
        e_exact = m_model - m_true
        kappa = e_abs / np.abs(e_exact)
        res[label] = dict(m_model=m_model, e_exact=e_exact, e_mc=e_mc,
                          e_mc_se=e_mc_se, e_abs=e_abs, e_abs_se=e_abs_se,
                          kappa=kappa)
        dev = np.abs(e_mc - e_exact) / np.maximum(e_mc_se, 1e-30)
        print(f"[{label}] eval {time.time()-t0:.0f}s; e_mc-e_exact max "
              f"{100*np.max(np.abs(e_mc-e_exact)/m_true):.2f}% of M_shell "
              f"({dev.max():.1f} sigma of e_mc SE); E max SE/M "
              f"{np.max(e_abs_se/m_true)*100:.2f}%")

    # ---- band aggregates --------------------------------------------------
    bands = ((r_edges[0], 2.0), (2.0, 10.0), (10.0, 30.0), (30.0, 50.0),
             (50.0, r_edges[-1]))
    band_rows = []
    print("=== band aggregates: e/M [%], E/M [%], kappa=E/|e| (shell sums) ===")
    print("  band [kpc]      sig_e/M  Efl/M   "
          + "  ".join(f"{l:>20}" for l in MODELS))
    for lo, hi in bands:
        m = (r_c >= lo) & (r_c < hi)
        mb = float(m_true[m].sum())
        s_e = float(np.sqrt((sigma_e[m] ** 2).sum()))
        f_e = float(e_floor[m].sum())
        cells = []
        for l in MODELS:
            r = res[l]
            eb = float(r["e_exact"][m].sum())
            ab = float(r["e_abs"][m].sum())
            kb = ab / abs(eb) if eb != 0 else np.inf
            cells.append(f"{100*eb/mb:+5.2f} {100*ab/mb:5.2f} {kb:6.1f}")
            band_rows.append(dict(band_lo=lo, band_hi=hi, model=l,
                                  M_band_msun=mb, e_band_msun=eb,
                                  E_band_msun=ab, kappa_band=kb,
                                  sigma_e_band_msun=s_e,
                                  E_floor_band_msun=f_e))
        print(f"  {lo:5.2f}-{hi:5.2f} ({int(m.sum()):2d} sh)  "
              f"{100*s_e/mb:7.2f}  {100*f_e/mb:5.2f}  "
              + "  ".join(cells))

    # ---- per-shell table ---------------------------------------------------
    print("=== per-shell: r [kpc], M_true, floors [%], then per model "
          "e/M [%], E/M [%], kappa ===")
    for j in range(n_shell):
        row = (f"  {r_lo[j]:6.3f}-{r_hi[j]:6.3f}  {m_true[j]:.3e}  "
               f"{100*sigma_e[j]/m_true[j]:5.2f} {100*e_floor[j]/m_true[j]:5.2f}")
        for l in MODELS:
            r = res[l]
            row += (f"  | {100*r['e_exact'][j]/m_true[j]:+6.2f}"
                    f" {100*r['e_abs'][j]/m_true[j]:6.2f}"
                    f" {r['kappa'][j]:6.1f}")
        print(row)
    for l in MODELS:
        kap = res[l]["kappa"]
        n_below = int((np.abs(res[l]["e_exact"]) < sigma_e).sum())
        print(f"  kappa[{l}]: min {kap.min():.2f} median {np.median(kap):.2f} "
              f"max {kap.max():.1f} (at r={r_c[np.argmax(kap)]:.2f} kpc); "
              f"shells with |e|<sigma_e: {n_below}/{n_shell}")

    # ---- outputs -----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "shell_error_pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["r_lo_kpc", "r_hi_kpc", "r_center_kpc", "M_true_msun",
                     "sigma_e_msun", "E_floor_msun", "N_eff"]
                    + [f"{l}_{c}" for l in MODELS for c in
                       ("e_msun", "e_over_M", "E_msun", "E_over_M", "kappa",
                        "E_SE_msun")])
        for j in range(n_shell):
            row = [f"{r_lo[j]:.6f}", f"{r_hi[j]:.6f}", f"{r_c[j]:.6f}",
                   f"{m_true[j]:.8e}", f"{sigma_e[j]:.8e}",
                   f"{e_floor[j]:.8e}", f"{n_eff[j]:.1f}"]
            for l in MODELS:
                r = res[l]
                row += [f"{r['e_exact'][j]:.8e}",
                        f"{r['e_exact'][j]/m_true[j]:.6f}",
                        f"{r['e_abs'][j]:.8e}",
                        f"{r['e_abs'][j]/m_true[j]:.6f}",
                        f"{r['kappa'][j]:.6f}",
                        f"{r['e_abs_se'][j]:.8e}"]
            wr.writerow(row)
    with open(args.output_dir / "shell_error_pairs_bands.csv", "w",
              newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(band_rows[0].keys()))
        wr.writeheader()
        wr.writerows(band_rows)

    ofs.use_style()
    # main figure: (a) M_shell, (b) e/M + E/M + floors, (c) kappa
    fig, axes = ofs.figure_grid(3, 1, width=ofs.WIDE, ratio=1.08,
                                sharex=True)
    ax = axes[0]
    ax.plot(r_c, m_true, color="k", lw=1.8, marker="o", ms=3.0,
            label="particle truth (exact shell sum)", zorder=3)
    for l in MODELS:
        ax.plot(r_c, res[l]["m_model"], color=ofs.PALETTE[COLOR[l]], lw=1.1,
                marker="o", ms=2.2, label=l, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_ylabel(r"$M_{\rm shell}(r)$  [Msun]")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    ax = axes[1]
    ax.fill_between(r_c, -100 * sigma_e / m_true, 100 * sigma_e / m_true,
                    color="0.88", zorder=0, label=r"Poisson floor of $e$")
    ax.fill_between(r_c, 0, 100 * e_floor / m_true, color="0.72", zorder=0,
                    label=r"Poisson floor of $E$")
    for l in MODELS:
        c = ofs.PALETTE[COLOR[l]]
        ax.plot(r_c, 100 * res[l]["e_abs"] / m_true, color=c, lw=1.3,
                label=f"{l}  $E/M$")
        ax.plot(r_c, 100 * res[l]["e_exact"] / m_true, color=c, lw=1.0,
                ls="--", marker="o", ms=2.0, label=f"{l}  $e/M$")
    ax.axhline(0.0, color="0.2", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_ylabel("shell error / $M_{\\rm shell}$  [%]")
    ax.set_xlim(r_c[0] * 0.92, r_c[-1] * 1.08)
    ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=4)
    ax = axes[2]
    ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
    for l in MODELS:
        ax.plot(r_c, res[l]["kappa"], color=ofs.PALETTE[COLOR[l]], lw=1.2,
                marker="o", ms=2.0)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_yscale("log")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$\kappa(r) = E/|e|$")
    ax.set_xlim(r_c[0] * 0.92, r_c[-1] * 1.08)
    ax.annotate(r"$\kappa=1$: one-sided", xy=(0.99, 1.0),
                xycoords=("axes fraction", "data"), xytext=(-4, 3),
                textcoords="offset points", ha="right", fontsize=7,
                color="0.3")
    ofs.panel_labels(list(axes))
    for p in ofs.save(fig, str(args.output_dir / "shell-error-pairs")):
        print("saved", p)

    # per-model small multiples: the e/E pocket with floors
    fig, axes = ofs.figure_grid(2, 3, width=ofs.WIDE, ratio=0.62)
    flat = list(np.ravel(axes))
    for ax, l in zip(flat, MODELS):
        ax.fill_between(r_c, -100 * sigma_e / m_true, 100 * sigma_e / m_true,
                        color="0.88", zorder=0)
        ax.fill_between(r_c, 0, 100 * e_floor / m_true, color="0.72", zorder=0)
        c = ofs.PALETTE[COLOR[l]]
        ax.plot(r_c, 100 * res[l]["e_abs"] / m_true, color=c, lw=1.4)
        ax.plot(r_c, 100 * res[l]["e_exact"] / m_true, color=c, lw=1.0,
                ls="--", marker="o", ms=2.0)
        ax.axhline(0.0, color="0.2", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks([2, 10, 50])
        ax.annotate(l + "\n$E/M$ solid, $e/M$ dashed", xy=(0.97, 0.95),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=7, color="#333333")
    for ax in flat[len(MODELS):]:
        ax.set_visible(False)
    flat[0].set_ylabel("e, E / $M_{\\rm shell}$  [%]")
    flat[3].set_ylabel("e, E / $M_{\\rm shell}$  [%]")
    for ax in flat[3:len(MODELS)]:
        ax.set_xlabel("r [kpc]")
    ofs.panel_labels(flat[:len(MODELS)])
    for p in ofs.save(fig, str(args.output_dir / "shell-error-pairs-permodel")):
        print("saved", p)

    np.savez(args.output_dir / "shell_error_pairs.npz",
             r_edges=r_edges, r_center=r_c, m_true=m_true, v_sh=v_sh,
             sigma_e=sigma_e, e_floor=e_floor, n_eff=n_eff, m_true_mc=m_true_mc,
             bands=np.asarray([(a, b) for a, b in bands]),
             floor_mode=floor_mode, h_cell=h_cell,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in res[l].items()})
    print(f"TOTAL WALL {time.time()-t_start:.1f}s")
    print("SHELL_ERROR_PAIRS_DONE")


if __name__ == "__main__":
    sys.exit(main())
