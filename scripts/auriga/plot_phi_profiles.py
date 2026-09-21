#!/usr/bin/env python
"""Phi-direction mass profiles per shell in fixed theta bands.

Analysis 2 of the paired-shell line (parent nodes 96c6df11 / 3da61b51):
for selected shells of the 60-shell particle truth, the mass in phi bins is
compared between each model and the particles, WITHOUT integrating over the
full sphere - theta is held in fixed bands so that positive/negative
residuals at different theta cannot cancel inside a phi bin (the trap that
would re-hide the radial cancellation one level down):

  bands: equatorial  theta 45-135 deg (primary),
         north/south mid-latitude 31.8-45 / 135-148.2 deg;
         polar caps |cos theta| > 0.85 excluded (small solid angle per
         dphi cell, shot-noise dominated, weak physical meaning).

  truth per (shell, band, phi bin) = EXACT particle sum from the source
    asset (sha256-verified against the grid product), with the exact
    Poisson sigma = sqrt(sum m^2) per cell.
  model per cell = V_cell x GL(24 radial nodes in r^3) x mean over
    Sobol(512 directions per cell, scrambled per cell) of the Laplacian
    density; block SE (8 direction blocks) reported in the CSV.
  V_cell = (r2^3 - r1^3)/3 * dphi * dmu  (exact wedge volume).

Figures (linear y axes; phi in degrees):
  phi-profiles-equator: one row per selected shell - top: M(phi bin),
    truth vs the five models (model dips below zero while truth is
    positive = negative density in the phi direction); bottom: signed
    DeltaM/M_true per bin with the +/-1 sigma Poisson band and, annotated
    per model, Sigma (signed sum over bins) and Sigma|.| (absolute sum) -
    alternating +/- lobes with Sigma ~ 0 and Sigma|.| large are the
    phi-cancellation fingerprint.
  phi-profiles-midlat: same for the north and south mid-latitude bands,
    three shells.
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
    sobol_directions, rho_from_phi)
from plot_particle_truth_2d import load_phi_f32  # noqa: E402
import orx_figstyle as ofs  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
MODELS = ("base", "S1", "gridprior", "innerA", "innerB")
COLOR = {"base": "blue", "S1": "red", "gridprior": "green",
         "innerA": "purple", "innerB": "orange"}
TYPES = ("PartType0", "PartType1", "PartType4")
MU_EQ = np.sqrt(0.5)          # cos 45 deg
MU_POLAR = 0.85               # |cos theta| > 0.85 excluded
BANDS = (("eq", -MU_EQ, MU_EQ, "equatorial 45-135 deg"),
         ("north", MU_EQ, MU_POLAR, "north mid-lat 31.8-45 deg"),
         ("south", -MU_POLAR, -MU_EQ, "south mid-lat 135-148.2 deg"))


def sha256_of(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cell_dirs(n, mu_lo, mu_hi, phi_lo, phi_hi, seed):
    """n directions uniform in the (mu, phi) wedge via a scrambled Sobol."""
    from scipy.stats import qmc
    eng = qmc.Sobol(d=2, scramble=True, seed=int(seed))
    uv = eng.random(int(n))
    mu = mu_lo + uv[:, 0] * (mu_hi - mu_lo)
    phi = phi_lo + uv[:, 1] * (phi_hi - phi_lo)
    s = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), mu])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=Path,
                    default=Path("data/auriga/halo12_particle_truth_grids.h5"))
    ap.add_argument("--truth60", type=Path, required=True)
    ap.add_argument("--particles", type=Path, default=None,
                    help="source particle asset (exact per-cell truth)")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("figures/phi-profiles"))
    ap.add_argument("--shell-r", type=float, action="append", default=None,
                    help="target radius [kpc] of a shell to profile (repeat)")
    ap.add_argument("--midlat-r", type=float, action="append", default=None,
                    help="subset of --shell-r used in the mid-lat figure")
    ap.add_argument("--r-min", type=float, default=1.0906)
    ap.add_argument("--r-max", type=float, default=70.0)
    ap.add_argument("--n-phi", type=int, default=24)
    ap.add_argument("--n-dirs-cell", type=int, default=512)
    ap.add_argument("--n-radial", type=int, default=24)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--sobol-seed", type=int, default=20260917)
    args = ap.parse_args()
    if args.shell_r is None:
        args.shell_r = [1.5, 4.4, 9.6, 20.0, 40.0, 62.0]
    if args.midlat_r is None:
        args.midlat_r = [4.4, 9.6, 62.0]

    import jax
    jax.config.update("jax_enable_x64", True)
    print(f"JAX x64={bool(jax.config.jax_enable_x64)} devices={jax.devices()}")
    print(f"CONFIG shells(r)={args.shell_r} midlat={args.midlat_r} "
          f"n_phi={args.n_phi} n_dirs_cell={args.n_dirs_cell} "
          f"n_radial={args.n_radial} n_blocks={args.n_blocks} "
          f"bands={[(b[0], round(b[1],3), round(b[2],3)) for b in BANDS]} "
          f"polar_cut=|mu|>{MU_POLAR}")
    t_start = time.time()

    # ---- shell set ---------------------------------------------------------
    with h5py.File(args.truth60, "r") as f:
        r_edges_all = np.asarray(f["r_edges"][:], dtype=float)
        m60 = np.asarray(f["M_shell_total"][:], dtype=float)
    i0 = int(np.argmin(np.abs(r_edges_all - args.r_min)))
    if not np.isclose(r_edges_all[i0], args.r_min, rtol=1e-4):
        ap.error(f"r_min {args.r_min} not a truth edge")
    i1 = int(np.flatnonzero(r_edges_all <= args.r_max)[-1])
    edges_all = r_edges_all[i0:i1 + 1]
    r_lo_all, r_hi_all = edges_all[:-1], edges_all[1:]
    r_c_all = np.sqrt(r_lo_all * r_hi_all)
    sel = sorted({int(np.argmin(np.abs(r_c_all - rv))) for rv in args.shell_r})
    n_sh = len(sel)
    r_lo, r_hi, r_c = r_lo_all[sel], r_hi_all[sel], r_c_all[sel]
    m_true_shell = m60[i0:i1][sel]
    print("selected shells:" + "".join(
        f"\n  {r_lo[j]:.3f}-{r_hi[j]:.3f} kpc (r_c={r_c[j]:.2f}, "
        f"M_true={m_true_shell[j]:.3e})" for j in range(n_sh)))
    mid_sel = []
    for rv in args.midlat_r:
        j = min(range(n_sh), key=lambda k: abs(r_c[k] - rv))
        if j not in mid_sel:
            mid_sel.append(j)

    # ---- truth grid product (for the field-consistency check) --------------
    with h5py.File(args.grids, "r") as f:
        ga = dict(f.attrs)
        rho3d = np.asarray(f["density/rho3d"][:], dtype=float)
        edges3d = np.asarray(f["density/rho3d_edges_kpc"][:], dtype=float)
    edge_list = [edges3d[a] for a in range(3)]

    # ---- cells: directions, radial nodes, volumes --------------------------
    n_phi = args.n_phi
    dphi = 2.0 * np.pi / n_phi
    phi_c = (np.arange(n_phi) + 0.5) * dphi
    u_g, w_g = np.polynomial.legendre.leggauss(args.n_radial)
    u_g = 0.5 * (u_g + 1.0)
    w_g = 0.5 * w_g
    assert np.isclose(w_g.sum(), 1.0)
    B = args.n_blocks
    if args.n_dirs_cell % B:
        ap.error("--n-dirs-cell must be divisible by --n-blocks")
    D = args.n_dirs_cell // B
    cells = []
    for j in range(n_sh):
        r_g = (r_lo[j] ** 3 + u_g * (r_hi[j] ** 3 - r_lo[j] ** 3)) ** (1 / 3)
        for bi, (bname, mu1, mu2, _) in enumerate(BANDS):
            for p in range(n_phi):
                d = cell_dirs(args.n_dirs_cell, mu1, mu2, p * dphi,
                              (p + 1) * dphi,
                              args.sobol_seed + 1000 * j + 100 * bi + p)
                pts = r_g[:, None, None] * d[None, :, :]      # (G, N, 3)
                v_cell = (r_hi[j] ** 3 - r_lo[j] ** 3) / 3.0 * dphi * (mu2 - mu1)
                idx = [np.clip(np.searchsorted(edge_list[a], pts[..., a],
                                               side="right") - 1, 0,
                               rho3d.shape[a] - 1) for a in range(3)]
                rho_t_field = rho3d[idx[0], idx[1], idx[2]]
                cells.append(dict(shell=j, band=bname, phi=p, dirs=d, r_g=r_g,
                                  pts=pts, v_cell=v_cell,
                                  rho_t_field=rho_t_field))
    v_sum = sum(c["v_cell"] for c in cells if c["shell"] == 0
                and c["band"] == "eq")
    v_shell0 = 4 * np.pi / 3 * (r_hi[0] ** 3 - r_lo[0] ** 3)
    omega_frac = MU_EQ  # eq band solid-angle fraction = sqrt(0.5)
    assert np.isclose(v_sum / v_shell0, omega_frac, rtol=1e-12), \
        "wedge volume bookkeeping broken"
    print(f"cells: {len(cells)} = {n_sh} shells x {len(BANDS)} bands x "
          f"{n_phi} phi bins; eq-band volume fraction check "
          f"{v_sum/v_shell0:.6f} == sqrt(0.5) OK")

    # ---- exact per-cell truth from the particle asset ----------------------
    n_cell = len(cells)
    m_true = np.zeros(n_cell)
    sig_true = np.zeros(n_cell)
    if args.particles is not None and Path(args.particles).is_file():
        got = sha256_of(args.particles)
        if got != str(ga.get("source_sha256", "")):
            ap.error("particle asset sha256 mismatch")
        with h5py.File(args.particles, "r") as f:
            for t in TYPES:
                xyz = np.column_stack([
                    np.asarray(f[f"{t}/x"][:], dtype=float),
                    np.asarray(f[f"{t}/y"][:], dtype=float),
                    np.asarray(f[f"{t}/z"][:], dtype=float)])
                m_i = np.asarray(f[f"{t}/mass"][:], dtype=float)
                r_i = np.linalg.norm(xyz, axis=1)
                for j in range(n_sh):
                    sm = (r_i >= r_lo[j]) & (r_i < r_hi[j])
                    if not sm.any():
                        continue
                    mu_i = xyz[sm, 2] / r_i[sm]
                    phi_i = np.mod(np.arctan2(xyz[sm, 1], xyz[sm, 0]),
                                   2 * np.pi)
                    for bi, (bname, mu1, mu2, _) in enumerate(BANDS):
                        bm = (mu_i >= mu1) & (mu_i < mu2)
                        if not bm.any():
                            continue
                        h1, _ = np.histogram(phi_i[bm], bins=n_phi,
                                             range=(0.0, 2 * np.pi),
                                             weights=m_i[sm][bm])
                        h2, _ = np.histogram(phi_i[bm], bins=n_phi,
                                             range=(0.0, 2 * np.pi),
                                             weights=m_i[sm][bm] ** 2)
                        for p in range(n_phi):
                            k = j * len(BANDS) * n_phi + bi * n_phi + p
                            m_true[k] += h1[p]
                            sig_true[k] += h2[p]
        sig_true = np.sqrt(sig_true)
        for j in range(n_sh):
            m_band = sum(m_true[k] for k in range(len(cells))
                         if cells[k]["shell"] == j)
            print(f"  shell {r_lo[j]:.2f}-{r_hi[j]:.2f}: 3-band mass "
                  f"{m_band/m_true_shell[j]*100:.2f}% of exact M_shell_total "
                  f"(geometric expectation {MU_POLAR*100:.1f}%)")
        truth_mode = "exact particle sums per cell"
    else:
        print("WARNING: particle asset missing - truth falls back to the "
              "rho3d field; Poisson sigma unavailable")
        for k, c in enumerate(cells):
            m_true[k] = c["v_cell"] * float(np.einsum(
                "g,g->", w_g, c["rho_t_field"].mean(axis=1)))
        truth_mode = "rho3d histogram field"
    print(f"truth mode: {truth_mode}")

    # field-vs-exact consistency per band (quantization + noise level)
    if truth_mode.startswith("exact"):
        rel_band = []
        for j in range(n_sh):
            for bi, (bname, _, _, _) in enumerate(BANDS):
                me = ms = 0.0
                for p in range(n_phi):
                    k = j * len(BANDS) * n_phi + bi * n_phi + p
                    me += m_true[k]
                    ms += cells[k]["v_cell"] * float(np.einsum(
                        "g,g->", w_g, cells[k]["rho_t_field"].mean(axis=1)))
                rel_band.append(abs(ms / me - 1.0))
        print(f"field-integral vs exact particle mass per (shell, band): "
              f"median {100*np.median(rel_band):.2f}% max "
              f"{100*np.max(rel_band):.2f}% (cell quantization + shot noise)")

    # ---- model evaluation ---------------------------------------------------
    res = {l: {"m": np.empty(n_cell), "se": np.empty(n_cell)}
           for l in MODELS}
    for spec in args.model:
        label, run_dir = spec.split("=", 1)
        t0 = time.time()
        phi = load_phi_f32(run_dir)
        for k, c in enumerate(cells):
            q = (c["pts"] / L_KPC).reshape(-1, 3)
            rho_m = np.asarray(rho_from_phi(phi, q, L_KPC, V_KMS)).reshape(
                args.n_radial, args.n_dirs_cell)
            wm = w_g[:, None] * rho_m                     # (G, N)
            res[label]["m"][k] = c["v_cell"] * float(wm.sum(axis=0).mean())
            blocks = wm.reshape(args.n_radial, B, D).mean(axis=2)
            res[label]["se"][k] = c["v_cell"] * float(
                blocks.sum(axis=0).std(ddof=1) / np.sqrt(B))
        print(f"[{label}] eval {time.time()-t0:.0f}s")

    # ---- summaries ----------------------------------------------------------
    dm = {l: res[l]["m"] - m_true for l in MODELS}
    print("=== phi-cancellation fingerprint per (shell, band): "
          "Sigma and Sigma|.| of per-bin DeltaM/M_true [%]; model mass-"
          "normalized pair sum(dM)/M_band, sum|dM|/M_band [%] ===")
    summary_rows = []
    for j in range(n_sh):
        for bi, (bname, _, _, _) in enumerate(BANDS):
            ks = [j * len(BANDS) * n_phi + bi * n_phi + p for p in range(n_phi)]
            mt = m_true[ks]
            row = (f"  r={r_lo[j]:5.2f}-{r_hi[j]:5.2f} {bname:5s}: "
                   f"sig_e/M {100*np.sqrt(np.sum(sig_true[ks]**2))/mt.sum():5.2f}%  ")
            for l in MODELS:
                rr = dm[l][ks] / mt
                s_rel = 100 * float(np.sum(rr))
                a_rel = 100 * float(np.sum(np.abs(rr)))
                s_mass = 100 * float(np.sum(dm[l][ks]) / mt.sum())
                a_mass = 100 * float(np.sum(np.abs(dm[l][ks])) / mt.sum())
                row += f"| {l} {s_rel:+6.1f}/{a_rel:6.1f} {s_mass:+6.2f}/{a_mass:6.2f}"
                summary_rows.append(
                    dict(shell_j=j, r_lo_kpc=r_lo[j], r_hi_kpc=r_hi[j],
                         band=bname, model=l, sum_rel_signed_pct=s_rel,
                         sum_rel_abs_pct=a_rel, sum_dM_over_M_pct=s_mass,
                         sum_absdM_over_M_pct=a_mass,
                         sigma_e_over_M_pct=100 * float(
                             np.sqrt(np.sum(sig_true[ks] ** 2)) / mt.sum())))
            print(row)

    # ---- outputs ------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "phi_profiles.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["r_lo_kpc", "r_hi_kpc", "band", "phi_center_deg",
                     "M_true_msun", "sigma_e_msun"]
                    + [f"{l}_{c}" for l in MODELS for c in
                       ("M_msun", "dM_over_M_true", "SE_msun")])
        for k, c in enumerate(cells):
            row = [f"{r_lo[c['shell']]:.6f}", f"{r_hi[c['shell']]:.6f}",
                   c["band"], f"{np.degrees(phi_c[c['phi']]):.3f}",
                   f"{m_true[k]:.8e}", f"{sig_true[k]:.8e}"]
            for l in MODELS:
                row += [f"{res[l]['m'][k]:.8e}",
                        f"{dm[l][k]/m_true[k]:.6f}", f"{res[l]['se'][k]:.8e}"]
            wr.writerow(row)
    with open(args.output_dir / "phi_profiles_summary.csv", "w",
              newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    ofs.use_style()

    def profile_panels(ax_m, ax_r, ks, tag=""):
        ph = np.degrees(phi_c)
        ax_m.axhline(0.0, color="0.2", lw=0.7)
        ax_m.plot(ph, m_true[ks], color="k", lw=1.6, marker="o", ms=2.4,
                  label="truth", zorder=3)
        for l in MODELS:
            ax_m.plot(ph, res[l]["m"][ks], color=ofs.PALETTE[COLOR[l]],
                      lw=1.0, marker="o", ms=1.6, zorder=2)
        ax_m.set_xlim(0.0, 360.0)
        ax_m.set_xticks([0, 90, 180, 270, 360])
        ax_m.set_ylabel(r"$M(\phi\,{\rm bin})$  [Msun]")
        ax_m.set_xlabel(r"$\phi$ [deg]")
        j_sh = cells[ks[0]]["shell"]
        ax_m.annotate(f"r {r_lo[j_sh]:.2f}-{r_hi[j_sh]:.2f} kpc{tag}",
                      xy=(0.02, 0.05), xycoords="axes fraction",
                      fontsize=7, color="#333333")
        ax_r.axhline(0.0, color="0.2", lw=0.7)
        if np.isfinite(sig_true[ks]).all():
            ax_r.fill_between(ph, -100 * sig_true[ks] / m_true[ks],
                              100 * sig_true[ks] / m_true[ks], color="0.88",
                              zorder=0)
        for l in MODELS:
            ax_r.plot(ph, 100 * dm[l][ks] / m_true[ks],
                      color=ofs.PALETTE[COLOR[l]], lw=1.0, marker="o", ms=1.6)
        ax_r.set_xlim(0.0, 360.0)
        ax_r.set_xticks([0, 90, 180, 270, 360])
        ax_r.set_ylabel(r"$\Delta M/M_{\rm true}$  [%]")
        ax_r.set_xlabel(r"$\phi$ [deg]")
        txt = "  ".join(
            f"{l} Σ{100*np.sum(dm[l][ks]/m_true[ks]):+.0f}/"
            f"Σ|·|{100*np.sum(np.abs(dm[l][ks]/m_true[ks])):.0f}"
            for l in MODELS)
        ax_r.annotate(txt, xy=(0.5, 0.99), xycoords="axes fraction",
                      ha="center", va="top", fontsize=5.0, color="#333333")

    # paired-stack layout: every (shell, band) combo gets its mass panel
    # directly above its residual panel; combos fill two columns of pairs.
    def paired_figure(combos, stem):
        n_pairs = (len(combos) + 1) // 2
        nrows = 2 * n_pairs
        fig, axes = ofs.figure_grid(nrows, 2, width=ofs.WIDE,
                                    ratio=(0.95 * nrows + 0.5) / ofs.WIDE)
        flat = list(np.ravel(axes))
        for hide in flat[2 * len(combos):]:
            hide.set_visible(False)
        for i, (ks, tag) in enumerate(combos):
            top = flat[2 * (i // 2) + (i % 2)]
            bot = flat[2 * (i // 2 + 1) + (i % 2)]
            profile_panels(top, bot, ks, tag=tag)
            top.tick_params(labelbottom=False)
            if i % 2:
                top.tick_params(labelleft=False)
                bot.tick_params(labelleft=False)
        ofs.panel_labels(flat[:2 * len(combos)])
        for p in ofs.save(fig, str(args.output_dir / stem)):
            print("saved", p)

    eq_combos = [([j * len(BANDS) * n_phi + p for p in range(n_phi)],
                  "  (equatorial θ 45–135°)" if j == 0 else "")
                 for j in range(n_sh)]
    paired_figure(eq_combos, "phi-profiles-equator")

    mid_combos = []
    for bi in (1, 2):
        for j in mid_sel:
            ks = [j * len(BANDS) * n_phi + bi * n_phi + p
                  for p in range(n_phi)]
            mid_combos.append((ks, "  (" + BANDS[bi][3].replace(" deg", "°") + ")"))
    paired_figure(mid_combos, "phi-profiles-midlat")

    np.savez(args.output_dir / "phi_profiles.npz",
             r_lo=r_lo, r_hi=r_hi, r_c=r_c, phi_center_deg=np.degrees(phi_c),
             bands=np.asarray([b[0] for b in BANDS]), m_true=m_true,
             sigma_e=sig_true, truth_mode=truth_mode, n_phi=n_phi,
             **{f"{l}_{k}": v for l in MODELS
                for k, v in (("m", res[l]["m"]), ("se", res[l]["se"]))})
    print(f"TOTAL WALL {time.time()-t_start:.1f}s")
    print("PHI_PROFILES_DONE")


if __name__ == "__main__":
    sys.exit(main())
