#!/usr/bin/env python
"""Radial r-bin marginal distributions: normalizing-flow model vs truth.

Generic counterpart of the radial-marginals family (keep/df_radial_rbin*,
keep/df_radial_velocity_marginals), reduced to the two data kinds that
matter routinely: the trained phase-space flow and the particle data it
models.

  radial   fine r-histogram (mass or number weighting) of spatial-flow
           samples vs the particle data, full training domain, plus the
           per-bin mass-fraction comparison bar panel.
  velocity three spherical velocity components per r-bin: conditional
           draws of the flow at the truth positions (K per position, the
           rbin-family protocol incl. the explicit-vs-builtin sampling
           guard) vs the truth histograms; per-bin W1 annotated, summary
           table persisted (csv + json).

Truth tables come from auriga/truth_products.radial_bin_table (build once
with `truth_products.py build-radial-hists`, then --cache reuses the h5).

Data-level conventions follow the rbin family: code-unit velocities, r in
kpc (length_scale from the --input attrs), unnormalized mass histograms so
curves integrate to their own total mass; figures to --fig-dir (png).

Run from the repo root (GPU recommended for the flow draws):
  python scripts/plot_radial_marginals.py --input data/auriga/halo12.h5 \
      --model S1=runs/halo12-cap-w1024 --edges 0,10,20,30,45,64,75 \
      --weighting mass --fig-dir runs/radial-marginals/figs
"""
import argparse
import json
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

import orx_figstyle as ofs  # noqa: E402
from truth_products import radial_bin_table  # noqa: E402

COMP = ("vr", "vth", "vT")
FLAT_CHUNK = 8192


def w1(a, b, wa, wb):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def load_flow_marginal(run_dir):
    import fit_all
    return fit_all.load_flow(Path(run_dir) / "models" / "df" / "flow",
                             checkpoint_index=-1)


def sample_conditional(cvf, z0, x):
    """Explicit flatten-draw of the conditional velocity flow (rbin protocol)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    @eqx.filter_jit
    def sample_flat(cvf_, z0_, x_):
        cond = (x_ - cvf_.cond_mean) / cvf_.cond_std
        return jax.vmap(
            lambda z_, c_: cvf_.flow.bijection.transform(z_, condition=c_))(
            z0_, cond)

    out = np.empty((z0.shape[0], 3), dtype=np.float32)
    for i in range(0, z0.shape[0], FLAT_CHUNK):
        rows = range(i, min(i + FLAT_CHUNK, z0.shape[0]))
        out[i:i + FLAT_CHUNK] = np.asarray(
            sample_flat(cvf, jnp.asarray(z0[list(rows)]),
                        jnp.asarray(x[np.asarray(rows)])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="particle eta h5 (truth side; attrs carry units)")
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=RUN_DIR", help="flow run dir; repeatable")
    ap.add_argument("--edges", type=str, default="0,10,20,30,45,64,75",
                    help="comma-separated r-bin outer edges [kpc]")
    ap.add_argument("--weighting", choices=("mass", "number"), default="mass")
    ap.add_argument("--split", choices=("val", "all"), default="val",
                    help="truth split: val = first quarter (rbin protocol)")
    ap.add_argument("--mode", choices=("radial", "velocity", "both"),
                    default="both")
    ap.add_argument("--k-draws", type=int, default=4,
                    help="conditional draws per truth position (velocity mode)")
    ap.add_argument("--draw-seed", type=int, default=921)
    ap.add_argument("--n-spatial", type=int, default=1_000_000,
                    help="spatial-flow sample count (radial mode)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="truth_products radial-hists h5; built if missing")
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--fig-fmt", default="png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax

    t_start = time.time()
    r_edges = np.asarray([float(v) for v in args.edges.split(",") if v != ""],
                         dtype=float)
    n_bins = r_edges.size - 1

    # ---------------- truth side (cached table or inline) ----------------
    if args.cache is not None and args.cache.is_file():
        with h5py.File(args.cache, "r") as f:
            assert f.attrs["weighting"] == args.weighting, "cache weighting mismatch"
            assert f.attrs["split"] == args.split, "cache split mismatch"
            assert np.allclose(f["r_edges"][:], r_edges), "cache edges mismatch"
            L = float(f.attrs["length_scale_kpc"])
            vel_edges = np.asarray(f["vel_bin_edges"][:])
            bins = []
            for b in range(n_bins):
                g = f[f"bin{b:02d}"]
                bins.append({"count": int(g["count"][()]),
                             "mass": float(g["mass"][()]),
                             "r_hist": np.asarray(g["r_hist"][:]),
                             "r_sub_edges": np.asarray(g["r_sub_edges"][:]),
                             "v_hists": np.asarray(g["v_hists"][:])})
        print(f"truth tables from cache {args.cache}")
    else:
        tab = radial_bin_table(args.input, r_edges, args.weighting, args.split)
        L = tab["L_kpc"]
        vel_edges = tab["vel_bin_edges"]
        bins = tab["bins"]
        print(f"truth tables computed inline from {args.input}")

    with h5py.File(args.input, "r") as f:
        n_tot = f["eta"].shape[0]
        if args.split == "val":
            n_val = int(n_tot * 0.25)
            eta_t = np.asarray(f["eta"][:n_val], dtype=np.float64)
            w_t = np.asarray(f["weights"][:n_val], dtype=np.float64)
        else:
            eta_t = np.asarray(f["eta"][:], dtype=np.float64)
            w_t = np.asarray(f["weights"][:], dtype=np.float64)
    pos_t = eta_t[:, :3]
    r_t = np.linalg.norm(pos_t, axis=1) * L
    v_t = sph_vel(pos_t, eta_t[:, 3:])
    weights_t = w_t if args.weighting == "mass" else np.ones_like(w_t)
    idx_t = np.clip(np.digitize(r_t, r_edges) - 1, 0, n_bins - 1)
    inside_t = (r_t >= r_edges[0]) & (r_t < r_edges[-1])

    models = []
    for spec in args.model:
        label, _, run_dir = spec.partition("=")
        models.append((label.strip(), Path(run_dir.strip())))

    # fine radial histogram edges (shared, log inside the domain)
    r_fine = np.geomspace(max(r_edges[0], 1e-2), r_edges[-1], 384)
    r_centres = 0.5 * (r_fine[1:] + r_fine[:-1])

    rows_summary = []

    def save(fig, stem):
        for fmt in args.fig_fmt.split(","):
            path = args.fig_dir / f"{stem}.{fmt.strip()}"
            fig.savefig(path, dpi=200 if fmt.strip() == "png" else None)
            print("saved", path)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    ofs.use_style()

    # ================= mode: radial =====================================
    if args.mode in ("radial", "both"):
        print("=== radial mode: spatial-flow samples vs particle r ===")
        h_true, _ = np.histogram(r_t, bins=r_fine, weights=weights_t)
        fig, axes = ofs.figure_grid(2, 1, width=ofs.TEXT, ratio=0.62)
        ax, ax2 = np.asarray(axes).ravel()
        ax.loglog(r_centres, h_true, color="0.2", lw=1.6, label="truth (particles)")
        model_r = {}
        for j, (label, run_dir) in enumerate(models):
            flow = load_flow_marginal(run_dir)
            key = jax.random.key(args.draw_seed + j)
            xs = np.empty((args.n_spatial, 3), dtype=np.float32)
            CH = 200_000
            for i in range(0, args.n_spatial, CH):
                n_ = min(CH, args.n_spatial - i)
                xs[i:i + n_] = np.asarray(
                    flow.sample_position(jax.random.fold_in(key, i), n_))
            r_s = np.linalg.norm(xs, axis=1) * L
            keep = (r_s >= r_edges[0]) & (r_s < r_edges[-1])
            h_s, _ = np.histogram(r_s[keep], bins=r_fine)
            # scale sample counts to the truth total mass in range
            scale = weights_t[inside_t].sum() / max(keep.sum(), 1)
            h_s = h_s * scale
            model_r[label] = (r_s, keep, scale)
            ax.loglog(r_centres, h_s, color=ofs.CYCLE[j % len(ofs.CYCLE)],
                      lw=1.2, label=label)
            w1_full = w1(r_t[inside_t], r_s[keep],
                         weights_t[inside_t], np.full(keep.sum(), scale))
            rows_summary.append(dict(mode="radial", model=label,
                                     w1_full_r=w1_full,
                                     n_draw=int(keep.sum())))
            print(f"[{label}] W1(r, full domain) = {w1_full:.4f} kpc")
            del flow
            jax.clear_caches()
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("mass per log bin" if args.weighting == "mass"
                      else "count per log bin")
        ax.legend(frameon=False, fontsize=7)

        width = 0.8 / (len(models) + 1)
        centres = 0.5 * (r_edges[:-1] + r_edges[1:])
        mass_true = np.array([b["mass"] for b in bins])
        ax2.bar(centres - width * len(models) / 2, mass_true, width=width,
                color="0.6", label="truth")
        for j, (label, _) in enumerate(models):
            r_s, keep, scale = model_r[label]
            m_s = np.histogram(r_s[keep], bins=r_edges)[0] * scale
            ax2.bar(centres + width * (j - len(models) / 2 + 0.5), m_s,
                    width=width, color=ofs.CYCLE[j % len(ofs.CYCLE)],
                    label=label, alpha=0.85)
        ax2.set_xlabel("r [kpc]")
        ax2.set_ylabel("bin mass fraction" if args.weighting == "mass"
                       else "bin count fraction")
        tot = mass_true.sum()
        ax2.set_ylim(0, max(mass_true.max(), 1.0) / tot * tot * 1.15)
        ax2.legend(frameon=False, fontsize=7)
        ofs.panel_labels([ax, ax2])
        save(fig, "radial_marginals_r")

    # ================= mode: velocity ===================================
    if args.mode in ("velocity", "both"):
        print("=== velocity mode: conditional draws at truth positions ===")
        draws = {}
        for j, (label, run_dir) in enumerate(models):
            flow = load_flow_marginal(run_dir)
            cvf = flow.conditional_velocity_flow
            # sampling guard (rbin protocol): explicit flatten vs builtin
            n_chk = min(256, pos_t.shape[0])
            key_chk = jax.random.key(11)
            z_chk = cvf.flow.base_dist.sample(key_chk, (n_chk,))
            x_chk = jax.numpy.asarray(pos_t[:n_chk])
            explicit = sample_conditional(
                cvf, np.tile(z_chk[:, None, :], (1, args.k_draws, 1)).reshape(-1, 3),
                np.repeat(pos_t[:n_chk], args.k_draws, axis=0)).reshape(n_chk, args.k_draws, 3)
            builtin = np.asarray(cvf.sample(key_chk, n_chk, condition=x_chk))
            dmax = float(np.abs(explicit[:, 0] - builtin).max())
            print(f"guard {label} max|explicit-builtin| = {dmax:.2e}")
            assert dmax < 2e-2, f"sampling protocol mismatch for {label}"

            K = args.k_draws
            n = pos_t.shape[0]
            rng = np.random.default_rng(args.draw_seed + 100 + j)
            zflat = rng.standard_normal((n * K, 3))
            order = np.repeat(np.arange(n), K)
            out = sample_conditional(cvf, zflat, pos_t[order])
            draws[label] = sph_vel(np.repeat(pos_t, K, axis=0), out).reshape(n, K, 3)
            del flow, cvf
            jax.clear_caches()

        fig, axes = plt.subplots(3, n_bins,
                                 figsize=(3.1 * n_bins, 7.8), sharex=True)
        axes = np.atleast_2d(axes)
        for ci in range(3):
            for b in range(n_bins):
                ax = axes[ci, b]
                m_t = inside_t & (idx_t == b)
                ax.hist(v_t[m_t, ci], bins=vel_edges, weights=weights_t[m_t],
                        alpha=0.4, color="C0", label="truth" if (ci == 0 and b == 0) else None)
                txt = []
                for j, (label, _) in enumerate(models):
                    wd = np.repeat(weights_t[m_t], K)
                    d = draws[label][m_t][:, :, ci].reshape(-1)
                    rows_summary.append(
                        dict(mode="velocity", model=label,
                             bin_lo=r_edges[b], bin_hi=r_edges[b + 1],
                             comp=COMP[ci],
                             w1=w1(v_t[m_t, ci], d, weights_t[m_t], wd),
                             n_true=int(m_t.sum())))
                    txt.append(f"{label} W1 "
                               f"{rows_summary[-1]['w1']:.3f}")
                    ax.hist(d, bins=vel_edges, weights=wd / K, histtype="step",
                            lw=1.2, color=ofs.CYCLE[j % len(ofs.CYCLE)],
                            label=label if (ci == 0 and b == 0) else None)
                ax.set_xlim(vel_edges[0], vel_edges[-1])
                ax.tick_params(labelsize=7)
                ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes,
                        fontsize=6.0, va="top")
                if ci == 0:
                    ax.set_title(f"{r_edges[b]:g}-{r_edges[b+1]:g} kpc",
                                 fontsize=9)
                if b == 0:
                    ax.set_ylabel(f"{COMP[ci]}\n"
                                  f"{'mass/bin' if args.weighting == 'mass' else 'count/bin'}",
                                  fontsize=9)
        axes[0, 0].legend(fontsize=6.5, loc="upper right")
        fig.tight_layout()
        save(fig, "radial_marginals_v")

    # summary persistence (rbin-family convention)
    import pandas as pd
    df = pd.DataFrame(rows_summary).drop_duplicates()
    df.to_csv(args.fig_dir / "radial_marginals.csv", index=False)
    with open(args.fig_dir / "summary.json", "w") as f:
        json.dump({"w1": df.to_dict(orient="records"),
                   "edges_kpc": r_edges.tolist(),
                   "weighting": args.weighting, "split": args.split,
                   "k_draws": args.k_draws,
                   "hist": "unnormalized mass; draws weighted w/K"},
                  f, indent=1)
    print(f"PLOT_RADIAL_MARGINALS_DONE total {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
