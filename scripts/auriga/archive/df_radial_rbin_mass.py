#!/usr/bin/env python
"""Radial r-bins (clump-covering), per-model own truth, UNNORMALIZED mass
histograms.

Same protocol/keys as run 6dfb9703 (own-truth node): bins [0,10,20,30,45,64,
75] kpc; w1024full/w128 vs halo12-val (key 921); w1024clean vs clean-val
(key 922). Plot change only: truth histograms use particle mass weights and
model draws carry w/K per draw, so every model curve integrates to exactly
its truth total mass. truth_h12 - truth_clean per velocity bin is then the
clump mass distribution, and full >= clean holds by construction (up to the
small val-split field difference).

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_radial_rbin_mass.py
"""

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx  # noqa: E402
import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REG = REPO / "data" / "auriga" / "clump_pid_registry.npz"
OUT = REPO / "runs" / "radial-rbin-mass-20260918"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = [("w1024full", REPO / "runs" / "w1024full", "h12"),
          ("w128", REPO / "runs" / "halo12-baseline", "h12"),
          ("w1024clean", REPO / "runs" / "halo12-cap-w1024", "clean")]
NAMES = [m[0] for m in MODELS]
SHORT = {"w1024full": "full", "w128": "128", "w1024clean": "clean"}
K = 4
FLAT_CHUNK = 8192
COMP = ["vr", "vth", "vT"]
R_EDGES = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 6.4, 7.5])


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def w1(a, b, wa, wb):
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


truth = {}
for tag, fn in [("h12", "halo12.h5"), ("clean", "halo12-clean.h5")]:
    with h5py.File(REPO / "data" / "auriga" / fn, "r") as f:
        n_tot = f["eta"].shape[0]
        nv = int(n_tot * 0.25)
        eta_t = f["eta"][:nv]
        w_t = f["weights"][:nv]
    pos_t = eta_t[:, :3]
    r_t = np.linalg.norm(pos_t, axis=1)
    truth[tag] = dict(eta=eta_t, w=w_t, pos=pos_t, r=r_t,
                      ir=np.clip(np.digitize(r_t, R_EDGES) - 1, 0, len(R_EDGES) - 2),
                      v=sph_vel(pos_t, eta_t[:, 3:]), n=nv)
    print("truth %s: n=%d" % (tag, nv), flush=True)

det = np.asarray(np.load(REG)["detection_pids"]).astype(np.uint64)
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    pid_f = f["particle_id"][:int(f["eta"].shape[0] * 0.25)]
mem_h12 = np.isin(pid_f, det)

# per-bin truth total masses (mass units: mean particle mass = 1)
for b in range(len(R_EDGES) - 1):
    mh = truth["h12"]["ir"] == b
    mc = truth["clean"]["ir"] == b
    print("bin %g-%g kpc: mass h12=%.1f clean=%.1f clump=%.1f (%.1f%%)"
          % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10,
             truth["h12"]["w"][mh].sum(), truth["clean"]["w"][mc].sum(),
             truth["h12"]["w"][mh & mem_h12].sum(),
             100 * truth["h12"]["w"][mh & mem_h12].sum() / truth["h12"]["w"][mh].sum()),
          flush=True)

streams = {"h12": jax.random.normal(jax.random.key(921), (truth["h12"]["n"], K, 3)),
           "clean": jax.random.normal(jax.random.key(922), (truth["clean"]["n"], K, 3))}


@eqx.filter_jit
def sample_flat(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(z0, cond)


draws = {}
truth_of = {}
for nm, run_dir, tag in MODELS:
    print("===== %s (truth=%s) =====" % (nm, tag), flush=True)
    truth_of[nm] = tag
    tr = truth[tag]
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(tr["pos"][:256])
    explicit = np.asarray(sample_flat(cvf, np.tile(z_chk[:, None, :], (1, K, 1)).reshape(-1, 3),
                                       jnp.repeat(x_chk, K, axis=0))).reshape(256, K, 3)
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2
    flat = tr["n"] * K
    zflat = streams[tag].reshape(flat, 3)
    out = np.empty((flat, 3), dtype=np.float32)
    for i in range(0, flat, FLAT_CHUNK):
        rows = np.arange(i, min(i + FLAT_CHUNK, flat))
        star = rows // K
        out[rows] = np.asarray(sample_flat(cvf, jnp.asarray(zflat[rows]),
                                           jnp.asarray(tr["pos"][star])))
    draws[nm] = sph_vel(np.repeat(tr["pos"], K, axis=0), out).reshape(tr["n"], K, 3)
    del flow, cvf
    jax.clear_caches()

rows = []


def panel(ax, b_lo, b_hi, clump_frac=None):
    bins = np.linspace(-4.5, 4.5, 73)
    th = truth["h12"]
    tc = truth["clean"]
    mh = (th["r"] >= b_lo) & (th["r"] < b_hi)
    mc = (tc["r"] >= b_lo) & (tc["r"] < b_hi)
    ax.hist(th["v"][mh, ci], bins=bins, weights=th["w"][mh],
            alpha=0.4, color="C0", label="truth halo12-val")
    ax.hist(tc["v"][mc, ci], bins=bins, weights=tc["w"][mc],
            histtype="step", lw=1.0, color="C7", label="truth clean-val")
    w1p = {}
    styles = {"w1024full": ("C1", "-", 1.3), "w128": ("C2", "--", 1.3),
              "w1024clean": ("C3", ":", 2.0)}
    for nm in NAMES:
        tag = truth_of[nm]
        tr = truth[tag]
        m = (tr["r"] >= b_lo) & (tr["r"] < b_hi)
        d = draws[nm][m][:, :, ci].reshape(-1)
        wd_full = np.repeat(tr["w"][m], K)
        w1p[nm] = w1(tr["v"][m, ci], d, tr["w"][m], wd_full)
        c, ls, lw = styles[nm]
        ax.hist(d, bins=bins, weights=wd_full / K, histtype="step",
                lw=lw, color=c, ls=ls, label=nm)
        rows.append(dict(bin_lo=b_lo * 10, bin_hi=b_hi * 10, comp=COMP[ci], model=nm,
                         truth=tag, w1=w1p[nm], n_true=int(m.sum()),
                         truth_mass=float(tr["w"][m].sum()),
                         clump_mass_frac_h12=clump_frac if clump_frac is not None else ""))
    ax.set_xlim(-4.5, 4.5)
    ax.tick_params(labelsize=7)
    txt = "W1 own-truth: " + " ".join("%s %.3f" % (SHORT[n], w1p[n]) for n in NAMES)
    if clump_frac is not None:
        txt = "clump %.0f%% of h12 mass\n" % (100 * clump_frac) + txt
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.0, va="top")


fig, axes = plt.subplots(3, len(R_EDGES) - 1, figsize=(19, 7.8), sharex=True)
for ci in range(3):
    for b in range(len(R_EDGES) - 1):
        ax = axes[ci, b]
        cf = None
        if b == 4:
            mh = truth["h12"]["ir"] == b
            cf = float(truth["h12"]["w"][mh & mem_h12].sum() / truth["h12"]["w"][mh].sum())
        panel(ax, R_EDGES[b], R_EDGES[b + 1], clump_frac=cf)
        ax.set_title("%g-%g kpc" % (R_EDGES[b] * 10, R_EDGES[b + 1] * 10), fontsize=9)
        if b == 0:
            ax.set_ylabel(COMP[ci] + "\nmass/bin", fontsize=9)
axes[0, 0].legend(fontsize=6.5, loc="upper right")
fig.suptitle("UNNORMALIZED mass histograms; each model vs its OWN truth (bins cover the clump)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_rbin_mass_full.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for ci in range(3):
    for b, (lo, hi) in enumerate([(4.5, 6.4), (6.4, 7.5)]):
        ax = axes[ci, b]
        cf = None
        if lo == 4.5:
            mh = (truth["h12"]["r"] >= lo) & (truth["h12"]["r"] < hi)
            cf = float(truth["h12"]["w"][mh & mem_h12].sum() / truth["h12"]["w"][mh].sum())
        panel(ax, lo, hi, clump_frac=cf)
        ax.set_title("outer %g-%g kpc" % (lo * 10, hi * 10), fontsize=9)
        if b == 0:
            ax.set_ylabel(COMP[ci] + "\nmass/bin", fontsize=9)
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("outer bins, mass histograms: truth_h12 - truth_clean = clump mass", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "figs" / "fig_radial_rbin_mass_outer.png", dpi=150)
plt.close(fig)

df = pd.DataFrame(rows).drop_duplicates(subset=["bin_lo", "bin_hi", "comp", "model", "truth"])
df.to_csv(OUT / "radial_rbin_mass_w1.csv", index=False)
print("=== W1(vT) per bin, own truth (identical to density version) ===", flush=True)
print(df[df["comp"] == "vT"].pivot(index="bin_lo", columns="model", values="w1").round(4).to_string(), flush=True)
with open(OUT / "summary.json", "w") as f:
    json.dump({"w1": df.to_dict(orient="records"),
               "edges_kpc": (R_EDGES * 10).tolist(),
               "truth_of": {nm: truth_of[nm] for nm in NAMES},
               "hist": "unnormalized mass; model draws weighted w/K"},
              f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("outputs under", OUT, flush=True)
