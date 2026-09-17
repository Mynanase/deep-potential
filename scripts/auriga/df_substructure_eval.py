#!/usr/bin/env python
"""Substructure evaluation for the w1024-on-full-population DF: did the
retrained model express the outer clump stream?

One protocol, three models:
  w1024full  = runs/orx               (this run; halo12.h5 population)
  w128       = runs/halo12-baseline   (same halo12.h5 population)
  w1024clean = runs/halo12-cap-w1024  (halo12-clean.h5 population)

Checks (pre-registered in the node description):
  1. clump-position conditional draws (halo12-val outer, the val part of
     eval_protocol.removed_rows): vT mean/std bias per bin 4/5, clump-only
     W1(vT), and field-row vT mean bias as the no-regression check.
  2. halo12-val outer mixed-population W1(vr/vth/vT), all vs excl-clump.
  3. strict common set: weighted log p(v|x) (global + per radial bin) and
     sample-B paired W1(vr/vth/vT) per radial bin.
  4. spatial occupancy of the clump angular cell
     (r in [4.5,6), cth in [-1,-0.6), phi in [pi/2,pi]) vs the halo12.h5
     population (and halo12-clean.h5 for the clean model's own reference).

Noise protocols replicate the phase-1 audit (key 20260916 for the strict
full-coverage draw and spatial samples; key 917 for the halo12-val outer
K=4 draws), so w128/w1024clean numbers are directly comparable with the
audit artifacts; they are recomputed here and cross-checked against the
saved audit values.

Run from repo root (server):
  CUDA_VISIBLE_DEVICES=6 python scripts/auriga/df_substructure_eval.py
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
AUD = REPO / "runs" / "halo12-phase1-df-audit-20260916"
OUT = REPO / "runs" / "w1024full-substructure-eval"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import fit_all  # noqa: E402

MODELS = [("w1024full", REPO / "runs" / "orx"),
          ("w128", REPO / "runs" / "halo12-baseline"),
          ("w1024clean", REPO / "runs" / "halo12-cap-w1024")]
NAMES = [m[0] for m in MODELS]
K = 4
N_POS = 262144
CHUNK = 8192


def sph_vel(pos, vel_cart):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    vr = (x * vel_cart[:, 0] + y * vel_cart[:, 1] + z * vel_cart[:, 2]) / r
    vth = (z * vr - r * vel_cart[:, 2]) / R
    vT = (-vel_cart[:, 0] * y + vel_cart[:, 1] * x) / R
    return np.stack([vr, vth, vT], axis=1)


def wmean(a, w):
    return float(np.sum(w * a) / np.sum(w))


def wstd(a, w):
    mu = wmean(a, w)
    return float(np.sqrt(np.sum(w * (a - mu) ** 2) / np.sum(w)))


def w1(a, b, wa, wb):
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


# ------------------------------------------------------------------
# shared protocol inputs
# ------------------------------------------------------------------
prot = np.load(AUD / "eval_protocol.npz")
eta_strict = prot["eta_strict"]
w_strict = prot["w_strict"]
ir = prot["ir_strict"]
v_true_strict = np.stack([prot["vr_strict"], prot["vth_strict"], prot["vT_strict"]], axis=1)
R_EDGES = prot["r_edges"]
n_rbins = len(R_EDGES) - 1

# audit-saved reference values (cross-check for w128 / w1024clean)
audit_ref = {}
for nm, fn in [("w128", "model_w128.npz"), ("w1024clean", "model_w1024.npz")]:
    d = np.load(AUD / fn)
    audit_ref[nm] = dict(nll=d["nll_strict"], logp=float(wmean(-d["nll_strict"], w_strict)))
    print("audit reference logp(%s) = %.4f" % (nm, audit_ref[nm]["logp"]), flush=True)

# strict full-coverage base noise: identical derivation to the audit
key0 = jax.random.key(20260916)
_, key_z0_all, _, _, key_z0_pos = jax.random.split(key0, 5)
z0_all = jax.random.normal(key_z0_all, (len(eta_strict), 3))
z0_pos = jax.random.normal(key_z0_pos, (N_POS, 3))

# halo12-val outer rows + clump membership
with h5py.File(REPO / "data" / "auriga" / "halo12.h5", "r") as f:
    eta_full = f["eta"][:]
    w_full = f["weights"][:]
n_val = int(len(eta_full) * 0.25)
eta_val = eta_full[:n_val]
w_val = w_full[:n_val]
clump_val = np.zeros(n_val, dtype=bool)
clump_val[prot["removed_rows"][prot["removed_rows"] < n_val]] = True
r_val = np.linalg.norm(eta_val[:, :3], axis=1)
outer = r_val >= 4.5
pos_out = eta_val[outer, :3]
w_out = w_val[outer]
clump_out = clump_val[outer]
ir_out = (r_val[outer] >= 6.0).astype(int)
v_true_out = sph_vel(pos_out, eta_val[outer, 3:])
print("outer halo12-val stars: %d (clump=%d)" % (outer.sum(), int(clump_out.sum())), flush=True)

key917 = jax.random.key(917)
z0_out = jax.random.normal(key917, (len(pos_out), K, 3))


@eqx.filter_jit
def sample_vel_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(
        z0.reshape(-1, 3), jnp.repeat(cond, K, axis=0)).reshape(-1, K, 3)


@eqx.filter_jit
def sample_one_stage(cvf, z0, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return jax.vmap(lambda z_, c_: cvf.flow.bijection.transform(z_, condition=c_))(z0, cond)


@eqx.filter_jit
def nll_stage(cvf, v, x):
    cond = (x - cvf.cond_mean) / cvf.cond_std
    return cvf.flow.log_prob(v, condition=cond)


@eqx.filter_jit
def sample_pos_stage(sf, z0):
    return jax.vmap(lambda z: sf.flow.bijection.transform(z))(z0)


draws_out = {}       # (n_out, K, 3) spherical
strict_draws = {}    # (n_strict, 3) spherical, one draw per position
nll_strict = {}
pos_samples = {}
for nm, run_dir in MODELS:
    print("===== %s: %s =====" % (nm, run_dir), flush=True)
    flow = fit_all.load_flow(run_dir / "models" / "df" / "flow", checkpoint_index=-1)
    cvf = flow.conditional_velocity_flow
    sf = flow.spatial_flow
    key_chk = jax.random.key(11)
    z_chk = cvf.flow.base_dist.sample(key_chk, (256,)).block_until_ready()
    x_chk = jnp.asarray(eta_strict[:256, :3])
    explicit = np.asarray(sample_vel_stage(cvf, np.tile(z_chk[:, None, :], (1, K, 1)), x_chk))
    builtin = np.asarray(cvf.sample(key_chk, 256, condition=x_chk))
    dmax = float(np.abs(explicit[:, 0] - builtin).max())
    print("guard %s max|explicit-builtin| = %.2e" % (nm, dmax), flush=True)
    assert dmax < 2e-2, "sample path mismatch for " + nm

    vs = np.asarray(sample_vel_stage(cvf, z0_out, jnp.asarray(pos_out)))
    draws_out[nm] = sph_vel(np.repeat(pos_out, K, axis=0), vs.reshape(-1, 3)).reshape(len(pos_out), K, 3)

    v1 = np.empty((len(eta_strict), 3), dtype=np.float32)
    for i in range(0, len(eta_strict), CHUNK):
        sl = slice(i, min(i + CHUNK, len(eta_strict)))
        v1[sl] = np.asarray(sample_one_stage(cvf, z0_all[sl], jnp.asarray(eta_strict[sl, :3])))
    strict_draws[nm] = sph_vel(eta_strict[:, :3], v1)

    lp = np.empty(len(eta_strict), dtype=np.float32)
    for i in range(0, len(eta_strict), CHUNK):
        sl = slice(i, min(i + CHUNK, len(eta_strict)))
        lp[sl] = np.asarray(nll_stage(cvf, jnp.asarray(eta_strict[sl, 3:]), jnp.asarray(eta_strict[sl, :3])))
    nll_strict[nm] = lp
    print("logp(%s) = %.4f" % (nm, wmean(-lp, w_strict)), flush=True)

    ps = np.empty((N_POS, 3), dtype=np.float32)
    for i in range(0, N_POS, 65536):
        sl = slice(i, min(i + 65536, N_POS))
        ps[sl] = np.asarray(sample_pos_stage(sf, jnp.asarray(z0_pos[sl])))
    pos_samples[nm] = ps
    del flow, cvf, sf
    jax.clear_caches()

# consistency vs audit-saved strict draws/NLL
for nm in ("w128", "w1024clean"):
    d = np.load(AUD / ("model_w" + ("128" if nm == "w128" else "1024") + ".npz"))
    ref = sph_vel(eta_strict[:, :3], d["vel_samples_all"])
    md = float(np.abs(strict_draws[nm] - ref).max())
    dl = abs(wmean(-nll_strict[nm], w_strict) - audit_ref[nm]["logp"])
    print("consistency %s: max|draw diff| = %.2e, |dlogp| = %.2e" % (nm, md, dl), flush=True)

# ------------------------------------------------------------------
# 1. strict set: per-bin logp + W1 (sample-B paired draws)
# ------------------------------------------------------------------
print("=== 1. STRICT SET (logp and W1 per radial bin) ===", flush=True)
rows = []
for b in range(n_rbins):
    m = ir == b
    w_b = w_strict[m]
    row = dict(bin=b, r_lo=R_EDGES[b] * 10, r_hi=R_EDGES[b + 1] * 10, n=int(m.sum()))
    for nm in NAMES:
        row["logp_" + nm] = wmean(-nll_strict[nm][m], w_b)
        for ci, cn in enumerate(["vr", "vth", "vT"]):
            row["w1_%s_%s" % (cn, nm)] = w1(v_true_strict[m, ci], strict_draws[nm][m, ci], w_b, w_b)
    rows.append(row)
df_strict = pd.DataFrame(rows)
df_strict.to_csv(OUT / "strict_bins.csv", index=False)
sel = ["bin", "r_lo", "n"] + ["logp_" + nm for nm in NAMES] + ["w1_vT_" + nm for nm in NAMES]
print(df_strict[sel].round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------
# 2. halo12-val outer: clump vs field
# ------------------------------------------------------------------
print("=== 2. HALO12-VAL OUTER: CLUMP VS FIELD ===", flush=True)
rows = []
for b in (0, 1):
    m_bin = ir_out == b
    m_f = m_bin & ~clump_out
    m_c = m_bin & clump_out
    row = dict(bin=4 + b, n_field=int(m_f.sum()), n_clump=int(m_c.sum()),
               clump_vT_mean=wmean(v_true_out[m_c, 2], w_out[m_c]),
               clump_vT_std=wstd(v_true_out[m_c, 2], w_out[m_c]))
    for nm in NAMES:
        gT_c = draws_out[nm][m_c][:, :, 2].reshape(-1)
        wc = np.repeat(w_out[m_c], K)
        row["clump_bias_mean_vT_" + nm] = wmean(gT_c, wc) - row["clump_vT_mean"]
        row["clump_bias_std_vT_" + nm] = wstd(gT_c, wc) - row["clump_vT_std"]
        row["clump_w1_vT_" + nm] = w1(v_true_out[m_c, 2], gT_c, w_out[m_c], wc)
        gT_f = draws_out[nm][m_f][:, :, 2].reshape(-1)
        wf = np.repeat(w_out[m_f], K)
        row["field_bias_mean_vT_" + nm] = wmean(gT_f, wf) - wmean(v_true_out[m_f, 2], w_out[m_f])
        for ci, cn in enumerate(["vr", "vth", "vT"]):
            t_all = v_true_out[m_bin, ci]
            g_all = draws_out[nm][m_bin][:, :, ci].reshape(-1)
            row["w1_%s_%s_all" % (cn, nm)] = w1(t_all, g_all, w_out[m_bin], np.repeat(w_out[m_bin], K))
            t_f = v_true_out[m_f, ci]
            g_f = draws_out[nm][m_f][:, :, ci].reshape(-1)
            row["w1_%s_%s_exclclump" % (cn, nm)] = w1(t_f, g_f, w_out[m_f], wf)
    rows.append(row)
df_cl = pd.DataFrame(rows)
df_cl.to_csv(OUT / "outer_clump_field.csv", index=False)
sel2 = ["bin", "n_clump", "clump_vT_mean"]
for nm in NAMES:
    sel2 += ["clump_bias_mean_vT_" + nm, "clump_w1_vT_" + nm, "w1_vT_" + nm + "_all", "w1_vT_" + nm + "_exclclump"]
print(df_cl[sel2].round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------
# 3. spatial occupancy of the clump angular cell
# ------------------------------------------------------------------
print("=== 3. CLUMP ANGULAR CELL SPATIAL OCCUPANCY ===", flush=True)


def cell_mask(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(x, y)
    cth = np.divide(z, r, out=np.zeros_like(z), where=r > 1e-9)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi) - np.pi
    return (r >= 4.5) & (r < 6.0) & (cth >= -1.0) & (cth < -0.6) & (phi >= np.pi / 2) & (phi < np.pi)


pop_frac_full = float(w_full[cell_mask(eta_full)].sum() / w_full.sum())
with h5py.File(REPO / "data" / "auriga" / "halo12-clean.h5", "r") as f:
    w_clean = f["weights"][:]
    eta_clean = f["eta"][:]
pop_frac_clean = float(w_clean[cell_mask(eta_clean)].sum() / w_clean.sum())
rows = [dict(model="population(halo12.h5)", frac=pop_frac_full, reldev=0.0),
        dict(model="population(halo12-clean.h5)", frac=pop_frac_clean, reldev=pop_frac_clean / pop_frac_full - 1)]
for nm in NAMES:
    frac = float(cell_mask(pos_samples[nm]).mean())
    rows.append(dict(model=nm, frac=frac, reldev=frac / pop_frac_full - 1))
df_cell = pd.DataFrame(rows)
df_cell.to_csv(OUT / "clump_cell_occupancy.csv", index=False)
print(df_cell.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------
# figures
# ------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

m_b4 = (ir_out == 0) & clump_out
bins_h = np.linspace(-4, 4, 61)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(v_true_out[m_b4, 2], bins=bins_h, weights=w_out[m_b4], density=True,
        alpha=0.55, label="clump truth (vT=-1.98)")
for nm, ls in zip(NAMES, ["-", "--", ":"]):
    g = draws_out[nm][m_b4][:, :, 2].reshape(-1)
    ax.hist(g, bins=bins_h, weights=np.repeat(w_out[m_b4], K), density=True,
            histtype="step", lw=1.6, ls=ls, label=nm)
ax.set_xlabel("vT (100 km/s)")
ax.set_title("halo12-val 45-60 kpc, clump positions: conditional vT")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_clump_conditional_vT.png", dpi=150)
plt.close(fig)

centers = R_EDGES[:-1] * 10 + 5
fig, ax = plt.subplots(figsize=(7, 4))
for nm, mk in zip(NAMES, "osd"):
    ax.plot(centers, df_strict["logp_" + nm], mk + "-", label=nm)
ax.set_xlabel("r (kpc, bin center)")
ax.set_ylabel("log p(v|x) strict")
ax.legend()
ax.set_title("strict per-bin conditional NLL")
fig.tight_layout()
fig.savefig(OUT / "figs" / "fig_strict_logp.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# verdict vs pre-registered bars
# ------------------------------------------------------------------
glob_logp = {nm: wmean(-nll_strict[nm], w_strict) for nm in NAMES}
r4 = df_cl[df_cl["bin"] == 4].iloc[0]
summary = dict(
    strict_logp=glob_logp,
    strict_ref=dict(w128_full_baseline=-3.6191, w1024_clean=-3.5467),
    clump_bias_mean_vT={nm: float(r4["clump_bias_mean_vT_" + nm]) for nm in NAMES},
    clump_w1_vT={nm: float(r4["clump_w1_vT_" + nm]) for nm in NAMES},
    mixed_w1_vT={nm: float(r4["w1_vT_" + nm + "_all"]) for nm in NAMES},
    field_w1_vT={nm: float(r4["w1_vT_" + nm + "_exclclump"]) for nm in NAMES},
    clump_cell_occupancy=df_cell.to_dict(orient="records"),
    strict_bins=df_strict.to_dict(orient="records"),
    outer=df_cl.to_dict(orient="records"),
)
with open(OUT / "substructure_eval_summary.json", "w") as f:
    json.dump(summary, f, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

print("=== SUBSTRUCTURE VERDICT (pre-registered bars) ===", flush=True)
print("clump vT mean bias bin4: w1024full %+.4f (bar <= 0.10; w128 %+.4f; w1024clean %+.4f)"
      % (r4["clump_bias_mean_vT_w1024full"], r4["clump_bias_mean_vT_w128"],
         r4["clump_bias_mean_vT_w1024clean"]), flush=True)
print("mixed W1(vT) bin4: w1024full %.4f (bar <= 0.04; w128 %.4f)"
      % (r4["w1_vT_w1024full_all"], r4["w1_vT_w128_all"]), flush=True)
print("field W1(vT) bin4: w1024full %.4f (bar <= 0.03; w1024clean %.4f)"
      % (r4["w1_vT_w1024full_exclclump"], r4["w1_vT_w1024clean_exclclump"]), flush=True)
print("strict logp: w1024full %.4f (bar > -3.6191; w128 %.4f, w1024clean %.4f)"
      % (glob_logp["w1024full"], glob_logp["w128"], glob_logp["w1024clean"]), flush=True)
print("clump-cell spatial reldev:", {r["model"]: round(r["reldev"], 3) for r in df_cell.to_dict(orient="records")}, flush=True)
print("outputs under", OUT, flush=True)
