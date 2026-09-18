#!/usr/bin/env python
"""Evaluation-protocol bias diagnosis for the seed-0 data-file family.

The phase-1 audit built its strict-common held-out set as the intersection of
"first 25% of the pre-shuffled rows" of files prepared with the SAME seed 0,
and its v2 review recorded the consequence: the intersection is radially
biased (86.2% of its mass within 10 kpc vs 73% in the population) because the
two file orderings correlate strongly in the inner region (corr 0.89) and
decorrelate outward.  halo12-clean-smooth.h5 was prepared with the same seed 0,
so the frozen csmooth node requires an order-correlation check before any
cross-file evaluation protocol is reused.

This script measures, with no model and no GPU:
  1. order correlation between every pair of the three seed-0 files, per
     radial band (reproducing the v2 finding);
  2. the old strict-common pool's radial mass composition, globally and
     within each radial bin, against the clean population;
  3. the truth-side per-bin mean/dispersion difference between pool and
     population, straight from the data;
  4. the independent-seed re-shuffle protocol: held-out membership from a
     per-particle hash with a dedicated evaluation seed, independent of file
     order and of every training seed; the scatter of the truth statistics
     across three eval seeds is the protocol noise scale;
  5. a within-bin reweighting check: per-bin fine-grid importance weights
     restoring the population cell masses recover the population truth
     statistics, which validates reweighting as the bias corrector for
     model-side scores;
  6. truth half-split W1 floors for pool and population.

Outputs land in runs/halo12-eval-protocol-bias-20260918/ (CSV + JSON + md).
CPU only.  Self-test: python scripts/auriga/df_eval_protocol_bias.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get(
    "DPJAX_DATA_ROOT", "/localdisk/kosmos/my-deep-potential/data/auriga"))
FILES = {
    "full": DATA_ROOT / "halo12.h5",
    "clean": DATA_ROOT / "halo12-clean.h5",
    "csmooth": DATA_ROOT / "halo12-clean-smooth.h5",
}
OUT = REPO / "runs" / "halo12-eval-protocol-bias-20260918"

KPC = 10.0
KMS = 100.0
R_EDGES_KPC = np.array([0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0])
N_BINS = len(R_EDGES_KPC) - 1
VAL_FRAC = 0.25
EVAL_SEEDS = [20260918, 20260919, 20260920]
GRID_WIDTH_KPC = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
WEIGHT_CLIP = (0.25, 4.0)
B_SPLIT = 50
POP_FLOOR_SUBSAMPLE = 600_000
STRICT2_EXPECTED = 242_725


def sph_coords(eta):
    """Spherical frame of the audit (origin 0,0,0); code units in and out."""
    x, y, z = eta[:, 0], eta[:, 1], eta[:, 2]
    vx, vy, vz = eta[:, 3], eta[:, 4], eta[:, 5]
    r = np.linalg.norm(eta[:, :3], axis=1)
    R = np.hypot(x, y)
    vr = (x * vx + y * vy + z * vz) / r
    vth = (z * vr - r * vz) / R
    vT = (-vx * y + vy * x) / R
    return r, vr, vth, vT


def wmean(v, w):
    ok = np.isfinite(v) & np.isfinite(w)
    tot = w[ok].sum()
    return float((v[ok] * w[ok]).sum() / tot) if tot > 0 else float("nan")


def wstd(v, w):
    ok = np.isfinite(v) & np.isfinite(w)
    wt = w[ok]
    tot = wt.sum()
    if tot <= 0:
        return float("nan")
    mu = (v[ok] * wt).sum() / tot
    return float(np.sqrt((((v[ok] - mu) ** 2) * wt).sum() / tot))


def bin_of(r_kpc):
    return np.clip(np.digitize(r_kpc, R_EDGES_KPC) - 1, 0, N_BINS - 1)


def rank_spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_file(path, name):
    with h5py.File(path, "r") as f:
        if "particle_id" not in f or "eta" not in f:
            raise KeyError(f"{name}: missing particle_id/eta; has {sorted(f.keys())}")
        pid = np.asarray(f["particle_id"][:], dtype=np.int64)
        eta = np.asarray(f["eta"][:], dtype=np.float64)
        if "weights" not in f:
            raise KeyError(f"{name}: missing weights; has {sorted(f.keys())}")
        w = np.asarray(f["weights"][:], dtype=np.float64)
        shuffle_seed = f.attrs.get("shuffle_seed")
    assert np.unique(pid).size == pid.size, f"{name}: particle_id values are not unique"
    r, vr, vth, vT = sph_coords(eta)
    print(f"loaded {name}: n={len(pid)}  shuffle_seed={shuffle_seed}", flush=True)
    return dict(name=name, path=str(path), pid=pid, w=w, r_kpc=r * KPC,
                v=np.stack([vr, vth, vT], axis=1))


def in_sorted(pid_query, pid_sorted_ref):
    idx = np.clip(np.searchsorted(pid_sorted_ref, pid_query), 0,
                  len(pid_sorted_ref) - 1)
    return pid_sorted_ref[idx] == pid_query


def order_correlation_table(data):
    rows = []
    pid_sorted = {k: np.sort(d["pid"]) for k, d in data.items()}
    for a, b in [("full", "clean"), ("clean", "csmooth"), ("full", "csmooth")]:
        da, db = data[a], data[b]
        shared = np.isin(da["pid"], db["pid"])
        i_sh = np.flatnonzero(shared)
        pos_a = i_sh.astype(np.float64)
        pos_b = np.searchsorted(pid_sorted[b], da["pid"][i_sh]).astype(np.float64)
        band = bin_of(da["r_kpc"][i_sh])
        bands = [("all", np.ones(i_sh.size, dtype=bool))]
        bands += [(f"bin{k}", band == k) for k in range(N_BINS)]
        for bname, mb in bands:
            sel = np.flatnonzero(mb)
            if bname == "all":
                lo, hi = float(R_EDGES_KPC[0]), float(R_EDGES_KPC[-1])
            else:
                bb = int(bname[3:])
                lo, hi = float(R_EDGES_KPC[bb]), float(R_EDGES_KPC[bb + 1])
            rho = (rank_spearman(pos_a[sel], pos_b[sel])
                   if sel.size >= 100 else float("nan"))
            rows.append(dict(pair=f"{a}-{b}", band=bname, r_lo_kpc=lo,
                             r_hi_kpc=hi, n_shared=int(sel.size), spearman=rho))
    return pd.DataFrame(rows)


def truth_stats_table(v, w, ir, tag):
    rows = []
    for b in range(N_BINS):
        mb = ir == b
        for c, name in enumerate(("vr", "vth", "vT")):
            rows.append(dict(tag=tag, bin=b, r_lo_kpc=float(R_EDGES_KPC[b]),
                             r_hi_kpc=float(R_EDGES_KPC[b + 1]), comp=name,
                             n=int(mb.sum()),
                             mass_frac=float(w[mb].sum() / w.sum()),
                             mean_kms=wmean(v[mb, c], w[mb]) * KMS,
                             sigma_kms=wstd(v[mb, c], w[mb]) * KMS))
    return pd.DataFrame(rows)


def pid_hash01(pid, seed):
    """Order-independent held-out fraction: blake2b(pid || seed) -> [0,1)."""
    key = int(seed).to_bytes(8, "little")
    out = np.empty(len(pid), dtype=np.float64)
    blake2b = hashlib.blake2b
    for i, p in enumerate(pid.tolist()):
        out[i] = int.from_bytes(
            blake2b(int(p).to_bytes(8, "little"), key=key, digest_size=8).digest(),
            "little") / 2.0 ** 64
    return out


def reweight_table(v, w, r, v_pop, w_pop, r_pop, tag):
    """Within-bin fine-grid reweighting: pool weights -> population cell masses.

    For each radial bin the pool is re-weighted on a fine radius grid so that
    each grid cell carries the same mass as the population cell.  Cells the
    pool cannot fill (empty pool cells) are unrecoverable, so mass_recovery
    reports how much of the population bin mass the weighted pool covers.
    """
    rows = []
    ir = bin_of(r)
    ir_pop = bin_of(r_pop)
    for b in range(N_BINS):
        lo, hi = float(R_EDGES_KPC[b]), float(R_EDGES_KPC[b + 1])
        gw = float(GRID_WIDTH_KPC[b])
        ncell = max(int(np.ceil((hi - lo) / gw)), 1)
        ib = ir == b
        ibp = ir_pop == b
        cell_p = np.clip(((r[ib] - lo) / gw).astype(np.int64), 0, ncell - 1)
        cell_q = np.clip(((r_pop[ibp] - lo) / gw).astype(np.int64), 0, ncell - 1)
        m_pool = np.zeros(ncell)
        m_pop = np.zeros(ncell)
        np.add.at(m_pool, cell_p, w[ib])
        np.add.at(m_pop, cell_q, w_pop[ibp])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(m_pool > 0, m_pop / np.maximum(m_pool, 1e-300), np.inf)
        wgt = np.clip(ratio[cell_p], WEIGHT_CLIP[0], WEIGHT_CLIP[1])
        wgt = np.where(m_pop[cell_p] <= 0, 0.0, wgt)
        tot_pop = float(m_pop.sum())
        for c, name in enumerate(("vr", "vth", "vT")):
            rows.append(dict(
                tag=tag, bin=b, r_lo_kpc=lo, r_hi_kpc=hi, comp=name,
                n=int(ib.sum()),
                n_cells_covered=int(np.unique(cell_p[m_pool > 0]).size),
                n_cells_pop=int((m_pop > 0).sum()),
                pool_mean_raw=wmean(v[ib, c], w[ib]) * KMS,
                pool_sigma_raw=wstd(v[ib, c], w[ib]) * KMS,
                pool_mean_rew=wmean(v[ib, c], wgt) * KMS,
                pool_sigma_rew=wstd(v[ib, c], wgt) * KMS,
                pop_mean=wmean(v_pop[ibp, c], w_pop[ibp]) * KMS,
                pop_sigma=wstd(v_pop[ibp, c], w_pop[ibp]) * KMS,
                mean_weight=float(wgt.mean()) if wgt.size else float("nan"),
                max_weight=float(wgt.max()) if wgt.size else float("nan"),
                mass_recovery=float(wgt.sum() / tot_pop) if tot_pop > 0 else float("nan")))
    return pd.DataFrame(rows)


def halfsplit_w1_floor(v, w, reps, rng, subsample=None):
    """Truth-vs-truth W1 on random half splits: the finite-sample W1 scale."""
    n = len(w)
    if subsample and n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        v, w = v[idx], w[idx]
    out = {c: [] for c in ("vr", "vth", "vT")}
    for _ in range(reps):
        perm = rng.permutation(len(w))
        half = len(w) // 2
        i_a, i_b = perm[:half], perm[half:2 * half]
        for c, name in enumerate(("vr", "vth", "vT")):
            out[name].append(wasserstein_distance(
                v[i_a, c], v[i_b, c], u_weights=w[i_a], v_weights=w[i_b]) * KMS)
    return {c: float(np.mean(out[c])) for c in out}


def selftest():
    rng = np.random.default_rng(7)
    n = 4000
    pid = np.arange(n, dtype=np.int64) + 1000
    r_code = 0.5 + 6.5 * rng.random(n)
    v_r = rng.normal(0.0, 0.8 + 0.4 * np.exp(-r_code / 2.0))
    eta = np.zeros((n, 6))
    eta[:, 0], eta[:, 3] = r_code, v_r
    s = sph_coords(eta)
    assert np.allclose(s[1], v_r, rtol=0, atol=1e-6), "vr must recover radial velocity"

    h1 = pid_hash01(pid, 20260918)
    h2 = pid_hash01(pid, 20260918)
    h3 = pid_hash01(pid, 20260919)
    assert np.array_equal(h1, h2) and not np.array_equal(h1, h3)
    frac = float(np.mean(h1 < VAL_FRAC))
    assert abs(frac - VAL_FRAC) < 0.05, f"hash frac {frac} not ~{VAL_FRAC}"

    w = np.full(n, 1.0)
    v3 = np.stack([v_r, v_r, v_r], axis=1)
    f_small = halfsplit_w1_floor(v3, w, 10, rng)
    v_big = np.concatenate([v_r, rng.normal(0, 0.9, size=19 * n)])
    f_big = halfsplit_w1_floor(np.stack([v_big] * 3, axis=1),
                               np.full(20 * n, 1.0), 10, rng)
    assert f_big["vr"] < f_small["vr"], (f_big, f_small)

    # Reweighting selftest: a pool with the inner half of bin 0 fully kept and
    # everything else kept with prob 0.5 is radially biased; fine-grid
    # reweighting must recover the population bin-0 dispersion.
    n2 = 12000
    r_pop = rng.uniform(0.0, 75.0, n2)
    v_pop = rng.normal(0.0, 1.0, n2) * (0.5 + 2.0 * r_pop / 75.0)
    keep = rng.random(n2) < (0.5 + 0.5 * (r_pop < 5.0))
    v_pool = np.stack([v_pop[keep]] * 3, axis=1)
    rew = reweight_table(v_pool, np.ones(keep.sum()), r_pop[keep],
                         np.stack([v_pop] * 3, axis=1), np.ones(n2), r_pop, "selftest")
    row = rew[(rew["bin"] == 0) & (rew["comp"] == "vr")].iloc[0]
    raw_err = abs(row["pool_sigma_raw"] - row["pop_sigma"])
    rew_err = abs(row["pool_sigma_rew"] - row["pop_sigma"])
    assert rew_err < raw_err and rew_err < 5.0, (dict(row), raw_err, rew_err)
    assert row["mass_recovery"] > 0.8, row["mass_recovery"]
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load_file(p, k) for k, p in FILES.items()}
    clean = data["clean"]
    w_pop, r_pop, v_pop = clean["w"], clean["r_kpc"], clean["v"]
    ir_pop = bin_of(r_pop)

    # -- 1. order correlation -------------------------------------------------
    corr = order_correlation_table(data)
    corr.to_csv(OUT / "order_correlation.csv", index=False)
    print("order correlation (spearman, per band):")
    for _, r in corr[corr["band"].isin(("all", "bin0", "bin5"))].iterrows():
        print(f"  {r['pair']:>16s} {r['band']:>5s}: rho={r['spearman']:+.3f} "
              f"(n={r['n_shared']})", flush=True)

    # -- 2/3. old strict pool composition and truth-side stats ----------------
    n_rows = {k: len(d["pid"]) for k, d in data.items()}
    val_rows = {k: int(n_rows[k] * VAL_FRAC) for k in n_rows}
    val_pid = {k: np.sort(d["pid"][:val_rows[k]]) for k, d in data.items()}
    m_full = in_sorted(clean["pid"], val_pid["full"])
    m_clean = in_sorted(clean["pid"], val_pid["clean"])
    m_csm = in_sorted(clean["pid"], val_pid["csmooth"])
    strict2 = m_full & m_clean
    strict3 = m_full & m_clean & m_csm

    print(f"val rows: full={val_rows['full']} clean={val_rows['clean']} "
          f"csmooth={val_rows['csmooth']}; strict2={int(strict2.sum())} "
          f"strict3={int(strict3.sum())}", flush=True)
    assert int(strict2.sum()) == STRICT2_EXPECTED, (
        f"strict pool n={int(strict2.sum())} != {STRICT2_EXPECTED}: protocol mismatch")

    comp_rows = []
    for tag, mask in (("strict2", strict2), ("strict3", strict3)):
        f_in = float(w_pop[mask & (ir_pop == 0)].sum() / w_pop[mask].sum())
        f_pop = float(w_pop[ir_pop == 0].sum() / w_pop.sum())
        comp_rows.append(dict(tag=tag, n=int(mask.sum()), inner_mass_frac=f_in,
                              pop_inner_mass_frac=f_pop,
                              inner_overrep_ratio=f_in / f_pop))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "pool_composition.csv", index=False)
    for _, r in comp.iterrows():
        print(f"pool {r['tag']}: inner(0-10 kpc) mass frac {r['inner_mass_frac']:.3f} "
              f"vs population {r['pop_inner_mass_frac']:.3f} "
              f"(x{r['inner_overrep_ratio']:.2f})", flush=True)

    frames = {"population": truth_stats_table(v_pop, w_pop, ir_pop, "population")}
    for tag, mask in (("strict2", strict2), ("strict3", strict3)):
        frames[tag] = truth_stats_table(v_pop[mask], w_pop[mask], ir_pop[mask], tag)
    truth = pd.concat(frames.values(), ignore_index=True)
    truth.to_csv(OUT / "truth_stats.csv", index=False)
    base = frames["population"].set_index(["bin", "comp"])
    print("truth-side per-bin pool - population (km/s):")
    for tag in ("strict2", "strict3"):
        sub = frames[tag].set_index(["bin", "comp"])
        dmean = sub["mean_kms"] - base["mean_kms"]
        dsig = sub["sigma_kms"] - base["sigma_kms"]
        print(f"  {tag} bin0 dsigma: "
              + ", ".join(f"{c}={dsig.xs(0, level='bin')[c]:+.2f}"
                          for c in ("vr", "vth", "vT")))
        print(f"  {tag} all-bin dmean/dsigma summary: "
              f"max|dmean|={dmean.abs().max():.2f}, max|dsigma|={dsig.abs().max():.2f}",
              flush=True)

    # -- 4. independent pid-hash protocol ------------------------------------
    noise_frames = []
    for seed in EVAL_SEEDS:
        u = pid_hash01(clean["pid"], seed)
        hm = u < VAL_FRAC
        oc = rank_spearman(u, np.arange(len(u)))
        fr = truth_stats_table(v_pop[hm], w_pop[hm], ir_pop[hm], f"hash{seed}")
        fr["eval_seed"] = seed
        fr["spearman_vs_order"] = oc
        noise_frames.append(fr)
        print(f"hash seed {seed}: frac {hm.mean():.4f}  corr(row,hash) {oc:+.4f}",
              flush=True)
    noise = pd.concat(noise_frames, ignore_index=True)
    key = ["bin", "comp"]
    popref = base.reset_index()[["bin", "comp", "mean_kms", "sigma_kms"]].rename(
        columns={"mean_kms": "pop_mean_kms", "sigma_kms": "pop_sigma_kms"})
    noise = noise.merge(popref, on=key)
    noise["dmean_kms"] = noise["mean_kms"] - noise["pop_mean_kms"]
    noise["dsigma_kms"] = noise["sigma_kms"] - noise["pop_sigma_kms"]
    noise.to_csv(OUT / "pidhash_protocol_noise.csv", index=False)
    sd = noise.groupby(key)["dsigma_kms"].std(ddof=1)
    bias = noise.groupby(key)["dsigma_kms"].mean()
    print("protocol noise/bias (std/mean over eval seeds of dsigma), km/s:")
    print((pd.concat([bias.rename("mean_dsigma"), sd.rename("std_dsigma")],
                     axis=1)).round(2).to_string(), flush=True)

    # -- 5. within-bin reweighting recovery -----------------------------------
    rew_frames = []
    for tag, mask in (("strict2", strict2), ("strict3", strict3)):
        rew_frames.append(reweight_table(v_pop[mask], w_pop[mask], r_pop[mask],
                                         v_pop, w_pop, r_pop, tag))
    rew = pd.concat(rew_frames, ignore_index=True)
    rew.to_csv(OUT / "reweight_recovery.csv", index=False)
    print("within-bin reweighting (strict2), bin0 dsigma raw -> reweighted, km/s:")
    r2 = rew[(rew["tag"] == "strict2") & (rew["bin"] == 0)]
    for _, r in r2.iterrows():
        print(f"  {r['comp']}: raw {r['pool_sigma_raw'] - r['pop_sigma']:+.2f} -> "
              f"rew {r['pool_sigma_rew'] - r['pop_sigma']:+.2f} "
              f"(mass_recovery {r['mass_recovery']:.3f})", flush=True)

    # -- 6. W1 half-split floors ----------------------------------------------
    rng = np.random.default_rng(20260918)
    floor_rows = []
    for tag, mask, sub in (("pool_strict2", strict2, None),
                           ("population_sub600k", np.ones(len(w_pop), bool),
                            POP_FLOOR_SUBSAMPLE)):
        for b in range(N_BINS):
            mb = bin_of(r_pop[mask]) == b
            fl = halfsplit_w1_floor(v_pop[mask][mb], w_pop[mask][mb], B_SPLIT, rng,
                                    subsample=sub)
            for name in ("vr", "vth", "vT"):
                floor_rows.append(dict(tag=tag, bin=b, comp=name, w1_kms=fl[name]))
    floor = pd.DataFrame(floor_rows)
    floor.to_csv(OUT / "w1_floor.csv", index=False)
    print(floor.pivot_table(index="bin", columns=["tag", "comp"], values="w1_kms")
          .round(2).to_string(), flush=True)

    # -- report ---------------------------------------------------------------
    strict2_ratio = float(comp.loc[comp["tag"] == "strict2", "inner_overrep_ratio"].iloc[0])
    inner = corr[(corr["band"] == "bin0")].set_index("pair")["spearman"]
    max_rew_err = float((rew["pool_sigma_rew"] - rew["pop_sigma"]).abs().max())
    max_raw_err = float((rew["pool_sigma_raw"] - rew["pop_sigma"]).abs().max())
    min_rec = float(rew.groupby("bin")["mass_recovery"].min().min())
    max_noise = float(sd.max())
    b0_dsig = (frames["strict2"].set_index(["bin", "comp"])["sigma_kms"]
               - base["sigma_kms"]).xs(0, level="bin")
    lines = [
        "# Evaluation-protocol bias diagnosis (seed-0 file family)",
        "",
        f"* generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"* data root: `{DATA_ROOT}`",
        f"* files: " + ", ".join(f"`{k}` n={n_rows[k]}" for k in ("full", "clean", "csmooth")),
        "",
        "## 1. Order correlation (reproduces the v2 finding)",
        "",
        "| pair | all | bin0 (0-10) | bin5 (60-75) |",
        "|---|---|---|---|",
    ]
    for pair in ("full-clean", "clean-csmooth", "full-csmooth"):
        get = lambda band: corr[(corr["pair"] == pair) & (corr["band"] == band)]["spearman"].iloc[0]
        lines.append(f"| {pair} | {get('all'):+.3f} | {get('bin0'):+.3f} | {get('bin5'):+.3f} |")
    lines += [
        "",
        f"Inner-region order correlation stays strong across all three seed-0 "
        f"pairs (max |rho| = {inner.abs().max():.2f}), so the intersection of "
        f"row-order validation splits is radially biased for every pairing.",
        "",
        "## 2. Old strict pool composition",
        "",
        "```",
        comp.round(3).to_string(index=False),
        "```",
        "",
        f"strict2 n = {int(strict2.sum())} (expected {STRICT2_EXPECTED}, reproduced); "
        f"inner mass fraction over-represented by x{strict2_ratio:.2f} relative to "
        f"the population.",
        "",
        "## 3. Truth-side pool vs population",
        "",
        "Full per-bin table: `truth_stats.csv`.  Headline (strict2, bin0, "
        "sigma difference in km/s): "
        + ", ".join(f"{c} {b0_dsig[c]:+.2f}" for c in ("vr", "vth", "vT")) + ".",
        "",
        "## 4. Independent pid-hash protocol",
        "",
        f"Across eval seeds {EVAL_SEEDS}: per-bin truth sigma scatter (protocol "
        f"noise) max = {max_noise:.2f} km/s; per-bin mean bias max = "
        f"{float(bias.abs().max()):.2f} km/s.  Full table: `pidhash_protocol_noise.csv`.",
        "",
        "## 5. Within-bin reweighting recovery",
        "",
        f"Reweighting the pool to population cell masses reduces the max per-bin "
        f"truth dsigma error from {max_raw_err:.2f} to {max_rew_err:.2f} km/s; the "
        f"worst bin still recovers {min_rec * 100:.1f}% of the population mass.  "
        f"Full table: `reweight_recovery.csv`.",
        "",
        "## 6. Truth half-split W1 floors",
        "",
        "```",
        floor.pivot_table(index="bin", columns=["tag", "comp"], values="w1_kms")
             .round(2).to_string(),
        "```",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")

    summary = dict(files={k: str(v) for k, v in FILES.items()},
                   n={k: int(len(d["pid"])) for k, d in data.items()},
                   val_rows=val_rows, strict2_n=int(strict2.sum()),
                   strict3_n=int(strict3.sum()),
                   strict2_inner_overrep=strict2_ratio,
                   eval_seeds=EVAL_SEEDS, val_frac=VAL_FRAC, b_split=B_SPLIT,
                   protocol_noise_max_kms=max_noise,
                   reweight_max_err_raw_kms=max_raw_err,
                   reweight_max_err_rew_kms=max_rew,
                   elapsed_s=round(time.time() - t0, 1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"=== EVAL-PROTOCOL BIAS DONE in {summary['elapsed_s']}s; outputs in {OUT} ===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
