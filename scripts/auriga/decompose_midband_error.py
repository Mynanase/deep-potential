#!/usr/bin/env python
"""Decompose the mid-band (15-50 kpc) enclosed-mass sag into error sources,
using FROZEN models on the clean-data run.

Three radial accelerations per bin, all evaluated at the SAME DF-distributed
points (identical weighting, so differences are attributable):

  a_score : demanded by the flow score via stationarity. Per point,
            R = A - (grad phi . rhat) * C  with  A = p.dq lnF, C = dp lnF.rhat;
            zeroing R gives (grad phi . rhat) = A / C  (radial projection of a
            locally radial gradient). Per bin we take the OLS-through-origin
            slope of A on C (plus robust median of A/C over |C| above median).
  a_model : -V^2/L * (grad_q phi_model . rhat) from the frozen Phi.
  a_true  : G * M_true(<r) / r^2 from the unmodified simulation truth
            (spherical approximation of the truth profile).

Decomposition:  model-vs-true = total error;  score-vs-true = score/stationarity
share;  model-vs-score = fit/regularization share. Penalty activity is reported
per bin (fraction of points with lap < 0, penalty term vs CBE term). Capacity
sensitivity: a_score recomputed with independent w512/w1024 flows at a fixed
subsample of the same points.

Run from a repo root on the gpu host.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_enclosed_mass import G_KPC_KMS2_MSUN, load_truth  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0
ACC = V_KMS ** 2 / L_KPC  # code -> (km/s)^2 / kpc

BIN_EDGES = [2.0, 10.0, 15.0, 20.0, 30.0, 45.0, 50.0, 70.0]


def phi_grads(phi, q, batch=65536):
    import jax
    import jax.numpy as jnp
    from potential import calc_phi_derivatives
    fn = jax.vmap(calc_phi_derivatives, in_axes=(None, 0))
    g, l = [], []
    for i in range(0, len(q), batch):
        a, b = fn(phi, jnp.asarray(q[i:i + batch]))
        g.append(np.asarray(a)); l.append(np.asarray(b))
    return np.concatenate(g), np.concatenate(l)


def flow_grads(flow, eta, batch=512):
    import jax
    import flow_sampling
    out = []
    for i in range(0, len(eta), batch):
        _, g = flow_sampling.value_and_grad_lnf_fn(flow, eta[i:i + batch])
        out.append(np.asarray(g).block_until_ready() if hasattr(g, "block_until_ready") else np.asarray(g))
    return np.concatenate(out)


def bin_stats(r, A, C, a_model, a_true, mask):
    rr, AA, CC = r[mask], A[mask], C[mask]
    slope = float(np.sum(AA * CC) / np.sum(CC * CC))
    strong = np.abs(CC) > np.median(np.abs(CC))
    med = float(np.median(AA[strong] / CC[strong]))
    return dict(
        n=int(mask.sum()),
        r_mean=float(rr.mean()),
        a_score_ols=float(-ACC * slope),
        a_score_med=float(-ACC * med),
        a_model=float(a_model[mask].mean()),
        a_true=float(a_true[mask].mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="clean run dir containing runs layout (models/Phi, data/df_gradients.h5)")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--alt-flow", action="append", default=[],
                    help="label=flow_dir for capacity sensitivity (optional)")
    ap.add_argument("--alt-subsample", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/decompose"))
    args = ap.parse_args()

    import h5py
    import jax
    jax.config.update("jax_enable_x64", True)
    import fit_all

    run = Path(args.run_dir)
    with h5py.File(run / "data" / "df_gradients.h5", "r") as f:
        eta = f["eta"][:]
        dlnf = f["dlnf_deta"][:]
    print(f"loaded {len(eta)} clean training points")

    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", False)
    try:
        model = fit_all.load_potential(run / "models" / "Phi", load_history=False)
    finally:
        jax.config.update("jax_enable_x64", prev)
    phi = model.phi_model

    q = eta[:, :3]
    rhat = q / np.linalg.norm(q, axis=1, keepdims=True)
    r_kpc = np.linalg.norm(q, axis=1) * L_KPC
    A = np.sum(eta[:, 3:] * dlnf[:, :3], axis=1)
    C = np.sum(dlnf[:, 3:] * rhat, axis=1)

    print("evaluating frozen Phi gradients ...")
    g_phi, lap = phi_grads(phi, q)
    g_r = np.sum(g_phi * rhat, axis=1)
    a_model = -ACC * g_r
    R_model = A - np.sum(g_phi * dlnf[:, 3:], axis=1)

    truth = load_truth(args.truth)
    m_cum = np.r_[0.0, truth["M_cum_total"]]  # at r_edges[1:]
    r_edges = truth["r_edges"]
    M_at = lambda rr: np.interp(rr, r_edges[1:], truth["M_cum_total"])
    a_true = G_KPC_KMS2_MSUN * M_at(r_kpc) / r_kpc ** 2

    rows = []
    print()
    print(f"{'bin':>9} {'n':>7} {'a_true':>9} {'a_model':>9} {'a_score':>9} "
          f"{'mod/true%':>10} {'sco/true%':>10} {'mod/sco%':>9} {'negLap%':>8}")
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (r_kpc >= lo) & (r_kpc < hi)
        if not mask.any():
            continue
        s = bin_stats(r_kpc, A, C, a_model, a_true, mask)
        pen = np.arcsinh(np.maximum(-lap[mask], 0.0))
        s["neg_lap_pct"] = float((lap[mask] < 0).mean() * 100)
        s["pen_vs_cbe"] = float(pen.mean() / max(np.arcsinh(np.abs(R_model[mask])).mean(), 1e-12))
        s["model_vs_true_pct"] = 100 * (s["a_model"] / s["a_true"] - 1)
        s["score_vs_true_pct"] = 100 * (s["a_score_ols"] / s["a_true"] - 1)
        s["model_vs_score_pct"] = 100 * (s["a_model"] / s["a_score_ols"] - 1)
        rows.append(s)
        print(f"{lo:4.0f}-{hi:3.0f} {s['n']:7d} {s['a_true']:9.2f} {s['a_model']:9.2f} "
              f"{s['a_score_ols']:9.2f} {s['model_vs_true_pct']:10.2f} "
              f"{s['score_vs_true_pct']:10.2f} {s['model_vs_score_pct']:9.2f} "
              f"{s['neg_lap_pct']:8.1f}")

    alt = {}
    if args.alt_flow:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(eta), size=min(args.alt_subsample, len(eta)), replace=False)
        eta_sub, r_sub = eta[idx], r_kpc[idx]
        for spec in args.alt_flow:
            label, flow_dir = spec.split("=", 1)
            print(f"evaluating alt flow {label} at {len(idx)} points ...")
            flow = fit_all.load_flow(flow_dir, checkpoint_index=-1)
            d_alt = flow_grads(flow, eta_sub)
            A2 = np.sum(eta_sub[:, 3:] * d_alt[:, :3], axis=1)
            C2 = np.sum(d_alt[:, 3:] * (eta_sub[:, :3] / np.linalg.norm(eta_sub[:, :3], axis=1, keepdims=True)), axis=1)
            alt[label] = {}
            print(f"  {'bin':>9} {'a_score_ols':>11} {'vs_w128_clean%':>14}")
            base_lookup = {(round(s["r_mean"], 1)): s for s in rows}
            for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                m2 = (r_sub >= lo) & (r_sub < hi)
                if not m2.any():
                    continue
                sl = float(np.sum(A2[m2] * C2[m2]) / np.sum(C2[m2] * C2[m2]))
                a2 = -ACC * sl
                base = min(rows, key=lambda s: abs(((lo + hi) / 2) - s["r_mean"]))
                rel = 100 * (a2 / base["a_score_ols"] - 1)
                alt[label][f"{lo:g}-{hi:g}"] = dict(a_score_ols=a2, vs_main_pct=rel)
                print(f"  {lo:4.0f}-{hi:3.0f} {a2:11.2f} {rel:14.2f}")
            del flow
            import jax as _j
            _j.clear_caches()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "midband_decomposition.json", "w") as f:
        json.dump(dict(bins=rows, alt_flows=alt, acc_unit="(km/s)^2/kpc",
                       note="a_score = OLS slope of A on C per bin; a_true spherical from truth M_cum"), f, indent=2)
    print(f"saved {args.output_dir / 'midband_decomposition.json'}")
    print("DECOMPOSE_DONE")


if __name__ == "__main__":
    sys.exit(main())

