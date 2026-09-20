#!/usr/bin/env bash
# Grid-decoupled negative-density prior on the S1 route (w1024, clean+smooth).
# Identical to the S1 contract except the Phi loss: the CBE term keeps the S1
# mass-weighted estimator log(sum w*cbe_i / sum w) on phase-space samples, and
# the negative-density penalty moves OUT of the weighted likelihood onto an
# independent, volume-weighted spatial grid:
#     loss = log(sum w*cbe_i/sum w) + lambda * mean_grid(prior_neg)
# q_grid is a new batch channel in get_phi_loss; train_potential resamples
# uniform-ball grid points (radius prior_grid_q_max) every epoch. Flow training
# is unchanged and must reproduce the fb85670e reference; the Phi val loss is a
# new estimand again and is reported without reference.
set -eo pipefail

PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
SRC=/localdisk/kosmos/my-deep-potential
OLD=$SRC/data/auriga/halo12_all_mass_clean_outer_clump_smooth.h5
NEW=$SRC/data/auriga/halo12-clean-smooth.h5
REG=$SRC/data/auriga/clump_pid_registry.npz
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"

echo '=== W1024-CSMOOTH-S1-GRIDPRIOR PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$OLD" || { echo "PREFLIGHT FAIL: missing $OLD"; exit 2; }
test -f "$REG" || { echo "PREFLIGHT FAIL: missing $REG"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py, scipy' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import h5py; f=h5py.File("'"$OLD"'","r"); a=dict(f.attrs); print("source attrs:", {k:a[k] for k in ("cleaning_registry_set","cleaning_removed_count","length_unit","velocity_unit") if k in a}, "n:", f["eta"].shape[0])'
"$PY" - <<'PYEOF'
import json
opts = json.load(open("scripts/auriga/options.json"))
fs = opts["flow_sampling"]
ra = fs.get("radial_alloc")
lo = opts["Phi"]["loss_opts"]
assert ra is not None, "S1 options must contain flow_sampling.radial_alloc"
assert len(ra["fractions"]) == len(ra["bin_edges"]) - 1
assert abs(sum(ra["fractions"]) - 1.0) < 1e-9
assert int(lo.get("prior_grid_n", 0)) > 0, "grid-prior options must set prior_grid_n > 0"
assert float(lo.get("lambda_", 0.0)) != 0.0, "grid-prior run needs lambda_ != 0"
print("S1 radial_alloc:", ra)
print("Phi loss_opts:", lo)
print("S1 flow_sampling keys:", sorted(fs.keys()))
PYEOF

echo '=== STEP 0: grid-prior loss unit tests ==='
"$PY" -m pytest tests/test_grid_prior_loss.py -q

mkdir -p runs data/auriga
ln -sfn "$REG" data/auriga/clump_pid_registry.npz
if [ -f "$NEW" ]; then
  echo '=== STEP 1: converted file exists - verify lineage and reuse ==='
  "$PY" - <<'PYEOF'
import h5py, numpy as np, sys
NEW = "/localdisk/kosmos/my-deep-potential/data/auriga/halo12-clean-smooth.h5"
REG = "/localdisk/kosmos/my-deep-potential/data/auriga/clump_pid_registry.npz"
reg = np.load(REG)
union = np.unique(np.concatenate([np.asarray(reg[k]).astype(np.uint64)
                                  for k in ("detection_pids", "smooth65_pids", "smooth84_pids")]))
with h5py.File(NEW, "r") as f:
    n = f["eta"].shape[0]
    pid = f["particle_id"][:]
    a = dict(f.attrs)
    a.update(dict(f["eta"].attrs))
overlap = int(np.isin(pid, union).sum())
ok = (n == 1619615 and int(a.get("shuffle_seed", -1)) == 0
      and a.get("weighting") == "mass" and overlap == 0)
print("verify:", dict(n=n, shuffle_seed=int(a.get("shuffle_seed", -1)),
                      weighting=a.get("weighting"), registry_overlap=overlap,
                      r_in=float(a.get("r_in", -1)), r_out=float(a.get("r_out", -1))))
sys.exit(0 if ok else 2)
PYEOF
  test $? -eq 0 || { echo "PREFLIGHT FAIL: existing $NEW failed lineage verification"; exit 2; }
else
  echo '=== STEP 1: convert to new format (seed 0, mass weighting) ==='
  "$PY" scripts/auriga/prepare_data.py --input "$OLD" --output "$NEW" --seed 0 --weighting mass
  test -f "$NEW" || { echo "FAIL: conversion produced no output"; exit 2; }
fi
"$PY" -c 'import h5py; f=h5py.File("'"$NEW"'","r"); a=dict(f.attrs); a.update(dict(f["eta"].attrs)); print("NEW attrs:", {k:a[k] for k in ("r_in","r_out","train_r_max_kpc","weighting","shuffle_seed") if k in a}, "n:", f["eta"].shape[0])'

echo '=== STEP 2: fixed three-stage w1024 pipeline (S1 sampling + grid-decoupled prior) ==='
DATA_ROOT=$SRC/data/auriga
DATA=$(cat scripts/auriga/orx_data_path.txt 2>/dev/null || echo "$DATA_ROOT/halo12-clean-smooth.h5")
case "$DATA" in /*) ;; *) DATA="$DATA_ROOT/$DATA";; esac
test -f "$DATA" || { echo "PREFLIGHT FAIL: missing $DATA"; exit 2; }
mkdir -p runs/orx
cp scripts/auriga/options.json runs/orx/options.json
run() { "$PY" -u scripts/fit_all.py --input "$DATA" --run-dir runs/orx "$@"; }
run --flow-training && run --flow-sampling && run --potential-training

echo '=== FINAL LOSSES ==='
for f in runs/orx/models/df/flow/flow-*_loss.json runs/orx/models/Phi/potential-*_loss.json; do
  echo "-- $f"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: (v[-1] if isinstance(v, list) else v) for k, v in d.items()})' "$f"
done

echo '=== S1 STRATIFICATION EVIDENCE ==='
"$PY" - <<'PYEOF'
import h5py, numpy as np, sys, json
fs_opts = json.load(open("runs/orx/options.json"))["flow_sampling"]
n_req = int(fs_opts["n_samples"])
expected_n = int(sum(max(1, round(float(f) * n_req)) for f in fs_opts["radial_alloc"]["fractions"]))
with h5py.File("runs/orx/data/df_gradients.h5", "r") as f:
    keys = sorted(f.keys())
    w = f["importance_weights"][:]
    edges = f["radial_bin_edges"][:]
    masses = f["radial_bin_masses"][:]
    counts = f["radial_bin_counts"][:]
print("df_gradients keys:", keys)
print("radial bin edges (q):", edges.tolist())
print("radial quotas: counts={} pool_masses={}".format(counts.astype(int).tolist(), np.round(masses, 4).tolist()))
print("weights: n={} mean={:.6f} range=[{:.4f},{:.4f}]".format(len(w), w.mean(), w.min(), w.max()))
print("expected quota total (rounded per-bin):", expected_n)
ok = ("importance_weights" in keys and len(w) == expected_n and int(counts.sum()) == expected_n and abs(w.mean() - 1.0) < 1e-3)
print("S1 WEIGHTS CHECK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 4)
PYEOF
test $? -eq 0 || { echo "S1 WEIGHTS CHECK FAILED"; exit 4; }

echo '=== GRID-PRIOR TRAINING EVIDENCE ==='
"$PY" - <<'PYEOF'
import glob, json, re, sys
files = sorted(glob.glob("runs/orx/models/Phi/potential-*_loss.json"),
               key=lambda f: int(re.search(r"potential-(\d+)_loss", f).group(1)))
p = json.load(open(files[-1]))
hist = p.get("prior_neg_grid", [])
assert len(hist) == len(p["train"]), "prior_neg_grid history must cover every epoch"
print("prior_neg_grid per-epoch mean penalty: first={:.4f} mid={:.4f} final={:.4f} min={:.4f}".format(
    hist[0], hist[len(hist)//2], hist[-1], min(hist)))
print("phi_val (decoupled estimand, reported without reference): {:.4f}".format(p["val"][-1]))
sys.exit(0)
PYEOF
test $? -eq 0 || { echo "GRID-PRIOR TRAINING EVIDENCE FAILED"; exit 5; }

echo '=== REPRODUCTION CHECK (flow stage vs fb85670e; Phi is a new estimand) ==='
"$PY" - <<'PYEOF'
import json, sys
ref = dict(val_pos=3.7245, val_vel=3.4077)
tol = 1e-2
f = json.load(open("runs/orx/models/df/flow/flow-21_loss.json"))
got_flow = dict(val_pos=f["val_pos"][-1], val_vel=f["val_vel"][-1])
ok = True
for k in ref:
    rel = abs(got_flow[k] - ref[k]) / abs(ref[k])
    ok &= rel < tol
    print("%s: got %.4f ref %.4f rel %.2e %s" % (k, got_flow[k], ref[k], rel, "PASS" if rel < tol else "FAIL"))
sys.exit(0 if ok else 3)
PYEOF
test $? -eq 0 || { echo "FLOW REPRODUCTION CHECK FAILED"; exit 3; }

echo '=== GRID-PRIOR NEGATIVE-DENSITY EVIDENCE (fixed Sobol grid, 30-70 kpc) ==='
"$PY" - <<'PYEOF'
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import jax
jax.config.update("jax_enable_x64", False)
import fit_all
phi_model = fit_all.load_potential(Path("runs/orx/models/Phi")).phi_model
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import potential as pmod
from scipy.stats import qmc

L, V, G = 10.0, 100.0, 4.30091e-6
def lap_batch(q):
    # Closure + bare vmap (the pattern proven by validate_enclosed_mass on
    # these checkpoints): the model is captured as a tracing-time constant,
    # so non-array leaves such as net.activation never become jit args.
    def lap_one(x):
        return pmod.calc_phi_laplacian(phi_model, x)
    return jax.vmap(lap_one)(jnp.asarray(q))

sob = qmc.Sobol(d=2, scramble=True, seed=0)
uv = sob.random(2048)
mu = 2.0 * uv[:, 0] - 1.0
az = 2.0 * np.pi * uv[:, 1]
s = np.sqrt(np.maximum(0.0, 1.0 - mu ** 2))
dirs = np.column_stack([s * np.cos(az), s * np.sin(az), mu])

radii = np.arange(30.0, 71.0, 5.0)
frac_neg, pen_mean, mean_rho = [], [], []
for r in radii:
    q = (r / L) * dirs
    lap = np.asarray(lap_batch(q))
    assert np.isfinite(lap).all()
    rho = lap * V ** 2 / (4.0 * np.pi * G * L ** 2)
    frac_neg.append(float((lap < 0).mean()))
    pen_mean.append(float(np.mean(np.arcsinh(np.maximum(-lap, 0.0)))))
    mean_rho.append(float(rho.mean()))
band = dict(radii=radii.tolist(), frac_rho_neg=frac_neg, mean_prior_neg=pen_mean,
            mean_rho_msun_kpc3=mean_rho,
            band_mean_frac_rho_neg=float(np.mean(frac_neg)),
            band_mean_prior_neg=float(np.mean(pen_mean)))
for r, fn, pm in zip(radii, frac_neg, pen_mean):
    print("r={:5.1f} kpc  frac(rho<0)={:6.1%}  mean prior_neg={:.4f}".format(r, fn, pm))
print("BAND 30-70 kpc: mean frac(rho<0)={:.1%}  mean prior_neg={:.4f}".format(
    band["band_mean_frac_rho_neg"], band["band_mean_prior_neg"]))

rng = np.random.default_rng(0)
u = rng.random(65536)
r = 7.0 * u ** (1.0 / 3.0)
g = rng.normal(size=(65536, 3))
g /= np.linalg.norm(g, axis=1, keepdims=True)
qg = r[:, None] * g
lapg = np.concatenate([np.asarray(lap_batch(chunk)) for chunk in np.array_split(qg, 4)])
rg_kpc = r * L
probe = dict(
    n=65536, radius_kpc=70.0,
    frac_rho_neg_all=float((lapg < 0).mean()),
    frac_rho_neg_inner=float((lapg[rg_kpc < 30.0] < 0).mean()),
    frac_rho_neg_30_70=float((lapg[(rg_kpc >= 30.0) & (rg_kpc <= 70.0)] < 0).mean()),
    mean_prior_neg_all=float(np.mean(np.arcsinh(np.maximum(-lapg, 0.0)))),
)
print("uniform-ball probe (n=65536, R=70 kpc): frac(rho<0) all={:.1%} r<30={:.1%} 30-70={:.1%}".format(
    probe["frac_rho_neg_all"], probe["frac_rho_neg_inner"], probe["frac_rho_neg_30_70"]))
print("reference (published pt-adjudication, node 11a39162, its own 30-70 kpc Sobol grid):")
print("  mean frac(rho<=0): base 20.8%  s11 23.1%  S1 15.4%")
out = dict(band=band, uniform_ball_probe=probe)
json.dump(out, open("runs/orx/grid_prior_evidence.json", "w"), indent=2)
print("wrote runs/orx/grid_prior_evidence.json")
sys.exit(0)
PYEOF
test $? -eq 0 || { echo "GRID-PRIOR NEGATIVE-DENSITY EVIDENCE FAILED"; exit 6; }
echo 'FLOW REPRODUCTION PASSED - GRID-PRIOR RUN COMPLETE'
echo '=== W1024-CSMOOTH-S1-GRIDPRIOR DONE ==='
