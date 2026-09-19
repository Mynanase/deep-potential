#!/usr/bin/env bash
# S1 stratified radial sampling on the frozen clean+smooth route (w1024).
# Identical to the parent route-closure contract except flow_sampling.radial_alloc
# (stratified radial quotas + importance weights). Flow training is unchanged and
# must reproduce the fb85670e reference; the sampling and Phi stages switch to the
# stratified importance-sampled estimator, so Phi val is reported without reference.
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

echo '=== W1024-CSMOOTH-S1 PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -f "$OLD" || { echo "PREFLIGHT FAIL: missing $OLD"; exit 2; }
test -f "$REG" || { echo "PREFLIGHT FAIL: missing $REG"; exit 2; }
"$PY" -c 'import jax, equinox, diffrax, flowjax, h5py' || { echo 'PREFLIGHT FAIL: deps'; exit 2; }
"$PY" -c 'import h5py; f=h5py.File("'"$OLD"'","r"); a=dict(f.attrs); print("source attrs:", {k:a[k] for k in ("cleaning_registry_set","cleaning_removed_count","length_unit","velocity_unit") if k in a}, "n:", f["eta"].shape[0])'
"$PY" - <<'PYEOF'
import json
opts = json.load(open("scripts/auriga/options.json"))
fs = opts["flow_sampling"]
ra = fs.get("radial_alloc")
assert ra is not None, "S1 options must contain flow_sampling.radial_alloc"
assert len(ra["fractions"]) == len(ra["bin_edges"]) - 1
assert abs(sum(ra["fractions"]) - 1.0) < 1e-9
print("S1 radial_alloc:", ra)
print("S1 flow_sampling keys:", sorted(fs.keys()))
PYEOF

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

echo '=== STEP 2: fixed three-stage w1024 pipeline (S1 stratified sampling) ==='
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

echo '=== REPRODUCTION CHECK (flow stage vs fb85670e; Phi is a different S1 estimand) ==='
"$PY" - <<'PYEOF'
import json, sys
ref = dict(val_pos=3.7245, val_vel=3.4077)
tol = 1e-2
f = json.load(open("runs/orx/models/df/flow/flow-21_loss.json"))
p = json.load(open("runs/orx/models/Phi/potential-10_loss.json"))
got_flow = dict(val_pos=f["val_pos"][-1], val_vel=f["val_vel"][-1])
phi_val = p["val"][-1]
ok = True
for k in ref:
    rel = abs(got_flow[k] - ref[k]) / abs(ref[k])
    ok &= rel < tol
    print("%s: got %.4f ref %.4f rel %.2e %s" % (k, got_flow[k], ref[k], rel, "PASS" if rel < tol else "FAIL"))
print("phi_val (S1 weighted estimand, reported without reference): %.4f" % phi_val)
sys.exit(0 if ok else 3)
PYEOF
test $? -eq 0 || { echo "FLOW REPRODUCTION CHECK FAILED"; exit 3; }
echo 'FLOW REPRODUCTION PASSED - S1 run complete'
echo '=== W1024-CSMOOTH-S1 DONE ==='
