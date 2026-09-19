#!/usr/bin/env bash
# CPU inventory of particle-level total-matter data on the gpu server.
set -eo pipefail
PY=/home/qiutao/miniforge3/envs/dp-jax/bin/python
DROOT=/localdisk/kosmos/my-deep-potential/data/auriga
export MPLCONFIGDIR=/tmp/orx-mpl
mkdir -p "$MPLCONFIGDIR"
echo '=== TOTAL-MATTER INVENTORY PREFLIGHT ==='
test -x "$PY" || { echo "PREFLIGHT FAIL: missing $PY"; exit 2; }
test -d "$DROOT" || { echo "PREFLIGHT FAIL: missing $DROOT"; exit 2; }
echo '=== DIRECTORY LISTING ==='
ls -la "$DROOT"
echo '=== H5 INSPECTION ==='
"$PY" - "$DROOT" <<'PYEOF'
import sys, os, glob
import h5py, numpy as np
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*")))
for path in files:
    if not os.path.isfile(path) or os.path.getsize(path) < 1e3:
        continue
    name = os.path.basename(path)
    print(f"--- {name}  ({os.path.getsize(path)/1e6:.1f} MB)")
    try:
        with h5py.File(path, "r") as f:
            print("  attrs:", {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in f.attrs.items()})
            def walk(g, prefix="  "):
                for k in list(g.keys())[:24]:
                    obj = g[k]
                    if isinstance(obj, h5py.Dataset):
                        print(f"{prefix}{k}: shape={obj.shape} dtype={obj.dtype}", end="")
                        if obj.attrs:
                            print(" attrs=" + str({a: (v.tolist() if hasattr(v, "tolist") else str(v)) for a, v in obj.attrs.items()}), end="")
                        print()
                    else:
                        print(f"{prefix}{k}/ (group)")
                        walk(obj, prefix + "  ")
            walk(f)
    except Exception as e:
        print("  (not h5py-readable:", repr(e)[:120], ")")
PYEOF
echo '=== TOTAL-MATTER INVENTORY DONE ==='
