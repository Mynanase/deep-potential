import h5py, numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/halo_12_train.h5"
with h5py.File(path, "r") as f:
    print("Datasets:", list(f.keys()))
    for k in f:
        d = f[k]
        print(f"\n{k}: shape={d.shape}, dtype={d.dtype}")
        n = d.shape[0]
        sample = d[:min(5000, n)]
        print(f"  sample size: {len(sample)}")
        if len(d.shape) == 2 and d.shape[1] >= 3:
            print(f"  x range: [{sample[:,0].min():.4f}, {sample[:,0].max():.4f}]")
            print(f"  y range: [{sample[:,1].min():.4f}, {sample[:,1].max():.4f}]")
            print(f"  z range: [{sample[:,2].min():.4f}, {sample[:,2].max():.4f}]")
            if d.shape[1] >= 6:
                print(f"  vx range: [{sample[:,3].min():.4f}, {sample[:,3].max():.4f}]")
                print(f"  vy range: [{sample[:,4].min():.4f}, {sample[:,4].max():.4f}]")
                print(f"  vz range: [{sample[:,5].min():.4f}, {sample[:,5].max():.4f}]")
            print(f"  mean: {sample.mean(axis=0)}")
            print(f"  std:  {sample.std(axis=0)}")
            r = np.sqrt(np.sum(sample[:,:3]**2, axis=1))
            print(f"  r range: [{r.min():.4f}, {r.max():.4f}], median: {np.median(r):.4f}")
            if d.shape[1] >= 6:
                v = np.sqrt(np.sum(sample[:,3:6]**2, axis=1))
                print(f"  |v| range: [{v.min():.4f}, {v.max():.4f}], median: {np.median(v):.4f}")
