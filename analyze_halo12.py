import h5py
import numpy as np

# Load the HDF5 data
filepath = '/localdisk/kosmos/my-deep-potential/data/halo_12_train.h5'
f = h5py.File(filepath, 'r')

print("Keys in HDF5 file:", list(f.keys()))
for key in f.keys():
    d = f[key][:]
    print(f"Dataset '{key}': shape={d.shape}, dtype={d.dtype}")

keys = list(f.keys())
if len(keys) == 1:
    data = f[keys[0]][:]
else:
    # Try common key names
    for k in ['data', 'positions_velocities', 'pos_vel', 'features']:
        if k in keys:
            data = f[k][:]
            break
    else:
        data = f[keys[0]][:]

f.close()

print(f"\nData shape: {data.shape}")
print(f"Data dtype: {data.dtype}")

# Interpret as x,y,z,vx,vy,vz (6 columns)
if data.ndim == 2 and data.shape[1] >= 6:
    x, y, z, vx, vy, vz = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5]
else:
    raise ValueError(f"Cannot parse data of shape {data.shape}")

dims_raw = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}

print("\n" + "="*70)
print("RAW DATA STATISTICS")
print("="*70)
print(f"{'Dim':<6} {'Mean':>14} {'Std':>14} {'Min':>14} {'Max':>14}")
print("-"*56)
for name, arr in dims_raw.items():
    print(f"{name:<6} {np.mean(arr):>14.4f} {np.std(arr):>14.4f} {np.min(arr):>14.4f} {np.max(arr):>14.4f}")

# Asinh transform
dims_asinh = {name: np.arcsinh(arr) for name, arr in dims_raw.items()}

print("\n" + "="*70)
print("ASINH TRANSFORMED STATISTICS")
print("="*70)
print(f"{'Dim':<12} {'Mean':>14} {'Std':>14} {'Min':>14} {'Max':>14}")
print("-"*58)
for name, arr in dims_asinh.items():
    print(f"asinh({name}){'':<5} {np.mean(arr):>14.4f} {np.std(arr):>14.4f} {np.min(arr):>14.4f} {np.max(arr):>14.4f}")

# Compute stds
raw_std = {name: np.std(arr) for name, arr in dims_raw.items()}
asinh_std = {name: np.std(arr) for name, arr in dims_asinh.items()}

print("\n" + "="*70)
print("SCALE ANALYSIS: STD RATIOS (Velocity / Spatial)")
print("="*70)

print("\nRaw data standard deviations:")
for name in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
    print(f"  std({name}) = {raw_std[name]:.4f}")

print("\nAsinh-transformed standard deviations:")
for name in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
    print(f"  std(asinh({name})) = {asinh_std[name]:.4f}")

# Aggregate ratios
raw_pos_std = np.mean([raw_std['x'], raw_std['y'], raw_std['z']])
raw_vel_std = np.mean([raw_std['vx'], raw_std['vy'], raw_std['vz']])
raw_ratio = raw_vel_std / raw_pos_std

asinh_pos_std = np.mean([asinh_std['x'], asinh_std['y'], asinh_std['z']])
asinh_vel_std = np.mean([asinh_std['vx'], asinh_std['vy'], asinh_std['vz']])
asinh_ratio = asinh_vel_std / asinh_pos_std

print(f"\n--- RAW DATA ---")
print(f"  Mean spatial std (x,y,z):  {raw_pos_std:.4f}")
print(f"  Mean velocity std (vx,vy,vz): {raw_vel_std:.4f}")
print(f"  Ratio velocity/spatial: {raw_ratio:.4f}")

print(f"\n--- ASINH TRANSFORMED ---")
print(f"  Mean spatial std (asinh(x,y,z)):  {asinh_pos_std:.4f}")
print(f"  Mean velocity std (asinh(vx,vy,vz)): {asinh_vel_std:.4f}")
print(f"  Ratio velocity/spatial: {asinh_ratio:.4f}")

# Per-dimension ratios
print("\n--- Per-dimension std ratios ---")
print(f"{'Pair':<10} {'Raw ratio':>12} {'Asinh ratio':>12} {'Compression':>14}")
print("-"*50)
for pos_dim, vel_dim in zip(['x','y','z'], ['vx','vy','vz']):
    raw_r = raw_std[vel_dim] / raw_std[pos_dim]
    asinh_r = asinh_std[vel_dim] / asinh_std[pos_dim]
    compression = raw_r / asinh_r if asinh_r > 0 else float('inf')
    print(f"{pos_dim}/{vel_dim:<6} {raw_r:>12.2f} {asinh_r:>12.2f} {compression:>12.2f}x")

print(f"\n=> asinh reduces the velocity/spatial scale mismatch by {raw_ratio/asinh_ratio:.2f}x")