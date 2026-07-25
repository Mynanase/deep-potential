#!/usr/bin/env python3
"""
Extract Auriga halo_12 star particles with full attributes,
perform principal axis alignment + rotation, kinematic classification,
export to HDF5, and generate diagnostic plots.

Usage:
    .venv/bin/python extract_halo12_stars.py
"""

import sys
sys.path.insert(0, '/home/tqiu/Auriga/auriga_analysis/')

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import auriga_public as ap
import numpy.linalg as LA
from scipy.stats import binned_statistic_dd

# ============================================================
# Configuration
# ============================================================
BASEPATH = '/home/share/Auriga'
SNAP_NUM = 127
HALO = 'halo_12'
ID = 12
PHI0_DEG = -40.0          # rotation angle in x-y plane
R_CUT = 75.0              # radial cut in kpc
RB_SHAPE = 40.0           # radius for get_shape
OUTDIR  = '/home/tqiu/Auriga/data'
PLOTDIR = '/home/tqiu/Auriga/plots'
OUTFILE = f'{OUTDIR}/halo_12_stars.hdf5'

h = 0.6777

COMP_NAMES = ['cold disk', 'warm disk', 'hot/bulge', 'counter-rot']
COMP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


# ============================================================
# Data loading
# ============================================================
def getpara(partType, ID):
    basePath = BASEPATH
    snap_num = SNAP_NUM
    strID = '%d' % ID
    halo = 'halo_' + strID
    directory = basePath + '//' + halo + '/'

    attrstoload = ['GroupFirstSub', 'Coordinates', 'Velocities', 'GFM_StellarFormationTime',
                   'GFM_InitialMass', 'Masses', 'ParticleIDs', 'Potential', 'SubhaloPos',
                   'SubhaloStellarPhotometrics', 'SubhaloMassType',
                   'Group_R_Crit200', 'Group_M_Crit200', 'GroupPos', 'SubhaloIDMostbound',
                   'SubhaloHalfmassRadType', 'SubhaloLenType', 'SubhaloLen']
    if partType == 4:
        attrstoload = ['GroupFirstSub', 'Coordinates', 'Velocities', 'GFM_StellarFormationTime',
                        'GFM_Metallicity', 'GFM_InitialMass', 'Masses', 'ParticleIDs', 'Potential',
                        'SubhaloPos', 'SubhaloStellarPhotometrics',
                        'GFM_Metals', 'GFM_StellarPhotometrics',
                        'Group_R_Crit200', 'Group_M_Crit200', 'GroupPos', 'SubhaloIDMostbound',
                        'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloLenType', 'SubhaloLen']

    outputdir = directory + '/'
    snapobj = ap.snapshot.load_snapshot(snap_num, partType, loadlist=attrstoload,
                                        snappath=outputdir, verbose=False)

    subhaloID = 0
    subobj = ap.subhalos.subfind(snap_num, directory=outputdir,
                                 loadlist=['ParticleIDs', 'GroupFirstSub', 'SubhaloPos',
                                           'Group_R_Crit200', 'Group_M_Crit200', 'SubhaloMassType',
                                           'Velocities', 'SubhaloLenType', 'SubhaloLen'])
    snapobj = ap.util.CentreOnHalo(snapobj, subobj.data['SubhaloPos'][subhaloID])

    if partType == 4:
        snapobj = ap.util.apply_mask(snapobj, stars=True, radialcut=None)
    else:
        snapobj = ap.util.apply_mask(snapobj, stars=False, radialcut=None)

    # Coordinate remap: [0,1,2] = [Z,Y,X] -> extract as (x,y,z)
    x = snapobj.data['Coordinates'][:, 2] * 1e3   # kpc
    y = snapobj.data['Coordinates'][:, 1] * 1e3
    z = snapobj.data['Coordinates'][:, 0] * 1e3
    mass = snapobj.data['Masses'] * 1e10            # Msun
    vx = snapobj.data['Velocities'][:, 2]           # km/s
    vy = snapobj.data['Velocities'][:, 1]
    vz = snapobj.data['Velocities'][:, 0]
    pid = snapobj.data['ParticleIDs']
    potential = snapobj.data['Potential']            # (km/s)^2

    met = np.zeros_like(mass)
    initial_mass = np.zeros_like(mass)
    formation_time = np.zeros_like(mass)
    metals = None
    photometrics = None

    if partType == 4:
        met = snapobj.data['GFM_Metallicity'] / 0.0127
        initial_mass = snapobj.data['GFM_InitialMass'] * 1e10
        formation_time = snapobj.data['GFM_StellarFormationTime']
        metals = snapobj.data['GFM_Metals']
        photometrics = snapobj.data['GFM_StellarPhotometrics']

    SubhaloLen = subobj.data['SubhaloLenType'][:, partType]
    if subhaloID == 0:
        n = SubhaloLen[subhaloID]
        x = x[0:n]; y = y[0:n]; z = z[0:n]
        vx = vx[0:n]; vy = vy[0:n]; vz = vz[0:n]
        mass = mass[0:n]; met = met[0:n]; pid = pid[0:n]
        potential = potential[0:n]
        initial_mass = initial_mass[0:n]
        formation_time = formation_time[0:n]
        if metals is not None:
            metals = metals[0:n]
            photometrics = photometrics[0:n]

    return x, y, z, mass, vx, vy, vz, met, pid, potential, initial_mass, formation_time, metals, photometrics


# ============================================================
# Shape determination
# ============================================================
def get_shape(pos, mass, Rb=40., tol=1e-3, maxiter=100):
    """Iterative reduced inertia tensor for principal axis alignment."""
    pos = pos.copy()

    r_sph = np.sqrt(np.sum(pos**2, axis=1))
    sel = r_sph < Rb

    q = 1.0  # b/a
    s = 1.0  # c/a
    Tiv_total = np.eye(3)

    for iteration in range(maxiter):
        if np.sum(sel) < 10:
            print("Warning: too few particles in selection (%d)" % np.sum(sel))
            break

        r_ell = np.sqrt(pos[sel, 0]**2 + (pos[sel, 1] / q)**2 + (pos[sel, 2] / s)**2)
        r_ell = np.maximum(r_ell, 1e-10)

        m_sel = mass[sel]
        p_sel = pos[sel]

        I = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                I[i, j] = np.sum(m_sel * p_sel[:, i] * p_sel[:, j] / r_ell**2)
                I[j, i] = I[i, j]

        eigvals, eigvecs = np.linalg.eigh(I)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        q_new = np.sqrt(eigvals[1] / eigvals[0]) if eigvals[0] > 0 else 1.0
        s_new = np.sqrt(eigvals[2] / eigvals[0]) if eigvals[0] > 0 else 1.0

        rot = eigvecs.T
        pos = np.dot(pos, eigvecs)
        Tiv_total = np.dot(rot, Tiv_total)

        if abs(q_new - q) < tol and abs(s_new - s) < tol and iteration > 0:
            q = q_new
            s = s_new
            break

        q = q_new
        s = s_new
        r_ell_all = np.sqrt(pos[:, 0]**2 + (pos[:, 1] / q)**2 + (pos[:, 2] / s)**2)
        sel = r_ell_all < Rb

    ba = q
    ca = s
    angle = np.array([
        np.arctan2(Tiv_total[2, 1], Tiv_total[2, 2]),
        np.arcsin(-np.clip(Tiv_total[2, 0], -1, 1)),
        np.arctan2(Tiv_total[1, 0], Tiv_total[0, 0])
    ]) * 180.0 / np.pi

    print("get_shape converged after %d iterations: b/a=%.4f, c/a=%.4f" % (iteration + 1, ba, ca))
    return ba, ca, angle, Tiv_total


# ============================================================
# Kinematic classification (Elz_bin / Zhu+2018 method)
# ============================================================
def classify_kinematics(x, y, z, vx, vy, vz, potential, bins3d=None):
    """Compute phase-space averaged lambda_z and assign kinematic labels.

    Follows Zhu+2018 / Elz_bin.py: particles are binned in rank-transformed
    (Eb, Lz, L) space; r, Lz, v^2 are averaged per bin to give lambda_z.
    """
    if bins3d is None:
        bins3d = [101, 101, 101]

    N = len(x)
    Ek = 0.5 * (vx**2 + vy**2 + vz**2)
    Eb = potential + Ek
    lx = y * vz - z * vy
    ly = z * vx - x * vz
    lz = x * vy - y * vx
    ll = np.sqrt(lx**2 + ly**2 + lz**2)
    r0 = np.sqrt(x**2 + y**2 + z**2)
    vrms2 = vx**2 + vy**2 + vz**2

    s_Eb = np.argsort(Eb)
    ranks_Eb = np.empty_like(s_Eb); ranks_Eb[s_Eb] = np.arange(N)
    s_lz = np.argsort(lz)
    ranks_lz = np.empty_like(s_lz); ranks_lz[s_lz] = np.arange(N)
    s_ll = np.argsort(ll)
    ranks_ll = np.empty_like(s_ll); ranks_ll[s_ll] = np.arange(N)

    range3d = [[0, N - 1], [0, N - 1], [0, N - 1]]
    data = np.column_stack([ranks_Eb, ranks_lz, ranks_ll])

    ret = binned_statistic_dd(data, r0, statistic='mean', bins=bins3d, range=range3d, expand_binnumbers=True)
    bindn = ret.binnumber - 1
    r_a = ret.statistic[bindn[0], bindn[1], bindn[2]]

    ret = binned_statistic_dd(data, lz, statistic='mean', bins=bins3d, range=range3d, expand_binnumbers=True)
    bindn = ret.binnumber - 1
    lz_a = ret.statistic[bindn[0], bindn[1], bindn[2]]

    ret = binned_statistic_dd(data, vrms2, statistic='mean', bins=bins3d, range=range3d, expand_binnumbers=True)
    bindn = ret.binnumber - 1
    v2_a = ret.statistic[bindn[0], bindn[1], bindn[2]]

    lambdaz = lz_a / (r_a * np.sqrt(v2_a))

    disk_sel = (r_a > 2) & (r_a < 20)
    if np.sum(lambdaz[disk_sel]) < 0:
        lambdaz = -lambdaz

    label = np.zeros(N, dtype=np.int8)
    label[lambdaz > 0.8] = 0                                      # cold disk
    label[(lambdaz > 0.25) & (lambdaz <= 0.8)] = 1               # warm disk
    label[np.abs(lambdaz) <= 0.25] = 2                            # hot/bulge
    label[lambdaz < -0.25] = 3                                    # counter-rot

    return lambdaz, r_a, label


# ============================================================
# HDF5 export
# ============================================================
def export_hdf5(outfile, x, y, z, vx, vy, vz, mass, met, pid, potential,
                initmass, formtime, metals, photo, lambdaz, r_avg, label,
                Tiv_star, ba, ca):
    """Write all star data + kinematic labels to HDF5."""
    with h5py.File(outfile, 'w') as f:
        header = f.create_group('Header')
        header.attrs['Description'] = 'Auriga halo_12 star particles (snap 127, z=0)'
        header.attrs['NumParticles'] = len(x)
        header.attrs['CoordSystem'] = 'Principal axis aligned, phi0=%g deg rotation, r3d<%g kpc' % (PHI0_DEG, R_CUT)
        header.attrs['Tiv_star'] = Tiv_star
        header.attrs['phi0_deg'] = PHI0_DEG
        header.attrs['ba'] = ba
        header.attrs['ca'] = ca

        grp = f.create_group('PartType4')

        for name, arr, unit in [('x', x, 'kpc'), ('y', y, 'kpc'), ('z', z, 'kpc'),
                                 ('vx', vx, 'km/s'), ('vy', vy, 'km/s'), ('vz', vz, 'km/s')]:
            grp.create_dataset(name, data=arr.astype('f8'))
            grp[name].attrs['units'] = unit

        grp.create_dataset('Masses', data=mass.astype('f8'))
        grp['Masses'].attrs['units'] = 'Msun'

        grp.create_dataset('GFM_InitialMass', data=initmass.astype('f8'))
        grp['GFM_InitialMass'].attrs['units'] = 'Msun'

        grp.create_dataset('GFM_Metallicity', data=met.astype('f8'))
        grp['GFM_Metallicity'].attrs['units'] = 'Z/Zsun'

        grp.create_dataset('GFM_Metals', data=metals.astype('f4'))
        grp['GFM_Metals'].attrs['description'] = 'Element mass fractions: H, He, C, N, O, Ne, Mg, Si, Fe'

        grp.create_dataset('GFM_StellarFormationTime', data=formtime.astype('f4'))
        grp['GFM_StellarFormationTime'].attrs['description'] = 'Scale factor at formation (a)'

        grp.create_dataset('GFM_StellarPhotometrics', data=photo.astype('f4'))
        grp['GFM_StellarPhotometrics'].attrs['description'] = 'Stellar photometric magnitudes (8 bands: U,B,V,K,g,r,i,z)'

        grp.create_dataset('Potential', data=potential.astype('f8'))
        grp['Potential'].attrs['units'] = '(km/s)^2'

        grp.create_dataset('ParticleIDs', data=pid)

        grp.create_dataset('lambda_z', data=lambdaz.astype('f8'))
        grp['lambda_z'].attrs['description'] = 'Phase-space averaged circularity (Zhu+2018/Elz_bin method)'
        grp.create_dataset('r_avg', data=r_avg.astype('f8'))
        grp['r_avg'].attrs['description'] = 'Phase-space averaged radius [kpc]'
        grp.create_dataset('kinematic_label', data=label)
        grp['kinematic_label'].attrs['description'] = '0=cold disk, 1=warm disk, 2=hot/bulge, 3=counter-rot'

    print("Exported %d stars to %s" % (len(x), outfile))


# ============================================================
# Visualization
# ============================================================
def visualize(outfile, outdir):
    """Generate all diagnostic plots from the exported HDF5."""
    f = h5py.File(outfile, 'r')
    grp = f['PartType4']

    x  = grp['x'][:];  y  = grp['y'][:];  z  = grp['z'][:]
    vx = grp['vx'][:]; vy = grp['vy'][:]; vz = grp['vz'][:]
    mass = grp['Masses'][:]
    epsilon = grp['lambda_z'][:]
    label = grp['kinematic_label'][:]
    r3d = np.sqrt(x**2 + y**2 + z**2)
    R_cyl = np.sqrt(x**2 + y**2)
    f.close()

    # ---- 1. Circularity distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins_eps = np.linspace(-1.5, 1.5, 100)

    ax = axes[0]
    ax.hist(epsilon, bins=bins_eps, color='gray', alpha=0.5, label='All stars')
    for i in range(4):
        ax.hist(epsilon[label == i], bins=bins_eps, alpha=0.6,
                color=COMP_COLORS[i], label=COMP_NAMES[i], histtype='step', linewidth=2)
    ax.set_xlabel(r'$\lambda_z$ (phase-space averaged circularity)', fontsize=14)
    ax.set_ylabel('N', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_title(r'$\lambda_z$ Distribution (Zhu+2018 method)')

    ax = axes[1]
    for i in range(4):
        ax.hist(epsilon[label == i], bins=bins_eps, density=True, alpha=0.6,
                color=COMP_COLORS[i], label=COMP_NAMES[i], histtype='step', linewidth=2)
    ax.set_xlabel(r'$\lambda_z$', fontsize=14)
    ax.set_ylabel('Normalized density', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_title('Normalized by Component')

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_circularity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_circularity.png")

    # ---- 2. 2D projections: ALL + each component ----
    rmax = 75; nbins = 150
    xbin2 = np.linspace(-rmax, rmax, nbins + 1)
    projections = [('x', 'z', x, z), ('x', 'y', x, y), ('y', 'z', y, z)]

    fig, axes = plt.subplots(5, 3, figsize=(18, 25))

    row_labels = ['All stars'] + COMP_NAMES
    row_masks = [np.ones(len(x), dtype=bool)] + [label == i for i in range(4)]

    for row, (rl, mask) in enumerate(zip(row_labels, row_masks)):
        for j, (xlabel, ylabel, d1, d2) in enumerate(projections):
            ax = axes[row, j]
            H, _, _ = np.histogram2d(d1[mask], d2[mask], bins=[xbin2, xbin2])
            ax.imshow(np.log10(H.T + 1), origin='lower', cmap='inferno',
                      extent=[-rmax, rmax, -rmax, rmax], vmin=0, vmax=np.log10(H.max() + 1))
            ax.set_xlabel(f'{xlabel} [kpc]'); ax.set_ylabel(f'{ylabel} [kpc]')
            ax.set_title(f'{rl}: {ylabel} vs {xlabel}')

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_2d_projections.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_2d_projections.png")

    # ---- 3. Radial profiles ----
    rbins = np.linspace(0, 75, 40)
    rc = 0.5 * (rbins[1:] + rbins[:-1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    for i in range(4):
        sel = label == i
        H, _ = np.histogram(r3d[sel], bins=rbins, weights=mass[sel])
        ax.semilogy(rc, H, color=COMP_COLORS[i], label=COMP_NAMES[i], linewidth=2)
    H_all, _ = np.histogram(r3d, bins=rbins, weights=mass)
    ax.semilogy(rc, H_all, 'k--', label='All', linewidth=1)
    ax.set_xlabel('r3d [kpc]'); ax.set_ylabel('dM/dr [Msun/kpc]')
    ax.set_title('Spherical Mass Profile'); ax.legend(fontsize=9)

    ax = axes[0, 1]
    for i in range(4):
        sel = label == i
        H, _ = np.histogram(R_cyl[sel], bins=rbins, weights=mass[sel])
        ax.semilogy(rc, H, color=COMP_COLORS[i], label=COMP_NAMES[i], linewidth=2)
    H_all, _ = np.histogram(R_cyl, bins=rbins, weights=mass)
    ax.semilogy(rc, H_all, 'k--', label='All', linewidth=1)
    ax.set_xlabel('R_cyl [kpc]'); ax.set_ylabel('dM/dR [Msun/kpc]')
    ax.set_title('Cylindrical Mass Profile'); ax.legend(fontsize=9)

    ax = axes[1, 0]
    zbins = np.linspace(-30, 30, 60)
    zc = 0.5 * (zbins[1:] + zbins[:-1])
    for i in range(4):
        sel = label == i
        H, _ = np.histogram(z[sel], bins=zbins, weights=mass[sel])
        ax.semilogy(zc, H, color=COMP_COLORS[i], label=COMP_NAMES[i], linewidth=2)
    H_all, _ = np.histogram(z, bins=zbins, weights=mass)
    ax.semilogy(zc, H_all, 'k--', label='All', linewidth=1)
    ax.set_xlabel('z [kpc]'); ax.set_ylabel('dM/dz [Msun/kpc]')
    ax.set_title('Vertical Mass Profile'); ax.legend(fontsize=9)

    ax = axes[1, 1]
    for i in range(4):
        sel = label == i
        idx_r = np.digitize(R_cyl[sel], rbins) - 1
        sigma_vz = np.array([np.std(vz[sel][idx_r == k])
                             for k in range(len(rc)) if np.sum(idx_r == k) > 10])
        r_valid = rc[:len(sigma_vz)]
        ax.plot(r_valid, sigma_vz, color=COMP_COLORS[i], label=COMP_NAMES[i], linewidth=2)
    ax.set_xlabel('R_cyl [kpc]'); ax.set_ylabel(r'$\sigma_{vz}$ [km/s]')
    ax.set_title('Vertical Velocity Dispersion'); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_profiles.png")

    # ---- 4. Edge-on slices at different z heights ----
    z_slices = [2, 5, 10, 20]
    fig, axes = plt.subplots(len(z_slices), 3, figsize=(18, 5 * len(z_slices)))

    for row, dz in enumerate(z_slices):
        sel_z = np.abs(z) < dz
        for j, (xlabel, ylabel, d1, d2) in enumerate(projections):
            ax = axes[row, j]
            for i in range(4):
                sel = sel_z & (label == i)
                step = max(1, len(d1[sel]) // 5000)
                ax.scatter(d1[sel][::step], d2[sel][::step], s=0.5, c=COMP_COLORS[i],
                           alpha=0.3, label=COMP_NAMES[i] if j == 0 else '')
            ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
            ax.set_xlabel(f'{xlabel} [kpc]'); ax.set_ylabel(f'{ylabel} [kpc]')
            ax.set_title(f'|z| < {dz} kpc: {ylabel} vs {xlabel}')
            if j == 0:
                ax.legend(fontsize=8, markerscale=5)

    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_slices.png")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Auriga halo_12 star extraction pipeline")
    print("=" * 60)

    # --- Step 1: Load data ---
    print("\n[1/6] Loading star particles...")
    (star_x, star_y, star_z, star_mass, star_vx, star_vy, star_vz,
     star_met, star_pid, star_pot, star_initmass, star_formtime,
     star_metals, star_photo) = getpara(partType=4, ID=ID)
    print("  Stars loaded: N=%d" % len(star_x))

    print("\n[1/6] Loading DM particles (for velocity centering)...")
    (dm_x, dm_y, dm_z, dm_mass, dm_vx, dm_vy, dm_vz,
     dm_met, dm_pid, dm_pot, _, _, _, _) = getpara(partType=1, ID=ID)
    print("  DM loaded: N=%d" % len(dm_x))

    # --- Step 2: Velocity centering ---
    print("\n[2/6] Velocity centering...")
    star_vx_mean = np.mean(star_vx)
    star_vy_mean = np.mean(star_vy)
    star_vz_mean = np.mean(star_vz)
    print("  Mean stellar velocity: (%.3f, %.3f, %.3f) km/s" % (star_vx_mean, star_vy_mean, star_vz_mean))

    star_vx -= star_vx_mean; star_vy -= star_vy_mean; star_vz -= star_vz_mean
    dm_vx -= star_vx_mean; dm_vy -= star_vy_mean; dm_vz -= star_vz_mean

    # --- Step 3: Principal axis alignment ---
    print("\n[3/6] Computing principal axes...")
    star_xyz = np.array([star_x, star_y, star_z])
    star_vxyz = np.array([star_vx, star_vy, star_vz])
    ba, ca, angle, Tiv_star = get_shape(star_xyz.T, star_mass, Rb=RB_SHAPE)
    print("  Stellar shape: b/a=%.4f, c/a=%.4f" % (ba, ca))

    # --- Step 4: Align + rotate + radial cut ---
    print("\n[4/6] Aligning, rotating, and applying radial cut...")
    star_xpart_axis = np.dot(Tiv_star, star_xyz).T
    star_vpart_axis = np.dot(Tiv_star, star_vxyz).T

    phi0 = PHI0_DEG * np.pi / 180.0
    x_al = star_xpart_axis[:, 0]
    y_al = star_xpart_axis[:, 1]
    z_al = star_xpart_axis[:, 2]
    vx_al = star_vpart_axis[:, 0]
    vy_al = star_vpart_axis[:, 1]
    vz_al = star_vpart_axis[:, 2]

    x_rot = x_al * np.cos(phi0) + y_al * np.sin(phi0)
    y_rot = -x_al * np.sin(phi0) + y_al * np.cos(phi0)
    z_rot = z_al
    vx_rot = vx_al * np.cos(phi0) + vy_al * np.sin(phi0)
    vy_rot = -vx_al * np.sin(phi0) + vy_al * np.cos(phi0)
    vz_rot = vz_al

    r3d = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)
    sel = r3d < R_CUT
    print("  Radial cut: %d -> %d stars" % (len(r3d), np.sum(sel)))

    # Apply selection to all arrays
    star_x_s = x_rot[sel]; star_y_s = y_rot[sel]; star_z_s = z_rot[sel]
    star_vx_s = vx_rot[sel]; star_vy_s = vy_rot[sel]; star_vz_s = vz_rot[sel]
    star_mass_s = star_mass[sel]; star_met_s = star_met[sel]; star_pid_s = star_pid[sel]
    star_pot_s = star_pot[sel]; star_initmass_s = star_initmass[sel]
    star_formtime_s = star_formtime[sel]; star_metals_s = star_metals[sel]
    star_photo_s = star_photo[sel]

    # --- Step 5: Kinematic classification + export ---
    print("\n[5/6] Kinematic classification...")
    lambdaz, r_avg, label = classify_kinematics(
        star_x_s,
        star_y_s,
        star_z_s,
        star_vx_s,
        star_vy_s,
        star_vz_s,
        star_pot_s,
    )
    for i, name in enumerate(COMP_NAMES):
        n = np.sum(label == i)
        m = np.sum(star_mass_s[label == i])
        print("  %20s: N=%8d  mass=%.2e Msun  (%.1f%%)" % (name, n, m, m / np.sum(star_mass_s) * 100))

    print("\n[5/6] Exporting to HDF5...")
    export_hdf5(OUTFILE, star_x_s, star_y_s, star_z_s, star_vx_s, star_vy_s, star_vz_s,
                star_mass_s, star_met_s, star_pid_s, star_pot_s,
                star_initmass_s, star_formtime_s, star_metals_s, star_photo_s,
                lambdaz, r_avg, label, Tiv_star, ba, ca)

    # --- Step 6: Visualization ---
    print("\n[6/6] Generating plots...")
    import os; os.makedirs(OUTDIR, exist_ok=True); os.makedirs(PLOTDIR, exist_ok=True)
    visualize(OUTFILE, PLOTDIR)

    print("\n" + "=" * 60)
    print("Done! Output:")
    print("  HDF5: %s" % OUTFILE)
    print("  Plots: %s/fig_*.png" % PLOTDIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
