#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u
from faxes_transform import *
from scipy.stats import binned_statistic_dd

# Input the position, velocity and potential energy of particles.
# Output the phase-space averaged r and lambda_z
# Email: lzhu@shao.ac.cn

def Elz_bin(x0, y0, z0, vx0, vy0, vz0, pot):

#    x0 = x0* a /h
#    y0 = y0* a /h
#    z0 = z0* a /h
#    vx0 = vx0*np.sqrt(a)
#    vy0 = vy0*np.sqrt(a)
#    vz0 = vz0*np.sqrt(a)
#    pot = pot/a


#align x, y, z, vx, vy, vz  with principle axis
    xyz = np.array([x0,y0,z0])
    xyz.reshape([len(x0),3])
    vxyz = np.array([vx0,vy0,vz0])
    vxyz.reshape([len(x0),3])

    ba, ca, angle, Tiv = get_shape(xyz.T, mass, Rb=10.)
    xpart_axis = np.dot(Tiv, xyz).T
    vpart_axis = np.dot(Tiv, vxyz).T

    x = xpart_axis[:,0]
    y = xpart_axis[:,1]
    z = xpart_axis[:,2]
    vx = vpart_axis[:,0]
    vy = vpart_axis[:,1]
    vz = vpart_axis[:,2]


# Bin the particles in the phase-space of binding energy:Eb, total angular momentum: ll, angular momentum aound z axis: Lz
    Ek = 0.5*(vx**2 + vy**2 + vz**2)
    Eb = pot + Ek
    lx = -y*vz + z*vy
    ly = -x*vz + z*vx
    lz = -x*vy + y*vx
    ll = np.sqrt(lx**2.0 + ly**2.0 + lz**2.0)

    v2 = vx**2 + vy**2 + vz**2
    r2 = x**2 + y**2 + z**2
    r0 = np.sqrt(r2)
    vrms2 = vx**2 + vy**2 + vz**2 + 2*vx*vy + 2*vy*vz + 2*vx*vz
    Vrms = np.sqrt(vrms2)


# In Order to make ~equal number of particles in each bin,
# we map Eb, lz, ll to axises that the points are almost uniformly distributed.
    s_Eb = np.argsort(Eb)  # the indices sorted Eb, s_Eb and Eb are one-to-one maped
    ranks_Eb = np.empty_like(s_Eb)
    ranks_Eb[s_Eb] = np.arange(len(Eb))

    s_lz = np.argsort(lz)
    ranks_lz = np.empty_like(s_lz)
    ranks_lz[s_lz] = np.arange(len(lz))

    s_ll = np.argsort(ll)
    ranks_ll = np.empty_like(s_ll)
    ranks_ll[s_ll] = np.arange(len(ll))

    lz_a = np.zeros(len(lz))
    v2_a = np.zeros(len(lz))
    r_a = np.zeros(len(lz))

    maxEb = np.max(ranks_Eb)
    minEb = np.min(ranks_Eb)

    maxlz = np.max(ranks_Eb)
    minlz = np.min(ranks_Eb)

    maxll = np.max(ranks_Eb)
    minll = np.min(ranks_Eb)
    range3d = [[minEb, maxEb], [minlz, maxlz],[minll,maxll]]


    bins3d = [101,101,101]  # The bin number in the 3D phase-space, which you can play a bit
    N = len(ranks_lz)
    D = 3
    data = np.reshape([ranks_Eb,ranks_lz,ranks_ll],[D,N])
    data = data.T      # (in shape of [N, D])


# Calculate the average value of r, lz, vrms for particles in each bin
    ret = binned_statistic_dd(data, r0,  statistic='mean', bins=bins3d,range = range3d, expand_binnumbers=True)
    bindn = ret.binnumber-1
    retdn = ret.statistic
    for k in range(N):
        r_a[k] = retdn[bindn[0,k], bindn[1,k], bindn[2,k]]


    ret = binned_statistic_dd(data, lz,  statistic='mean', bins=bins3d,range = range3d, expand_binnumbers=True)
    bindn = ret.binnumber-1
    retdn = ret.statistic
    for k in range(N):
        lz_a[k] = retdn[bindn[0,k], bindn[1,k], bindn[2,k]]


    ret = binned_statistic_dd(data, vrms2,  statistic='mean', bins=bins3d,range = range3d, expand_binnumbers=True)
    bindn = ret.binnumber-1
    retdn = ret.statistic
    for k in range(N):
        v2_a[k] = retdn[bindn[0,k], bindn[1,k], bindn[2,k]]


    r = r_a
    lambdaz = lz_a/(r_a * np.sqrt(v2_a))

 # Align lambdaz with the major disk of the galaxy, if any
    sdd = np.where((r_a >2) & (r_a<rmax))
    if np.sum(lambdaz[sdd]) <0: lambdaz = -lambdaz

return r, lambdaz
