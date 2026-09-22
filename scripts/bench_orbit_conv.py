#!/usr/bin/env python
"""Benchmark: orbit-LOSVD x spectral-template convolution acceleration.

Problem (galaxy dynamics + full-spectrum / SED fitting):
    N_T templates (lines/SSP), N_O orbits, each orbit has a LOSVD kernel.
    Need (for every aperture) the convolved spectra  S_oj = T_j (*) L_o,
    or more often the weight-contracted model spectrum
        M_a = sum_o sum_j w_ao t_j (T_j (*) L_a,o).
    Naive double loop is O(N_T * N_O) convolutions.

Methods benchmarked here
    1. naive_loop    : scipy fftconvolve per (template, orbit) pair.
    2. batched_fft   : precomputed FFTs of templates and LOSVDs,
                       vectorised pointwise-multiply + batched irFFT.
    3. pca_losvd     : PCA basis {B_k} of the LOSVD family (K comps).
                       Precompute A_jk = T_j (*) B_k  (only N_T*K FFTs).
                       S_oj = sum_k c_ok A_jk  -> one tensor contraction.
                       Convolution is linear in the kernel, so this is exact
                       up to the LOSVD truncation error.
    4. double_pca    : also PCA the templates (M comps). Core G_mk = P_m (*) B_k
                       (only M*K FFTs). Pair tensor = d x c x G; the full
                       N_T*N_O pair tensor is never materialised.

Model-level contraction (the quantity actually used in Schwarzschild-style
fits): with the factorised form the orbit sum collapses BEFORE touching the
pixel axis:
        M_a = t^T A u_a ,   u_a = C^T w_a   (K-vector)
i.e. N_O disappears from the expensive step.

Usage:
    python scripts/bench_orbit_conv.py            # full benchmark
    python scripts/bench_orbit_conv.py --small    # quick smoke run
    python scripts/bench_orbit_conv.py --jax      # + JAX timing of model step
"""

import argparse
import time

import numpy as np
from numpy.fft import irfft, rfft

try:
    from scipy.signal import fftconvolve
except ImportError:  # pragma: no cover
    fftconvolve = None

RNG = np.random.default_rng


# ----------------------------------------------------------------------
# synthetic data: line-library-like templates and orbit-like LOSVDs
# ----------------------------------------------------------------------
def make_templates(n_t, n_pix, rng):
    """Continuum + superposition of lines, mean-normalised."""
    x = np.linspace(0.0, 1.0, n_pix)
    T = np.empty((n_t, n_pix))
    # shared line positions => templates differ by amplitudes/widths (like a
    # line-ratio library or SSP grid)
    n_lines = 40
    centers = rng.uniform(0.05, 0.95, n_lines)
    widths = rng.uniform(1.5, 8.0, n_lines) / n_pix
    for j in range(n_t):
        amps = rng.dirichlet(np.full(n_lines, 0.6)) * rng.uniform(0.3, 1.0)
        sign = rng.choice([-1.0, 1.0], n_lines, p=[0.3, 0.7])  # abs/em mix
        spec = 0.0
        for a, c, w, s in zip(amps, centers, widths, sign):
            spec = spec + a * s * np.exp(-0.5 * ((x - c) / w) ** 2)
        # low-order continuum
        spec = spec + np.polynomial.legendre.legval(
            x, rng.normal(0, 0.05, 3) + np.array([1.0, 0.0, 0.0])
        )
        T[j] = spec / np.mean(spec)
    return T.astype(np.float64)


def make_losvds(n_o, n_kern, rng):
    """Mixture-of-Gaussians LOSVDs on a centred velocity-pixel grid."""
    offs = np.arange(n_kern) - n_kern // 2
    L = np.empty((n_o, n_kern))
    for o in range(n_o):
        n_g = rng.integers(1, 4)  # 1-3 components -> single/double peaked
        wgts = rng.dirichlet(np.full(n_g, 2.0))
        for w_, mu, sg in zip(
            wgts,
            rng.uniform(-25, 25, n_g),
            rng.uniform(3.0, 45.0, n_g),
        ):
            L[o] += w_ * np.exp(-0.5 * ((offs - mu) / sg) ** 2) / (
                sg * np.sqrt(2 * np.pi)
            )
    return L / L.sum(axis=1, keepdims=True)  # unit flux


# ----------------------------------------------------------------------
# FFT helpers (linear convolution, 'same' crop; Cappellari 2017 style:
# work on a log-lambda grid so a velocity kernel is shift-invariant)
# ----------------------------------------------------------------------
class Conv:
    def __init__(self, n_pix, n_kern):
        self.n_pix, self.n_kern = n_pix, n_kern
        self.n_fft = n_pix + n_kern - 1
        self.crop0 = (n_kern - 1) // 2

    def kern_fft(self, kern):
        return rfft(kern, n=self.n_fft, axis=-1)

    def conv(self, spec, kern_fft):
        """spec (..., n_pix) x kern_fft (..., n_fft//2+1) -> (..., n_pix)."""
        out = irfft(
            rfft(spec, n=self.n_fft, axis=-1) * kern_fft,
            n=self.n_fft,
            axis=-1,
        )
        return out[..., self.crop0 : self.crop0 + self.n_pix]


# ----------------------------------------------------------------------
# PCA helpers (numpy SVD, no sklearn dependency)
# ----------------------------------------------------------------------
def pca(X, k):
    """Return (coeffs n x k, basis k x m, singular values). Robust SVD."""
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:  # gesdd flakiness -> gesvd fallback
        from scipy.linalg import svd

        U, S, Vt = svd(X, full_matrices=False, lapack_driver="gesvd")
    return (
        (U[:, :k] * S[:k]).astype(np.float64),
        Vt[:k].astype(np.float64),
        S,
    )


def variance_explained(S):
    ev = S**2
    return np.cumsum(ev) / ev.sum()


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ----------------------------------------------------------------------
# benchmark sections
# ----------------------------------------------------------------------
def bench(n_t, n_o, n_pix, n_kern, seed, n_pair_sample, model_sample, use_jax):
    rng = RNG(seed)
    print(f"setup: N_T={n_t} templates, N_O={n_o} orbits, "
          f"N_pix={n_pix}, N_kern={n_kern}")

    T = make_templates(n_t, n_pix, rng)
    L = make_losvds(n_o, n_kern, rng)
    conv = Conv(n_pix, n_kern)

    # ---------------- ground truth on sampled pairs (direct convolution)
    assert fftconvolve is not None, "scipy required"
    jj = rng.choice(n_t, n_pair_sample, replace=False)
    oo = rng.choice(n_o, n_pair_sample, replace=False)
    truth = np.stack([
        fftconvolve(T[j], L[o], mode="same")
        for j, o in zip(jj, oo)
    ])

    # ---------------- 1. naive loop cost (per pair) --------------------
    t0 = time.perf_counter()
    for j, o in zip(jj[: max(1, n_pair_sample // 2)], oo[: max(1, n_pair_sample // 2)]):
        fftconvolve(T[j], L[o], mode="same")
    t_pair_naive = (time.perf_counter() - t0) / max(1, n_pair_sample // 2)
    est = t_pair_naive * n_t * n_o
    print(f"\n[1] naive fftconvolve loop: {t_pair_naive*1e6:8.1f} us/pair "
          f"-> full {n_t}x{n_o} matrix ~ {fmt(est)}")

    # ---------------- 2. batched FFT (vectorised pairwise) -------------
    FT_T = rfft(T, n=conv.n_fft, axis=1)              # (n_t, n_f)
    FT_L = conv.kern_fft(L)                            # (n_o, n_f)
    nb = 8                                             # orbits per batch
    t0 = time.perf_counter()
    for ob in rng.choice(n_o, nb, replace=False):
        _ = irfft(FT_L[ob][None, :] * FT_T,
                  n=conv.n_fft, axis=-1)[..., conv.crop0:conv.crop0 + n_pix]
    t_pair_batched = (time.perf_counter() - t0) / (nb * n_t)
    est2 = t_pair_batched * n_t * n_o
    print(f"[2] batched-FFT pairwise : {t_pair_batched*1e6:8.1f} us/pair "
          f"-> full matrix ~ {fmt(est2)}  "
          f"(x{t_pair_naive/t_pair_batched:.0f} vs naive)")

    # ---------------- 3. PCA of the LOSVD family ------------------------
    C, B, S = pca(L, n_kern)
    ve = variance_explained(S)
    print(f"\n[3] LOSVD PCA variance explained: "
          f"K=4: {ve[3]:.5f}  K=8: {ve[7]:.6f}  K=16: {ve[15]:.8f} "
          f"K=32: {ve[min(31, len(ve)-1)]:.3e}")

    K = 32
    Ck, Bk = C[:, :K], B[:K]
    # renormalise reconstructed LOSVDs to unit flux (flux conservation)
    Lhat = Ck @ Bk
    Ck = Ck / Lhat.sum(axis=1, keepdims=True)

    # precompute A_jk = T_j (*) B_k   (N_T * K FFT convolutions, once)
    t0 = time.perf_counter()
    A = np.empty((K, n_t, n_pix), dtype=np.float32)
    for k in range(K):
        A[k] = conv.conv(T, conv.kern_fft(Bk[k])).astype(np.float32)
    t_pre = time.perf_counter() - t0
    print(f"    precompute A (N_T*K FFT convs, K={K}): {t_pre:.2f} s, "
          f"{A.nbytes/2**30:.2f} GiB (float32), done ONCE")

    # pair-level accuracy via A
    Af = A.astype(np.float32)
    err = []
    for j, o, tr in zip(jj, oo, truth):
        pred = (Ck[o] @ Af[:, j, :]).astype(np.float64)
        err.append(rms_rel(pred, tr))
    err = np.array(err)
    print(f"    pair error via PCA(K=32): RMS {err.mean():.2e}  "
          f"max {err.max():.2e}")

    # pair-level throughput (extrapolated from a subsample of orbits)
    ns = 16
    os_ = rng.choice(n_o, ns, replace=False)
    t0 = time.perf_counter()
    _ = np.tensordot(Ck[os_].astype(np.float32), Af, axes=([1], [0]))
    t_pair_pca = (time.perf_counter() - t0) / (ns * n_t)
    est3 = t_pair_pca * n_t * n_o + t_pre
    print(f"    pairwise assembly     : {t_pair_pca*1e6:8.1f} us/pair "
          f"-> full matrix ~ {fmt(est3)} (incl. precompute)")

    # ---------------- 4. double PCA (templates too) ---------------------
    Ct, Bt, St = pca(T, min(n_t, 128))
    vet = variance_explained(St)
    print(f"\n[4] template PCA variance explained: M=16: {vet[15]:.6f} "
          f"M=32: {vet[31]:.8f} M=64: {vet[min(63, len(vet)-1)]:.3e}")
    M = min(n_t, 128)
    t0 = time.perf_counter()
    G = np.empty((K, M, n_pix), dtype=np.float32)
    for k in range(K):
        G[k] = conv.conv(Bt[:M], conv.kern_fft(Bk[k])).astype(np.float32)
    t_pre2 = time.perf_counter() - t0
    print(f"    precompute core G (M*K FFT convs, M={M}): {t_pre2:.2f} s, "
          f"{G.nbytes/2**20:.1f} MiB")
    D = Ct[:, :M].astype(np.float32)  # template coefficients (n_t x M)
    print("    pair error vs M (K=32):  ", end="")
    for m in (16, 32, 64, M):
        err2 = [rms_rel(np.einsum("m,kmp,k->p", D[j, :m],
                                  G[:, :m], Ck[o]).astype(np.float64), tr)
                for j, o, tr in zip(jj, oo, truth)]
        print(f"M={m}: {np.mean(err2):.1e}  ", end="")
    print("(NOTE: variance-explained is continuum-dominated; line structure")
    print("     needs larger M -- use error-on-convolution to pick M)")

    # ---------------- 5. model-level contraction (real use case) --------
    print(f"\n[5] model spectrum M = sum_oj w_o t_j S_oj "
          f"(weights for {n_o} orbits, {n_t} templates):")
    w = rng.uniform(0, 1, n_o)          # orbit weights (aperture)
    t = rng.uniform(0, 1, n_t)          # template weights
    # direct (subset-extrapolated): batched-FFT route
    ns = 32
    os_ = rng.choice(n_o, ns, replace=False)
    t0 = time.perf_counter()
    acc = np.zeros((n_t, n_pix))
    for o in os_:
        acc += w[o] * irfft(
            conv.kern_fft(L[o])[None, :] * FT_T,
            n=conv.n_fft, axis=-1,
        )[..., conv.crop0:conv.crop0 + n_pix]
    t_dir = (time.perf_counter() - t0) / ns * n_o
    acc *= (t[:, None] / n_t)  # not needed for timing; keep shapes honest
    # factorised route (pca_losvd): u = C^T w  (K-vector), then t^T A
    t0 = time.perf_counter()
    u = Ck.T @ w                                    # (K,)
    Mspec = np.einsum("j,kjp,k->p", t, Af, u)       # t^T A u, one shot
    t_fac = time.perf_counter() - t0
    print(f"    batched-FFT direct : ~{fmt(t_dir)} (extrapolated)")
    print(f"    factorised (K={K}) : {t_fac*1e3:.2f} ms  "
          f"-> x{t_dir/t_fac:.0f} speed-up, exact up to PCA truncation")

    # accuracy of the factorised model spectrum vs subsampled direct sum
    wsub, osub = w[os_], os_
    direct_sub = np.zeros(n_pix)
    for i, o in enumerate(osub):
        direct_sub += wsub[i] * np.einsum(
            "j,jp->p", t,
            irfft(conv.kern_fft(L[o])[None, :] * FT_T,
                  n=conv.n_fft, axis=-1)[..., conv.crop0:conv.crop0 + n_pix],
        )
    u_sub = Ck[osub].T @ wsub
    fac_sub = np.einsum("j,kjp,k->p", t, Af, u_sub)
    print(f"    model error (subsample, PCA K={K}): "
          f"{rms_rel(fac_sub, direct_sub):.2e}")

    # ---------------- optional: JAX version of the factorised step -----
    if use_jax:
        try:
            import jax

            jax.config.update("jax_enable_x64", False)
            import jax.numpy as jnp

            AfX = jnp.asarray(Af)
            CkX = jnp.asarray(Ck, dtype=jnp.float32)
            wX = jnp.asarray(w, dtype=jnp.float32)
            tX = jnp.asarray(t, dtype=jnp.float32)

            @jax.jit
            def model(w, t):
                u = CkX.T @ w
                return jnp.einsum("j,kjp,k->p", t, AfX, u)

            model(wX, tX).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(20):
                model(wX, tX).block_until_ready()
            t_jax = (time.perf_counter() - t0) / 20
            print(f"    JAX jitted (CPU)  : {t_jax*1e3:.2f} ms "
                  f"(structure is GPU-ready; same einsum on GPU ~us-ms)")
        except Exception as e:  # pragma: no cover
            print(f"    (jax unavailable: {e})")

    return dict(ve_losvd=ve, ve_tmpl=vet, err_pca=err, err_dpca=err2)


def rms_rel(pred, ref):
    denom = max(np.abs(ref).max(), 1e-12)
    return np.sqrt(np.mean((pred - ref) ** 2)) / denom


def fmt(sec):
    if sec < 1e-3:
        return f"{sec*1e6:.0f} us"
    if sec < 1:
        return f"{sec*1e3:.1f} ms"
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec/60:.1f} min"
    return f"{sec/3600:.2f} h"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="quick smoke run")
    p.add_argument("--jax", action="store_true", help="also time a JAX step")
    p.add_argument("--n-tmpl", type=int, default=4000)
    p.add_argument("--n-orbit", type=int, default=3000)
    p.add_argument("--n-pix", type=int, default=1024)
    p.add_argument("--n-kern", type=int, default=257)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-pair-sample", type=int, default=64)
    args = p.parse_args()
    if args.small:
        args.n_tmpl, args.n_orbit, args.n_pix = 400, 300, 512
    bench(
        args.n_tmpl, args.n_orbit, args.n_pix, args.n_kern, args.seed,
        args.n_pair_sample, args.n_orbit, args.jax,
    )
