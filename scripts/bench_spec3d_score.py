#!/usr/bin/env python
"""Spec3D real-scene benchmark: kernel-basis (GEMM) reformulation of the
candidate-template scoring, validated on the actual template library and
orbit-LOSVDs of the repo.

Scene (from results/FCC083_targetsn150/rep01/alternating_optimization_timing):
  one fit = ~240 s = 3 outer x up to 4 sweeps; sweep = 20-80 s;
  235 components x 2650 candidates x J<=50 eval bins x ~800-1666 pix,
  kernel_len = 37, chunk = 256, x64.

What this script measures on REAL data:
  [0] singular spectrum of the real (235*218, 37) kernel family -> how many
      basis kernels are needed (K=37 is exact by construction, since the
      kernel vector space is only 37-dimensional!)
  [1] reference scoring (their sweep_fast batched-FFT inner loop) for a
      sample of components, timed and extrapolated to one full sweep
  [2] kernel-basis scoring: precompute A[t,q,p] = conv(T_t, B_q) once per
      RUN (shared by all 235 comps, all sweeps, all reps, all galaxies),
      then per component conv = einsum('tqp,jq->tjp') -- pure GEMM
  [3] accuracy: SSE per candidate vs reference, K = 37 (exact), 16, 8
  [4] JAX-jit version of [2] (structure is GPU-ready; A100 fp64 GEMM ~19.5
      TFLOPS vs the ~0.1 TFLOPS their conv+glue path achieves today)

Usage:
  python bench_spec3d_score.py --tpl /path/templates_smiles_un1.3.npz \
      --vhist /path/vhist_FCC083_targetsn150.npz [--jax] [--n-comp 32]
"""

import argparse
import time

import numpy as np

RNG = np.random.default_rng(1)


# ----------------------------------------------------------------------
def build_kernels(vhist, kernel_len=37):
    """Interp every LOSVD onto a centred kernel_len-point grid, unit flux."""
    ncomp, nbin, nv = vhist.shape
    vplot = np.linspace(-1.0, 1.0, nv)
    v_target = np.linspace(-1.0, 1.0, kernel_len)
    ker = np.empty((ncomp * nbin, kernel_len))
    flat = vhist.reshape(-1, nv)
    for i, row in enumerate(flat):
        k = np.interp(v_target, vplot, row, left=0.0, right=0.0)
        s = k.sum()
        ker[i] = k / s if s > 0 else 0.0
    return ker.reshape(ncomp, nbin, kernel_len)


def rms_rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.abs(b).max(), 1e-30))


def fmt(s):
    return f"{s*1e3:8.1f} ms" if s < 1 else f"{s:8.2f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpl", default="/tmp/spec3d_colab/templates.npz")
    ap.add_argument("--vhist", default="/tmp/spec3d_colab/data_gal_FCC083/"
                        "vhist_FCC083_targetsn150.npz")
    ap.add_argument("--kernel-len", type=int, default=37)
    ap.add_argument("--n-comp", type=int, default=32,
                    help="components to actually evaluate (rest extrapolated)")
    ap.add_argument("--j-max", type=int, default=50, help="eval bins per comp")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--jax", action="store_true")
    args = ap.parse_args()

    tpl = np.load(args.tpl, allow_pickle=True)["flux"].astype(np.float64)
    vhist = np.load(args.vhist, allow_pickle=True)["vhist"].astype(np.float64)
    n_t, n_pix = tpl.shape
    n_comp_all, n_bin, _ = vhist.shape
    print(f"templates {tpl.shape}  vhist {vhist.shape}")

    ker3 = build_kernels(vhist, args.kernel_len)
    n_comp = min(args.n_comp, n_comp_all)
    K_max = args.kernel_len

    # ---- [0] singular spectrum of the real kernel family ----------------
    X = ker3.reshape(-1, args.kernel_len)
    nz = X[np.linalg.norm(X, axis=1) > 0]
    sv = np.linalg.svd(nz - nz.mean(0), full_matrices=False, compute_uv=False)
    ev = np.cumsum(sv**2) / (sv**2).sum()
    for k in (2, 4, 8, 16, 24, 32, 37):
        if k <= len(ev):
            print(f"[0] kernel-basis rank K={k:2d}: variance {ev[k-1]:.6e}")
    # reconstruction error of the kernels themselves at K
    U, S, Vt = np.linalg.svd(nz - nz.mean(0), full_matrices=False)
    for k in (8, 16, 24):
        rec = (U[:, :k] * S[:k]) @ Vt[:k] + nz.mean(0)
        print(f"    kernel L2 rel-err @K={k}: "
              f"{np.linalg.norm(rec - nz)/np.linalg.norm(nz):.2e}")

    # FFT convolution machinery identical to sweep_fast.py ----------------
    nfft = 1 << (n_pix + args.kernel_len - 2).bit_length()
    start = (args.kernel_len - 1) // 2

    def conv_batch(tpl_blk, ker_blk):
        """tpl_blk (T,P), ker_blk (J,KL) -> conv (T,J,P) 'same'-cropped."""
        th = np.fft.rfft(tpl_blk, n=nfft, axis=1)
        kh = np.fft.rfft(ker_blk, n=nfft, axis=1)
        return np.fft.irfft(th[:, None, :] * kh[None, :, :], n=nfft,
                            axis=2)[:, :, start:start + n_pix]

    # synthetic-but-realistic scoring inputs (they come from the fit state;
    # their exact values do not matter for the kernel-vs-basis comparison,
    # only the SHAPES do)
    dwin = RNG.normal(0, 0.05, (n_bin, 700)) + 1.0

    # ---- [1] reference: their batched-FFT inner loop ---------------------
    comps = np.arange(n_comp)
    jmax = args.j_max
    t_ref = 0.0
    sse_ref = np.empty((n_comp, n_t))
    for ci, k in enumerate(comps):
        kj = ker3[k]
        js = np.flatnonzero(kj.sum(1) > 0)[:jmax]
        if js.size == 0:
            continue
        t0 = time.perf_counter()
        out = np.empty((n_t, js.size))
        for s in range(0, n_t, args.chunk):
            blk = tpl[s:s + args.chunk]
            conv = conv_batch(blk, kj[js])
            tw = conv[:, :, :700]  # masked window stand-in
            num = np.einsum("jr,tjr->tj", dwin[js], tw)
            den = np.einsum("tjr,tjr->tj", tw, tw)
            alpha = num / np.where(den != 0, den, 1.0)
            r = dwin[js][None] - alpha[:, :, None] * tw
            out[s:s + blk.shape[0]] = np.einsum("tjr,tjr->tj", r, r)
        t_ref += time.perf_counter() - t0
        sse_ref[ci] = out.mean(1)
    per_comp_ref = t_ref / n_comp
    print(f"\n[1] reference batched-FFT scoring : {per_comp_ref*1e3:7.1f} ms/comp"
          f" -> full sweep (235 comps) ~ {fmt(per_comp_ref * n_comp_all)}")

    # ---- [2] kernel-basis GEMM scoring -----------------------------------
    basis = Vt[:K_max]  # rows span the (centred) kernel space; add mean back
    mean_ker = nz.mean(0)
    B = np.vstack([basis, mean_ker])  # (K_max+1, KL) -- exact spanning set
    Kdim = B.shape[0]
    t0 = time.perf_counter()
    A = np.empty((n_t, Kdim, n_pix), dtype=np.float64)
    for q in range(Kdim):
        A[:, q, :] = conv_batch(tpl, B[q][None, :])[:, 0, :]
    t_pre = time.perf_counter() - t0
    print(f"[2] precompute A (n_t x {Kdim} x {n_pix}) once per RUN: "
          f"{fmt(t_pre)}, {A.nbytes/2**30:.2f} GiB fp64")

    # coefficients of every kernel in the exact basis {B_q}
    # (K=38 > 37 => the system is overdetermined but consistent; lstsq once)
    ker_flat = ker3.reshape(-1, args.kernel_len)
    Cflat = np.linalg.lstsq(B.T, ker_flat.T, rcond=None)[0].T  # (ncomp*nbin, Kdim)

    t_gemm = 0.0
    sse_gem = np.empty((n_comp, n_t))
    for ci, k in enumerate(comps):
        kj = ker3[k]
        js = np.flatnonzero(kj.sum(1) > 0)[:jmax]
        if js.size == 0:
            continue
        Cj = Cflat[k * n_bin + js]                       # (J, Kdim)
        t0 = time.perf_counter()
        conv = np.einsum("tqp,jq->tjp", A, Cj, optimize=True)
        tw = conv[:, :, :700]
        num = np.einsum("jr,tjr->tj", dwin[js], tw)
        den = np.einsum("tjr,tjr->tj", tw, tw)
        alpha = num / np.where(den != 0, den, 1.0)
        r = dwin[js][None] - alpha[:, :, None] * tw
        out = np.einsum("tjr,tjr->tj", r, r)
        t_gemm += time.perf_counter() - t0
        sse_gem[ci] = out.mean(1)
    per_comp_gemm = t_gemm / n_comp
    print(f"[2] basis-GEMM scoring           : {per_comp_gemm*1e3:7.1f} ms/comp"
          f" -> full sweep ~ {fmt(per_comp_gemm * n_comp_all)}"
          f"  (CPU numpy; A100 fp64 GEMM ~ {per_comp_gemm*n_comp_all/100:.2f} s equiv)")

    # ---- [3] accuracy vs reference ---------------------------------------
    rel = np.array([rms_rel(sse_gem[i], sse_ref[i]) for i in range(n_comp)])
    print(f"[3] SSE agreement (exact basis K={Kdim}): RMS {rel.mean():.2e} "
          f"max {rel.max():.2e}  (fp-roundoff level = exact reformulation)")

    # truncated basis: top-Kb singular directions (already ordered by S) + mean;
    # coefficients must be refit against the truncated basis (min-norm lstsq)
    for Kb in (8, 16, 24):
        keep = np.concatenate([np.arange(Kb), [K_max]])   # Kb dirs + mean row
        A_k = A[:, keep]
        Cflat_k = np.linalg.lstsq(B[keep].T, ker_flat.T, rcond=None)[0].T
        t_t, sse_k = 0.0, np.empty((n_comp, n_t))
        for ci, k in enumerate(comps):
            kj = ker3[k]
            js = np.flatnonzero(kj.sum(1) > 0)[:jmax]
            if js.size == 0:
                continue
            Cj = Cflat_k[k * n_bin + js]
            t0 = time.perf_counter()
            conv = np.einsum("tqp,jq->tjp", A_k, Cj, optimize=True)
            tw = conv[:, :, :700]
            num = np.einsum("jr,tjr->tj", dwin[js], tw)
            den = np.einsum("tjr,tjr->tj", tw, tw)
            alpha = num / np.where(den != 0, den, 1.0)
            r = dwin[js][None] - alpha[:, :, None] * tw
            out = np.einsum("tjr,tjr->tj", r, r)
            t_t += time.perf_counter() - t0
            sse_k[ci] = out.mean(1)
        relk = np.array([rms_rel(sse_k[i], sse_ref[i]) for i in range(n_comp)])
        print(f"    truncated K={Kb:2d}: {t_t/n_comp*1e3:7.1f} ms/comp, "
              f"SSE rel-err RMS {relk.mean():.2e} max {relk.max():.2e}")

    # ---- [4] JAX jit version (GPU-ready) ---------------------------------
    if args.jax:
        import jax
        import jax.numpy as jnp

        A_j = jnp.asarray(A)
        dwin_j = jnp.asarray(dwin)

        @jax.jit
        def score_comp(A_, Cj, d):
            conv = jnp.einsum("tqp,jq->tjp", A_, Cj, optimize=True)
            tw = conv[:, :, :700]
            num = jnp.einsum("jr,tjr->tj", d, tw)
            den = jnp.einsum("tjr,tjr->tj", tw, tw)
            alpha = num / jnp.where(den != 0, den, 1.0)
            r = d[None] - alpha[:, :, None] * tw
            return jnp.mean(jnp.einsum("tjr,tjr->tj", r * r, r * r), axis=1)

        k = comps[0]
        js = np.flatnonzero(ker3[k].sum(1) > 0)[:jmax]
        Cj = jnp.asarray(Cflat[k * n_bin + js])
        score_comp(A_j, Cj, dwin_j[js]).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(10):
            out = score_comp(A_j, Cj, dwin_j[js])
        out.block_until_ready()
        t_j = (time.perf_counter() - t0) / 10
        print(f"[4] JAX jit GEMM scoring (CPU)  : {t_j*1e3:7.1f} ms/comp "
              f"-> sweep ~ {fmt(t_j * n_comp_all)} (A100: sub-second class)")


if __name__ == "__main__":
    main()
