#!/usr/bin/env python
"""Diagnose: why is a per-(orbit, line) Python loop slow even with JAX?

Hypothesis: the 300 s/round is dominated by per-iteration *dispatch* overhead
(Python loop + JAX op launch + tracing + device sync), not by actual FLOPs,
when each iteration does a tiny "match a line against an orbit LOSVD" op.

This script measures, on your machine/env:
  [1] per-call cost of a tiny un-jitted JAX op     (dispatch bound)
  [2] per-call cost of the same op under jit, called in a Python loop
  [3] numpy per-op cost (comparison)
  [4] a realistic mini-scene: loop over (orbit, line) pairs doing a cheap
      match+accumulate with jax  -> extrapolate to 4000 lines x N_orbit
  [5] the SAME computation fully vectorised (few big ops)             -> ratio

Usage: python scripts/bench_dispatch_overhead.py [--n-orbit 2500] [--n-line 4000]
"""

import argparse
import time

import numpy as np


def bench_jax(n_orbit, n_line, n_pix):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)

    print("device:", jax.devices())

    x = jnp.ones(1024, dtype=jnp.float32)

    # [1] tiny un-jitted op, called in loop (dispatch per call)
    y = (x * 2.0).block_until_ready()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        y = x * 2.0
    y.block_until_ready()
    t1 = (time.perf_counter() - t0) / n
    print(f"[1] un-jitted jax op in loop : {t1*1e6:7.1f} us/call")

    # [2] jitted op, but CALLED per iteration (still dispatch per call)
    f = jax.jit(lambda a: a * 2.0 + 1.0)
    f(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        y = f(x)
    y.block_until_ready()
    t2 = (time.perf_counter() - t0) / n
    print(f"[2] jit(f) called per iter   : {t2*1e6:7.1f} us/call")

    # [3] numpy comparison
    xn = np.ones(1024, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n):
        yn = xn * 2.0
    t3 = (time.perf_counter() - t0) / n
    print(f"[3] numpy op in loop         : {t3*1e6:7.1f} us/call")

    # ------------------------------------------------------------------
    # [4] mini-scene: per (orbit, line) "match + accumulate", one by one.
    #     line: (amp, center_pix);  orbit: LOSVD profile on n_kern bins.
    #     match: add amp * LOSVD centred at the line position (sampled),
    #     with a simple condition (line inside range). This is the cheap
    #     end of realistic per-pair work (delta-line matching).
    # ------------------------------------------------------------------
    n_kern = 129
    losvd_np = rng.dirichlet(np.ones(n_kern) * 5, size=n_orbit)  # (O, K)
    lines_np = np.stack([rng.uniform(0.2, 0.8, n_line),          # amp
                         rng.uniform(50, n_pix - 50, n_line)], 1)  # center

    losvd = jnp.asarray(losvd_np)
    amps = jnp.asarray(lines_np[:, 0])
    cents = jnp.asarray(lines_np[:, 1]).astype(jnp.int32)

    # per-pair python-loop version (dispatch-bound). Subsample for timing.
    sub = 3000
    t0 = time.perf_counter()
    acc = jnp.zeros(n_pix)
    for i in range(sub):
        c = int(cents.at[i].get()) if hasattr(cents, "at") else int(lines_np[i, 1])
        k = int(losvd_np[i % n_orbit].argmax())  # simple condition/demo
        w = losvd[i % n_orbit]
        half = n_kern // 2
        lo, hi = max(0, c - half), min(n_pix, c + half + 1)
        seg = jnp.zeros(n_pix)
        seg = seg.at[lo:hi].add(float(amps[i % n_line]) * w[lo - c + half: hi - c + half])
        acc = acc + seg
    acc.block_until_ready()
    t_pair = (time.perf_counter() - t0) / sub
    total_pairs = n_orbit * n_line
    est4 = t_pair * total_pairs
    print(f"[4] per-pair loop (jax arrays, python control): "
          f"{t_pair*1e6:7.1f} us/pair -> {n_orbit}x{n_line} ~ {fmt(est4)}")

    # [5] fully vectorised version of the same computation:
    #     one big op for ALL pairs (scatter-add via precomputed index matrix)
    t0 = time.perf_counter()
    idx = (jnp.arange(n_kern)[None, :] - half
           + cents[:, None])                    # (L, K) pixel indices
    contrib = amps[:, None] * losvd[jnp.arange(n_orbit)[:, None] % n_orbit][
        jnp.arange(n_line)[:, None] % n_orbit
    ]  # broadcast demo: pair (line i, orbit i%O) as in the loop above
    acc_v = jnp.zeros(n_pix)
    acc_v = acc_v.at[idx.reshape(-1)].add(contrib.reshape(-1))
    acc_v.block_until_ready()
    t5 = time.perf_counter() - t0
    print(f"[5] vectorised (one scatter-add)  : {t5*1e3:7.2f} ms total "
          f"-> x{est4/t5:.0f} vs [4]")

    # [6] full jit of the vectorised computation
    def round_vectorised(losvd, amps, cents):
        idx = (jnp.arange(n_kern)[None, :] - half + cents[:, None])
        pair_orbit = jnp.arange(n_line) % n_orbit
        contrib = amps[:, None] * losvd[pair_orbit]
        out = jnp.zeros(n_pix)
        return out.at[idx.reshape(-1)].add(contrib.reshape(-1))

    jf = jax.jit(round_vectorised)
    jf(losvd, amps, cents).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        out = jf(losvd, amps, cents)
    out.block_until_ready()
    t6 = (time.perf_counter() - t0) / 10
    print(f"[6] jit(vectorised)              : {t6*1e3:7.2f} ms/round "
          f"-> x{est4/t6:.0f} vs [4]  (and jit enables GPU gratis)")

    print(f"\n=> if your 300 s round is overhead-bound (likely), the vectorised+"
          f"\n   jit form maps it to ~{300*t6/est4:.3g} s equivalent; even with a"
          "\n   safety margin of 100x for heavier per-pair work, that is seconds.")


def fmt(sec):
    if sec < 1:
        return f"{sec*1e3:.0f} ms"
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec/60:.1f} min"
    return f"{sec/3600:.2f} h"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-orbit", type=int, default=2500)
    p.add_argument("--n-line", type=int, default=4000)
    p.add_argument("--n-pix", type=int, default=1024)
    a = p.parse_args()
    bench_jax(a.n_orbit, a.n_line, a.n_pix)
