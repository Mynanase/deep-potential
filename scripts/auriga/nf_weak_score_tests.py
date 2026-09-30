#!/usr/bin/env python3
"""T2 weak Stein score tests: calibration first, exploratory real estimates second.

Card T2 / Agent B of docs/nf-score-audit-plan.md.  The weak Stein statistic is
the mass-weighted projection

    t(h) = sum_i w_i [div h(z_i) + s_model(z_i) . h(z_i)] / sum_i w_i ,

zero-mean under the null (model score = true score) for every fixed field h.
The frozen test library is h = e_j psi(z) with three families:

  * S_{j,m} (spatial radial):  h = e_{q_j} u_m(r) q_j/r,
    div h = u_m (1/r - q_j^2/r^3) + u_m' q_j^2/r^2  (window derivative in the
    divergence);
  * V_{j,m} (velocity mean side):  h = e_{p_j} u_m(r), div h = 0;
  * W_{j,m} (velocity dispersion side):  h = e_{p_j} u_m(r) p_j, div h = u_m.

u_m(r) = w(r) phi_m(r)/norm_m with a C-infinity window w supported on r_q in
[3,7] (30-70 kpc) and phi_m in {1, cos, sin(2 pi r/lambda)} at frozen
wavelengths lambda in {0.5, 1.0, 2.0} q units (5, 10, 20 kpc).  Within a
family the statistic index is j*7+m (j = x,y,z; m = 0..6), families ordered
S, V, W (63 statistics total).

The analytic mock is the frozen T0 detection-region analog: Plummer radial
profile a = 6.8 kpc (a_q = 0.68) with exact inverse-CDF radii, isotropic
Gaussian velocities sigma_p^2(r) = (1 + r^2/a_q^2)^{-1/2}, closed-form log F
and score (velocity coupling included).  Band counts match a heldout-sized
sample (frozen mock band fractions x N_mock x 0.25).  Mock seed = 5,
bootstrap seed = 4 (t0-manifest new_diagnostic_candidates).

Calibration arms: null (plain / lognormal mass weights / score numerical
noise at the T1 outer-band median and p99 floored-rel scales), injected
delta log F = eps u_k (the mandated spatial family, all seven k), velocity
mean / dispersion validation injections (k = 0), and a non-gradient
rotation-field perturbation (pressure test, never a log F effect).
False-positive rates carry replicate (Wilson) uncertainty; joint inference
uses per-family shrunk Hotelling Q, a global max|z| with split-half
empirical critical values, and cluster (angular) / block (radial) bootstrap
sensitivity.

Auriga real-sample estimates are EXPLORATORY (t0-manifest: no qualified
confirmation set) and require the certified T1 cache locally, sha256-matched
against docs/nf-score-audit/t1-cache-registry.json.  Local CPU only; no
model loading, no training, no server access.
"""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# frozen constants (t0-manifest / card T2)
# ---------------------------------------------------------------------------

L_KPC = 10.0
A_Q = 0.68                     # Plummer a = 6.8 kpc in q units
BETA_SIGMA = 0.5               # sigma_p^2 = (1 + r^2/a_q^2)^(-beta)
SIGMA0 = 1.0
W_LO, W_HI = 3.0, 7.0          # outer window r_q in [3,7] = 30-70 kpc
LAMBDA_Q = (0.5, 1.0, 2.0)     # preset wavelengths (5, 10, 20 kpc)
N_MOCK = 1619615               # frozen mock population size
HELDOUT_FRACTION = 0.25        # heldout = first 25% rows
BAND_EDGES_Q = ((3.0, 4.5), (4.5, 6.0), (6.0, 7.0))   # 30-45/45-60/60-70 kpc
BAND_FRACTIONS = (0.0391, 0.01434, 0.00497)           # frozen mock fractions
BAND_N = tuple(int(round(f * N_MOCK * HELDOUT_FRACTION)) for f in BAND_FRACTIONS)
SEED_MOCK = 5
SEED_BOOT = 4
N_REPLICATES = 240
N_BOOTSTRAP = 2000
N_SENS_REPLICATES = 24         # cluster/block bootstrap sensitivity subset
N_SENS_BOOT = 400
ANGULAR_CLUSTERS = (6, 8)      # theta x phi cells
RADIAL_BLOCKS = 8
EPS_GRID_REL = (0.5, 1.0, 2.0)  # multiples of eps_ref per injection family
NOISE_REL_MEDIAN = 2.7e-3       # T1 outer-band stored-vs-x64 floored rel, median
NOISE_REL_P99 = 3.0e-2          # T1 outer-band p99
LOGNORMAL_WEIGHT_SIGMA = 0.3    # mass-weight machinery arm
ALPHAS = (0.05, 0.01)
SHRINKAGE = 0.1                 # diagonal shrinkage for family covariance
FAMILY_NAMES = ("S", "V", "W")
N_FAMILY = 21                   # 3 components x 7 radial functions
K_ALL = 3 * N_FAMILY            # 63 statistics
CHI2_DF = N_FAMILY - 1          # rank after centering note; see family_Q
T1_REGISTRY = "docs/nf-score-audit/t1-cache-registry.json"
DEFAULT_CACHE_DIR = "runs/nf-score-audit/t1-cache-local"


# ---------------------------------------------------------------------------
# frozen test library: window, radial basis, per-point coefficients
# ---------------------------------------------------------------------------

def _bump(x):
    """C-infinity bump on (0,1), B(1/2)=1, with derivative."""
    def f(t):
        return np.where(t > 0.0, np.exp(-1.0 / np.maximum(t, 1e-300)), 0.0)
    norm = np.exp(-4.0)          # f(1/2)^2
    fx, gx = f(x), f(1.0 - x)
    eps = 1e-12
    b = fx * gx / norm
    db = (fx / np.maximum(x, eps) ** 2 * gx
          - fx * gx / np.maximum(1.0 - x, eps) ** 2) / norm
    return b, db


def window(r):
    """w(r), dw/dr on the outer window [W_LO, W_HI]."""
    x = (r - W_LO) / (W_HI - W_LO)
    b, db = _bump(x)
    inside = (x > 0.0) & (x < 1.0)
    w = np.where(inside, b, 0.0)
    dw = np.where(inside, db, 0.0) / (W_HI - W_LO)
    return w, dw


def phi_basis(r):
    """phi_m(r), phi_m'(r), m = 0..6: 1, then cos/sin at three wavelengths."""
    out, dout = [np.ones_like(r)], [np.zeros_like(r)]
    for lam in LAMBDA_Q:
        k = 2.0 * np.pi / lam
        out.append(np.cos(k * r)); dout.append(-k * np.sin(k * r))
        out.append(np.sin(k * r)); dout.append(k * np.cos(k * r))
    return out, dout


def radial_pdf(r):
    """Mock radial density r^2 rho(r), unnormalized."""
    return r ** 2 * (1.0 + r ** 2 / A_Q ** 2) ** -2.5


def _norm_constants():
    """Frozen amplitude norms: E_window[u_m^2] = 1 under the mock radial law.

    Deterministic 1D quadrature (no RNG, no data); values recorded in metrics.
    """
    r = np.linspace(W_LO, W_HI, 20001)
    pdf = radial_pdf(r)
    pdf = pdf / np.trapezoid(pdf, r)
    phis, _ = phi_basis(r)
    w, _ = window(r)
    return [float(1.0 / np.sqrt(np.trapezoid(pdf * (w * phi) ** 2, r))) for phi in phis]


NORMS = _norm_constants()


def u_basis(r):
    """u_m(r), u_m'(r) with frozen normalization; shapes (n, 7)."""
    w, dw = window(r)
    phis, dphis = phi_basis(r)
    u = np.stack([w * p * c for p, c in zip(phis, NORMS)], axis=1)
    du = np.stack([(dw * p + w * dp) * c for p, dp, c in zip(phis, dphis, NORMS)], axis=1)
    return u, du


def h_field(z, family, j, m):
    """Frozen library vector field h at points z (for tests / documentation)."""
    r = np.linalg.norm(z[:, :3], axis=1)
    u, _ = u_basis(r)
    h = np.zeros_like(z)
    if family == "S":
        h[:, j] = u[:, m] * z[:, j] / r
    elif family == "V":
        h[:, 3 + j] = u[:, m]
    elif family == "W":
        h[:, 3 + j] = u[:, m] * z[:, 3 + j]
    else:
        raise ValueError(family)
    return h


def library_summary():
    kinds = ["const"] + [f"{fn}_lam{lam}" for lam in LAMBDA_Q for fn in ("cos", "sin")]
    return {
        "window_q": [W_LO, W_HI], "window_kpc": [W_LO * L_KPC, W_HI * L_KPC],
        "lambda_q": list(LAMBDA_Q), "lambda_kpc": [L_KPC * l for l in LAMBDA_Q],
        "basis": [{"m": m, "kind": kinds[m], "norm": round(NORMS[m], 6)} for m in range(7)],
        "families": {"S": "h = e_{q_j} u_m(r) q_j/r (radial)",
                     "V": "h = e_{p_j} u_m(r) (velocity mean side)",
                     "W": "h = e_{p_j} u_m(r) p_j (velocity dispersion side)"},
        "divergence": {"S": "u(1/r - q_j^2/r^3) + u' q_j^2/r^2 (window derivative included)",
                       "V": "0", "W": "u_m(r)"},
        "column_order": "family S then V then W; within family j*7+m, j=x,y,z",
        "statistics_total": K_ALL,
    }


# ---------------------------------------------------------------------------
# analytic mock: exact log F, exact score, exact samplers
# ---------------------------------------------------------------------------

def sigma_p2(r):
    return SIGMA0 ** 2 * (1.0 + r ** 2 / A_Q ** 2) ** -BETA_SIGMA


def mock_logF(z):
    q, p = z[:, :3], z[:, 3:]
    r2 = np.einsum("ij,ij->i", q, q)
    s2 = sigma_p2(np.sqrt(r2))
    p2 = np.einsum("ij,ij->i", p, p)
    return -2.5 * np.log1p(r2 / A_Q ** 2) - 1.5 * np.log(s2) - p2 / (2.0 * s2)


def mock_score(z):
    """grad_(q,p) log F, closed form; the q block includes the p-coupling."""
    q, p = z[:, :3], z[:, 3:]
    r2 = np.einsum("ij,ij->i", q, q)
    s2 = sigma_p2(np.sqrt(r2))
    p2 = np.einsum("ij,ij->i", p, p)
    radial = -3.5 / (A_Q ** 2 + r2) - 0.5 * p2 * s2 / A_Q ** 2
    sq = radial[:, None] * q
    sp = -p / s2[:, None]
    return np.concatenate([sq, sp], axis=1)


def _plummer_mass_fraction(x):
    return x ** 3 * (1.0 + x ** 2) ** -1.5


def _invert_mass_fraction(u, x_lo, x_hi, iters=80):
    lo = np.full_like(u, x_lo)
    hi = np.full_like(u, x_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_big = _plummer_mass_fraction(mid) > u
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return 0.5 * (lo + hi)


def sample_radii_in_band(rng, r_lo, r_hi, n):
    """Exact inverse-CDF radii of the mock radial law restricted to a band."""
    x_lo, x_hi = r_lo / A_Q, r_hi / A_Q
    m_lo, m_hi = _plummer_mass_fraction(x_lo), _plummer_mass_fraction(x_hi)
    u = rng.random(n) * (m_hi - m_lo) + m_lo
    return _invert_mass_fraction(u, x_lo, x_hi) * A_Q


def sample_mock_window(rng, lognormal_weights=False):
    """One heldout-sized replicate restricted to the detection window.

    Radii are drawn from the mock radial law restricted to the FULL window
    [3,7] by exact inverse CDF (band counts then vary binomially around the
    frozen BAND_N means; the window-restricted density is exactly the law the
    analytic score belongs to, so the Stein null holds without boundary
    leakage at interior band edges).
    """
    n = int(sum(BAND_N))
    r = sample_radii_in_band(rng, W_LO, W_HI, n)
    direction = rng.standard_normal((n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    q = r[:, None] * direction
    p = np.sqrt(sigma_p2(r))[:, None] * rng.standard_normal((n, 3))
    z = np.concatenate([q, p], axis=1)
    if lognormal_weights:
        w = np.exp(LOGNORMAL_WEIGHT_SIGMA * rng.standard_normal(len(z)))
        return z, w / w.mean()
    return z, np.ones(len(z))


def sample_full_mock(rng, n=N_MOCK):
    """Full-sphere draw, used only for band-fraction verification."""
    r = _invert_mass_fraction(rng.random(n), 0.0, 60.0) * A_Q
    direction = rng.standard_normal((n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    q = r[:, None] * direction
    p = np.sqrt(sigma_p2(r))[:, None] * rng.standard_normal((n, 3))
    return np.concatenate([q, p], axis=1)


# ---------------------------------------------------------------------------
# statistic matrix G (n, 63) = div h + s . h, weighted mean and covariance
# ---------------------------------------------------------------------------

def divergence_blocks(z):
    """(div_S, div_W) each (n, 3, 7); the V family divergence is zero."""
    q = z[:, :3]
    r = np.linalg.norm(q, axis=1)
    u, du = u_basis(r)
    q_over_r = q / r[:, None]
    q2_over_r2 = q_over_r ** 2
    inv_r = 1.0 / r[:, None, None]
    div_s = u[:, None, :] * (1.0 - q2_over_r2[:, :, None]) * inv_r \
        + du[:, None, :] * q2_over_r2[:, :, None]
    div_w = np.repeat(u[:, None, :], 3, axis=1)
    return div_s, div_w


def statistic_matrix(z, score):
    """G (n, 63) = div h + s.h for the whole frozen library."""
    n = len(z)
    r = np.linalg.norm(z[:, :3], axis=1)
    u, _ = u_basis(r)
    q_over_r = z[:, :3] / r[:, None]
    div_s, div_w = divergence_blocks(z)
    g_s = div_s + (score[:, :3] * q_over_r)[:, :, None] * u[:, None, :]
    g_v = score[:, 3:6][:, :, None] * u[:, None, :]
    g_w = div_w + (score[:, 3:6] * z[:, 3:6])[:, :, None] * u[:, None, :]
    G = np.concatenate([g_s.reshape(n, N_FAMILY), g_v.reshape(n, N_FAMILY),
                        g_w.reshape(n, N_FAMILY)], axis=1)
    return G.astype(np.float32)


def weighted_stats(G, weights):
    """Weighted mean t (63,) and Hajek sandwich covariance (63, 63).

    Var(t) = sum_i w_i^2 (g_i - t)(g_i - t)^T / (sum_i w_i)^2, the joint-
    resampling form (particle and its mass weight resampled together); f64
    reductions throughout.
    """
    sw = float(np.sum(weights, dtype=np.float64))
    t = np.sum(G * np.asarray(weights, dtype=np.float64)[:, None], axis=0, dtype=np.float64) / sw
    centered = G.astype(np.float64) - t
    wc = np.asarray(weights, dtype=np.float64)[:, None] ** 2 * centered
    cov = centered.T @ wc / sw ** 2
    return t, cov


def z_scores(t, cov):
    return t / np.sqrt(np.maximum(np.diag(cov), 1e-300))


def family_Q(t, cov):
    """Shrunk per-family Hotelling statistics (3,)."""
    out = np.empty(3)
    for f in range(3):
        sl = slice(f * N_FAMILY, (f + 1) * N_FAMILY)
        cf = np.ascontiguousarray(cov[sl, sl])
        shrunk = (1.0 - SHRINKAGE) * cf + SHRINKAGE * np.diag(np.diag(cf))
        try:
            out[f] = float(t[sl] @ np.linalg.solve(shrunk, t[sl]))
        except np.linalg.LinAlgError:
            out[f] = np.inf
    return out


# ---------------------------------------------------------------------------
# injections: model score = true score + delta
# ---------------------------------------------------------------------------

def injection_delta(z, family, m, eps):
    """Exact score perturbation; S/V1/V2 come from delta log F, NG does not."""
    r = np.linalg.norm(z[:, :3], axis=1)
    u, du = u_basis(r)
    q_over_r = z[:, :3] / r[:, None]
    if family == "S":                  # delta log F = eps u_m(r)
        delta_q = eps * du[:, m][:, None] * q_over_r
        delta_p = np.zeros_like(z[:, 3:])
    elif family == "V1":               # delta log F = eps u_m(r) p_x
        delta_q = eps * du[:, m][:, None] * z[:, 3][:, None] * q_over_r
        delta_p = np.zeros_like(z[:, 3:])
        delta_p[:, 0] = eps * u[:, m]
    elif family == "V2":               # delta log F = eps u_m(r) |p|^2/2
        p2 = np.einsum("ij,ij->i", z[:, 3:], z[:, 3:])
        delta_q = eps * (0.5 * p2 * du[:, m])[:, None] * q_over_r
        delta_p = eps * u[:, m][:, None] * z[:, 3:]
    elif family == "NG":               # parity-preserving non-gradient field
        # delta = eps u_m(r) q_x p_x^2 e_{q_x}: even where the S statistics are
        # odd, so it has a nonzero mean projection, but its mixed curl
        # d(delta_qx)/d(p_x) != 0 with delta_px = 0 means it is NOT the
        # gradient of any scalar log F (pressure test only).
        delta = np.zeros_like(z)
        delta[:, 0] = z[:, 0] * z[:, 3] ** 2 * u[:, m]
        return eps * delta
    else:
        raise ValueError(family)
    return np.concatenate([delta_q, delta_p], axis=1)


def quadrature_expectations():
    """Analytic unit-eps mean-shift coefficients (deterministic quadrature).

    e_s[k]  = E[u_k' u_0]/3   : S-injection k on matched statistic S_{j,0}
    e_v1    = E[u_0^2]        : V1-injection on matched V_{x,0}
    e_w     = E[u_0^2 s2]     : V2-injection on matched W_{j,0}
    """
    r = np.linspace(W_LO, W_HI, 20001)
    pdf = radial_pdf(r)
    pdf = pdf / np.trapezoid(pdf, r)
    u, du = u_basis(r)
    e_s = np.trapezoid(pdf[:, None] * du * u[:, :1], r, axis=0) / 3.0
    e_v1 = float(np.trapezoid(pdf * u[:, 0] ** 2, r))
    e_w = float(np.trapezoid(pdf * u[:, 0] ** 2 * sigma_p2(r), r))
    return e_s, e_v1, e_w


# ---------------------------------------------------------------------------
# resampling: i.i.d. bootstrap, angular clusters, radial blocks
# ---------------------------------------------------------------------------

def angular_cluster_ids(z):
    n_theta, n_phi = ANGULAR_CLUSTERS
    q = z[:, :3]
    r = np.linalg.norm(q, axis=1)
    theta = np.arccos(np.clip(q[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(q[:, 1], q[:, 0])
    it = np.minimum((theta / np.pi * n_theta).astype(int), n_theta - 1)
    ip = np.minimum(((phi + np.pi) / (2.0 * np.pi) * n_phi).astype(int), n_phi - 1)
    return it * n_phi + ip


def radial_block_ids(z):
    r = np.linalg.norm(z[:, :3], axis=1)
    edges = np.linspace(W_LO, W_HI, RADIAL_BLOCKS + 1)
    return np.clip(np.searchsorted(edges, r, side="right") - 1, 0, RADIAL_BLOCKS - 1)


def bootstrap_t(G, weights, n_boot, rng, cluster_ids=None):
    """Bootstrap distributions of t (n_boot, 63); rows or clusters resampled.

    Vectorized through multinomial resample counts: t_boot = (c . w G)/(c . w)
    with the (w_i, g_i) pair resampled jointly.
    """
    w64 = np.asarray(weights, dtype=np.float64)
    Gw = np.asarray(G, dtype=np.float64) * w64[:, None]
    out = np.empty((n_boot, G.shape[1]), dtype=np.float64)
    if cluster_ids is None:
        n = len(w64)
        p = np.full(n, 1.0 / n)
        chunk = max(1, min(256, n_boot))
        for start in range(0, n_boot, chunk):
            b = min(chunk, n_boot - start)
            counts = rng.multinomial(n, p, size=b).astype(np.float64)
            out[start:start + b] = (counts @ Gw) / (counts @ w64)[:, None]
    else:
        uniq = np.unique(cluster_ids)
        cluster_Gw = np.stack([Gw[cluster_ids == c].sum(axis=0) for c in uniq])
        cluster_w = np.stack([w64[cluster_ids == c].sum() for c in uniq])
        p = np.full(len(uniq), 1.0 / len(uniq))
        chunk = max(1, min(512, n_boot))
        for start in range(0, n_boot, chunk):
            b = min(chunk, n_boot - start)
            counts = rng.multinomial(len(uniq), p, size=b).astype(np.float64)
            out[start:start + b] = (counts @ cluster_Gw) / (counts @ cluster_w)[:, None]
    return out


def maxT_from_bootstrap(boot_t, t_obs):
    """Centered bootstrap max|t|/se critical values and two-sided p-value."""
    centered = boot_t - boot_t.mean(axis=0, keepdims=True)
    sd = np.maximum(centered.std(axis=0, ddof=1), 1e-300)
    zboot = np.abs(centered / sd)
    stat = np.abs(t_obs - boot_t.mean(axis=0)) / sd
    return zboot, stat


def wilson_interval(k, n, conf=0.95):
    from scipy.stats import norm
    z = float(norm.ppf(0.5 + conf / 2.0))
    p = k / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2)) / denom
    return float(center - half), float(center + half)


def point_order_hash(name, ids, eta):
    """Same canonical point-order hash as the T1 cache builder."""
    hh = hashlib.sha256()
    hh.update(name.encode())
    hh.update(np.asarray(ids, dtype=np.int64).tobytes())
    hh.update(np.asarray(eta, dtype=np.float64).tobytes())
    return hh.hexdigest()


def sha256_file(path):
    hh = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            hh.update(block)
    return hh.hexdigest()


# ---------------------------------------------------------------------------
# calibration driver
# ---------------------------------------------------------------------------

def _arm_list(eps_ref):
    """Frozen arm schedule: 4 null, 7x4 spatial injections, 6 velocity
    validation injections, 2 non-gradient pressure arms."""
    arms = [{"name": n, "kind": "null", "family": None, "m": None, "eps": None}
            for n in ("null", "wt", "noise_med", "noise_p99")]
    for m in range(7):
        for rel in (0.25, 0.5, 1.0, 2.0):
            arms.append({"name": f"S_m{m}_e{rel:g}", "kind": "inject",
                         "family": "S", "m": m, "eps": rel * eps_ref["S"]})
    for fam in ("V1", "V2"):
        for rel in (0.5, 1.0, 2.0):
            arms.append({"name": f"{fam}_m0_e{rel:g}", "kind": "inject",
                         "family": fam, "m": 0, "eps": rel * eps_ref[fam]})
    for rel in (1.0, 2.0):
        arms.append({"name": f"NG_m1_e{rel:g}", "kind": "pressure",
                     "family": "NG", "m": 1, "eps": rel * eps_ref["NG"]})
    return arms


def _matched_column(family):
    return {"S": 0, "V1": N_FAMILY, "V2": 2 * N_FAMILY, "NG": 1}[family]


def calibrate(out_dir, n_replicates=N_REPLICATES):
    from scipy.stats import chi2, norm
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    expect = quadrature_expectations()
    e_s, e_v1, e_w = expect

    # pilot replicate -> per-family eps_ref (2 SE of the matched statistic)
    z0, w0 = sample_mock_window(np.random.default_rng(SEED_MOCK))
    G0 = statistic_matrix(z0, mock_score(z0))
    t0, cov0 = weighted_stats(G0, w0)
    se0 = np.sqrt(np.diag(cov0))
    eps_ref = {"S": 2.0 * se0[0] / abs(e_s[0]),
               "V1": 2.0 * se0[N_FAMILY] / abs(e_v1),
               "V2": 2.0 * se0[2 * N_FAMILY] / abs(e_w),
               "NG": 2.0 * se0[1] / abs(e_w)}
    arms = _arm_list(eps_ref)
    n_arms = len(arms)
    print(f"[t2.cal] arms={n_arms} replicates={n_replicates} band_n={BAND_N} "
          f"eps_ref={ {k: round(v, 5) for k, v in eps_ref.items()} }")

    t_rep = np.zeros((n_arms, n_replicates, K_ALL), dtype=np.float32)
    z_rep = np.zeros_like(t_rep)
    Q_rep = np.zeros((n_arms, n_replicates, 3), dtype=np.float32)
    maxT_rep = np.zeros((n_arms, n_replicates), dtype=np.float32)
    seeds = np.random.SeedSequence(SEED_MOCK).spawn(n_replicates)
    arm_idx = {a["name"]: i for i, a in enumerate(arms)}

    for rep in range(n_replicates):
        rng = np.random.default_rng(seeds[rep])
        z, w_unit = sample_mock_window(rng)
        w_ln = np.exp(LOGNORMAL_WEIGHT_SIGMA * rng.standard_normal(len(z)))
        w_ln /= w_ln.mean()
        s_true = mock_score(z)
        tau = NOISE_REL_MEDIAN * np.median(np.abs(s_true), axis=0)
        noise = rng.standard_normal((len(z), 6)) * tau
        for a in arms:
            if a["name"] == "wt":
                weights = w_ln
            else:
                weights = w_unit
            score = s_true
            if a["name"] == "noise_med":
                score = s_true + noise
            elif a["name"] == "noise_p99":
                score = s_true + noise * (NOISE_REL_P99 / NOISE_REL_MEDIAN)
            elif a["kind"] in ("inject", "pressure"):
                score = s_true + injection_delta(z, a["family"], a["m"], a["eps"])
            G = statistic_matrix(z, score)
            t, cov = weighted_stats(G, weights)
            zs = z_scores(t, cov)
            i = arm_idx[a["name"]]
            t_rep[i, rep] = t
            z_rep[i, rep] = zs
            Q_rep[i, rep] = family_Q(t, cov)
            maxT_rep[i, rep] = np.max(np.abs(zs))
        if (rep + 1) % 40 == 0:
            print(f"[t2.cal] replicate {rep + 1}/{n_replicates} "
                  f"({time.time() - t_start:.0f}s)")

    # ---- null summaries: FPR with replicate (Wilson) uncertainty ----
    crit_z = {a: float(norm.ppf(1.0 - a / 2.0)) for a in ALPHAS}
    crit_chi2 = {a: float(chi2.ppf(1.0 - a, N_FAMILY)) for a in ALPHAS}
    half = n_replicates // 2
    summaries = {}
    for a in arms:
        i = arm_idx[a["name"]]
        s = {"kind": a["kind"], "family": a["family"], "m": a["m"],
             "eps": a["eps"], "n_replicates": n_replicates}
        per_stat_fpr = np.mean(np.abs(z_rep[i]) > crit_z[0.05], axis=0)
        s["fpr_per_stat_mean_0.05"] = float(per_stat_fpr.mean())
        s["fpr_per_stat_range_0.05"] = [float(per_stat_fpr.min()), float(per_stat_fpr.max())]
        k_mean = round(s["fpr_per_stat_mean_0.05"] * n_replicates)
        lo95, hi95 = wilson_interval(k_mean, n_replicates)
        s["fpr_wilson_95_0.05"] = [lo95, hi95]
        if a["kind"] == "null":
            s["max_abs_mean_z"] = float(np.max(np.abs(z_rep[i].mean(axis=0))))
            crit_half = float(np.quantile(maxT_rep[i, :half], 0.95))
            s["fpr_maxT_splithalf_0.05"] = float(np.mean(maxT_rep[i, half:] > crit_half))
            famq_max = Q_rep[i].max(axis=1)
            crit_famq = float(np.quantile(famq_max[:half], 0.95))
            s["fpr_familyQ_empirical_0.05"] = float(np.mean(famq_max[half:] > crit_famq))
            s["fpr_familyQ_chi2_0.05"] = float(
                np.mean(np.any(Q_rep[i] > crit_chi2[0.05], axis=1)))
            s["mean_family_Q"] = Q_rep[i].mean(axis=0).tolist()
            s["coverage_95_mean"] = float(np.mean(np.abs(z_rep[i]) <= crit_z[0.05]))
        if a["kind"] in ("inject", "pressure"):
            col = _matched_column(a["family"])
            s["power_matched_0.05"] = float(np.mean(np.abs(z_rep[i, :, col]) > crit_z[0.05]))
            jnull = arm_idx["null"]
            crit_null = float(np.quantile(maxT_rep[jnull, :half], 0.95))
            famq_null = Q_rep[jnull].max(axis=1)
            crit_famq_null = float(np.quantile(famq_null[:half], 0.95))
            s["power_joint_maxT_0.05"] = float(np.mean(maxT_rep[i] > crit_null))
            s["power_joint_familyQ_0.05"] = float(
                np.mean(np.any(Q_rep[i] > crit_chi2[0.05], axis=1)))
            s["power_joint_familyQ_emp_0.05"] = float(
                np.mean(Q_rep[i].max(axis=1) > crit_famq_null))
            pred = {"S": a["eps"] * e_s[a["m"]], "V1": a["eps"] * e_v1,
                    "V2": a["eps"] * e_w, "NG": None}[a["family"]]
            if pred is not None:
                emp = float(t_rep[i, :, col].mean())
                se_emp = float(t_rep[i, :, col].std(ddof=1) / np.sqrt(n_replicates))
                s["mean_shift_analytic"] = pred
                s["mean_shift_mc"] = emp
                s["mean_shift_consistency_sigma"] = (emp - pred) / se_emp if se_emp > 0 else np.inf
        summaries[a["name"]] = s

    # ---- cluster / block bootstrap sensitivity on null replicates ----
    sens_rng = np.random.default_rng(SEED_BOOT)
    ratios_cl, ratios_bl, cover_cl, cover_bl, cover_iid = [], [], [], [], []
    sens_seeds = np.random.SeedSequence(SEED_BOOT + 1).spawn(N_SENS_REPLICATES)
    cols = [_matched_column("S"), _matched_column("V1"), _matched_column("V2")]
    for rep in range(N_SENS_REPLICATES):
        z, w = sample_mock_window(np.random.default_rng(sens_seeds[rep]))
        G = statistic_matrix(z, mock_score(z))
        t, cov = weighted_stats(G, w)
        boot_iid = bootstrap_t(G, w, N_SENS_BOOT, sens_rng)
        boot_cl = bootstrap_t(G, w, N_SENS_BOOT, sens_rng, angular_cluster_ids(z))
        boot_bl = bootstrap_t(G, w, N_SENS_BOOT, sens_rng, radial_block_ids(z))
        se_sandwich = np.sqrt(np.diag(cov))
        for c in cols:
            q = np.quantile(boot_iid[:, c], [0.025, 0.975])
            qcl = np.quantile(boot_cl[:, c], [0.025, 0.975])
            qbl = np.quantile(boot_bl[:, c], [0.025, 0.975])
            ratios_cl.append((qcl[1] - qcl[0]) / max(q[1] - q[0], 1e-300))
            ratios_bl.append((qbl[1] - qbl[0]) / max(q[1] - q[0], 1e-300))
            cover_cl.append(qcl[0] <= 0.0 <= qcl[1])
            cover_bl.append(qbl[0] <= 0.0 <= qbl[1])
            cover_iid.append(q[0] <= 0.0 <= q[1])
    sens = {"n_replicates": N_SENS_REPLICATES, "n_boot": N_SENS_BOOT,
            "angular_clusters": list(ANGULAR_CLUSTERS), "radial_blocks": RADIAL_BLOCKS,
            "width_ratio_cluster_over_iid": float(np.mean(ratios_cl)),
            "width_ratio_block_over_iid": float(np.mean(ratios_bl)),
            "coverage95_cluster": float(np.mean(cover_cl)),
            "coverage95_block": float(np.mean(cover_bl)),
            "coverage95_iid": float(np.mean(cover_iid))}

    # ---- calibration gates (card stop condition: fix the test first) ----
    null_names = [a["name"] for a in arms if a["kind"] == "null"]
    g1 = all(summaries[n]["max_abs_mean_z"] <= 4.0 / np.sqrt(n_replicates) + 0.02
             for n in null_names)
    g2 = all(0.02 <= summaries[n]["fpr_per_stat_mean_0.05"] <= 0.10 for n in null_names)
    g3 = all(0.015 <= summaries[n]["fpr_maxT_splithalf_0.05"] <= 0.12
             and 0.015 <= summaries[n]["fpr_familyQ_empirical_0.05"] <= 0.12
             for n in null_names)
    g4 = all(summaries[n]["coverage_95_mean"] >= 0.90 for n in null_names)
    s_hi, s_lo = summaries["S_m0_e2"], summaries["S_m0_e0.25"]
    g5 = (s_hi["power_matched_0.05"] >= 0.90) and (s_lo["power_matched_0.05"] <= 0.60)
    g6 = max(abs(summaries[n]["mean_shift_consistency_sigma"])
             for n in summaries if "mean_shift_consistency_sigma" in summaries[n]) <= 5.0
    g7 = (sens["coverage95_cluster"] >= 0.85) and (sens["coverage95_block"] >= 0.85)
    gates = {"g1_h0_bias": bool(g1), "g2_fpr_univariate": bool(g2),
             "g3_fpr_joint": bool(g3), "g4_coverage": bool(g4),
             "g5_power_matched": bool(g5), "g6_mean_shift_analytic": bool(g6),
             "g7_resample_sensitivity": bool(g7)}
    gates["all_ok"] = all(gates.values())

    metrics = {
        "schema": "dpjax.nf-score-audit.t2-weak-score-calibration.v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "library": library_summary(),
        "mock": {"a_q": A_Q, "beta_sigma": BETA_SIGMA, "sigma0": SIGMA0,
                 "band_edges_q": [list(b) for b in BAND_EDGES_Q],
                 "band_fractions_frozen": list(BAND_FRACTIONS),
                 "band_n": list(BAND_N), "seed": SEED_MOCK,
                 "note": "heldout-sized detection-window analog; equal-mass (weights=1) "
                         "plus a lognormal weight machinery arm"},
        "eps_reference": {k: float(v) for k, v in eps_ref.items()},
        "quadrature_units": {"e_s": e_s.tolist(), "e_v1": e_v1, "e_w": e_w},
        "n_replicates": n_replicates,
        "arms": summaries,
        "resampling_sensitivity": sens,
        "gates": gates,
        "joint_reference_note": "nominal chi2_21 family-Q is ANTI-CONSERVATIVE at "
                                "these settings (calibration FPR ~2.3-2.6x nominal); the "
                                "gated joint procedure uses split-half empirical critical "
                                "values (mock) and bootstrap critical values (real stage); "
                                "chi2 FPRs are retained as diagnostics",
        "runtime_s": time.time() - t_start,
        "auriga_status": "real-sample estimates gated on local certified T1 cache "
                         "(see real stage); all Auriga results exploratory",
    }
    np.savez_compressed(
        out_dir / "weak_score_tests_calibration.npz",
        arms_json=json.dumps(arms), t_rep=t_rep, z_rep=z_rep, Q_rep=Q_rep,
        maxT_rep=maxT_rep, norms=np.array(NORMS),
        library_json=json.dumps(library_summary()))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))
    for name in null_names:
        s = summaries[name]
        print(f"[t2.cal] null {name}: FPRuv {s['fpr_per_stat_mean_0.05']:.3f} "
              f"maxT {s['fpr_maxT_splithalf_0.05']:.3f} famQemp {s['fpr_familyQ_empirical_0.05']:.3f} "
              f"famQchi2 {s['fpr_familyQ_chi2_0.05']:.3f} "
              f"cov95 {s['coverage_95_mean']:.3f} bias {s['max_abs_mean_z']:.3f}")
    print(f"[t2.cal] S_m0 power @0.25/1/2 eps_ref: "
          f"{summaries['S_m0_e0.25']['power_matched_0.05']:.2f}/"
          f"{summaries['S_m0_e1']['power_matched_0.05']:.2f}/"
          f"{s_hi['power_matched_0.05']:.2f}")
    print(f"[t2.cal] resampling: cluster/iid width {sens['width_ratio_cluster_over_iid']:.2f} "
          f"block/iid {sens['width_ratio_block_over_iid']:.2f} cov {sens['coverage95_cluster']:.2f}")
    print(f"[t2.cal] CALIBRATION GATES all_ok: {gates['all_ok']} {gates}")
    return metrics


# ---------------------------------------------------------------------------
# real-sample stage (EXPLORATORY; requires the certified T1 cache locally)
# ---------------------------------------------------------------------------

def _budget_alarm(budget_s, what):
    """Portable wall-clock budget (macOS has no GNU timeout)."""
    def handler(signum, frame):
        raise TimeoutError(f"{what} exceeded budget of {budget_s}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(budget_s))


def _clear_alarm():
    signal.alarm(0)

def real_estimate(cache_dir, registry_path, out_dir, n_bootstrap=N_BOOTSTRAP):
    import h5py
    from scipy.stats import chi2, norm
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    points_path = cache_dir / "points_heldout.h5"
    arrays_path = cache_dir / "arrays_heldout.h5"
    if not (points_path.exists() and arrays_path.exists()):
        print("[t2.real] REAL-SAMPLE STAGE SKIPPED: certified cache files absent "
              f"({points_path}, {arrays_path}); no Auriga numbers are reported")
        return None
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise RuntimeError(f"T1 cache present but registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for path in (points_path, arrays_path):
        got = sha256_file(path)
        want = registry["files_sha256"][path.name]
        if got != want:
            raise RuntimeError(f"T1 cache sha256 mismatch for {path.name}: "
                               f"{got} != {want} (card T2 stop condition)")

    with h5py.File(points_path, "r") as f:
        particle_id = f["particle_id"][:].astype(np.int64)
        eta = f["eta"][:].astype(np.float64)
        weights = f["weights"][:].astype(np.float64)
        attr_hash = str(f.attrs["point_order_sha256"])
    with h5py.File(arrays_path, "r") as f:
        score = f["score_f32"][:].astype(np.float64)
        attr_hash_arrays = str(f.attrs["point_order_sha256"])
    want_hash = registry["point_order_sha256"]["heldout"]
    recomputed = point_order_hash("heldout", particle_id, eta)
    if not (attr_hash == want_hash and attr_hash_arrays == want_hash and recomputed == want_hash):
        raise RuntimeError("heldout point-order hash mismatch vs registry "
                           f"({attr_hash}, {attr_hash_arrays}, {recomputed} != {want_hash})")
    if score.shape != eta.shape:
        raise RuntimeError(f"cache shape mismatch: score {score.shape} vs eta {eta.shape}")

    r = np.linalg.norm(eta[:, :3], axis=1)
    window = (r >= W_LO) & (r <= W_HI)
    z = eta[window]
    w = weights[window]
    s_model = score[window]
    n_eff = float(np.sum(w, dtype=np.float64) ** 2 / np.sum(w ** 2, dtype=np.float64))
    print(f"[t2.real] EXPLORATORY heldout rows total={len(eta)} window={window.sum()} "
          f"n_eff={n_eff:.0f} weight mean={w.mean():.4f}")

    rng = np.random.default_rng(SEED_BOOT)
    G = statistic_matrix(z, s_model)
    t, cov = weighted_stats(G, w)
    zs = z_scores(t, cov)
    Q = family_Q(t, cov)
    boot = {"iid": bootstrap_t(G, w, n_bootstrap, rng),
            "angular_cluster": bootstrap_t(G, w, n_bootstrap, rng, angular_cluster_ids(z)),
            "radial_block": bootstrap_t(G, w, n_bootstrap, rng, radial_block_ids(z))}
    maxt_p = {}
    for scheme, bt in boot.items():
        _, stat = maxT_from_bootstrap(bt, t)
        maxt_p[scheme] = float(np.mean(np.max(np.abs(bt - bt.mean(axis=0, keepdims=True))
                                               / np.maximum(bt.std(axis=0, ddof=1), 1e-300),
                                               axis=1) >= np.max(stat)))
    famq_boot_p = {}
    for scheme, bt in boot.items():
        centered = bt - bt.mean(axis=0, keepdims=True)
        per_family = []
        for f in range(3):
            sl = slice(f * N_FAMILY, (f + 1) * N_FAMILY)
            cf = np.ascontiguousarray(cov[sl, sl])
            shrunk = (1.0 - SHRINKAGE) * cf + SHRINKAGE * np.diag(np.diag(cf))
            inv = np.linalg.inv(shrunk)
            qb = np.einsum("bi,ij,bj->b", centered[:, sl], inv, centered[:, sl])
            per_family.append(float(np.mean(qb >= Q[f])))
        famq_boot_p[scheme] = per_family
    crit_chi2 = float(chi2.ppf(0.95, N_FAMILY))

    bands = {"window_all": (W_LO, W_HI), "band_30_45": (3.0, 4.5),
             "band_45_60": (4.5, 6.0), "band_60_70": (6.0, 7.0)}
    band_rows = {}
    for bname, (lo, hi) in bands.items():
        if bname == "window_all":
            zb, wb, sb, Gb = z, w, s_model, G
        else:
            rb = np.linalg.norm(z[:, :3], axis=1)
            sel = (rb >= lo) & (rb < hi) if hi < W_HI else (rb >= lo) & (rb <= hi)
            zb, wb, sb = z[sel], w[sel], s_model[sel]
            Gb = statistic_matrix(zb, sb)
        tb, covb = weighted_stats(Gb, wb)
        neff_b = float(np.sum(wb) ** 2 / np.sum(wb ** 2))
        bt = bootstrap_t(Gb, wb, max(n_bootstrap // 4, 200), rng)
        lo95 = np.quantile(bt, 0.025, axis=0)
        hi95 = np.quantile(bt, 0.975, axis=0)
        band_rows[bname] = {"n": int(len(wb)), "n_eff": neff_b,
                            "t": tb, "se": np.sqrt(np.diag(covb)),
                            "z": z_scores(tb, covb),
                            "boot_lo95": lo95, "boot_hi95": hi95}
        print(f"[t2.real] EXPLORATORY {bname}: n={len(wb)} n_eff={neff_b:.0f} "
              f"maxz={np.max(np.abs(band_rows[bname]['z'])):.2f}")

    p_uni = 2.0 * norm.sf(np.abs(zs))
    q_p = float(chi2.sf(Q[0], N_FAMILY))
    discovery = {"maxT_boot_iid_p": maxt_p["iid"],
                 "maxT_boot_cluster_p": maxt_p["angular_cluster"],
                 "maxT_boot_block_p": maxt_p["radial_block"],
                 "familyQ_S_vs_chi2_0.05": bool(Q[0] > crit_chi2)}
    metrics = {
        "schema": "dpjax.nf-score-audit.t2-weak-score-real.v1",
        "grade": "EXPLORATORY (t0-manifest: no qualified confirmation set; "
                 "model/hyperparameters selected on the overlapping population)",
        "cache": {"points_sha256": registry["files_sha256"]["points_heldout.h5"],
                  "arrays_sha256": registry["files_sha256"]["arrays_heldout.h5"],
                  "point_order_sha256": registry["point_order_sha256"]["heldout"],
                  "t1_run": registry.get("run"), "t1_commit": registry.get("commit")},
        "n_window": int(window.sum()), "n_eff": n_eff,
        "t": t.tolist(), "se": np.sqrt(np.diag(cov)).tolist(), "z": zs.tolist(),
        "p_univariate_normal": p_uni.tolist(),
        "family_Q": Q.tolist(),
        "family_Q_p_chi2": [float(chi2.sf(q, N_FAMILY)) for q in Q],
        "family_Q_p_bootstrap": famq_boot_p,
        "maxT_p_values": maxt_p,
        "discovery_rule": "pre-registered: flag if min bootstrap maxT p < 0.05 "
                          "AND min bootstrap family-Q p < 0.05 (chi2_21 nominal retained "
                          "as diagnostic only: calibration showed it anticonservative)",
        "discovery": discovery,
        "flagged": bool(min(maxt_p.values()) < 0.05
                        and min(min(v) for v in famq_boot_p.values()) < 0.05),
        "bands": {k: {"n": v["n"], "n_eff": v["n_eff"],
                      "max_abs_z": float(np.max(np.abs(v["z"]))),
                      "t": v["t"].tolist(), "se": v["se"].tolist(),
                      "boot_lo95": v["boot_lo95"].tolist(),
                      "boot_hi95": v["boot_hi95"].tolist()}
                  for k, v in band_rows.items()},
        "caveats": ["stored f32 score with default-ODE tolerances; T1 outer-band "
                    "chain error median ~2.7e-3 / p99 ~0.03 floored-rel (2-4x the "
                    "inner bands) bounds the achievable effect scale",
                    "mock-calibrated critical values transfer the Plummer analog "
                    "radial law; bootstrap critical values are self-contained",
                    "band-level columns reuse the frozen global normalization"],
        "runtime_s": time.time() - t_start,
    }
    np.savez_compressed(
        out_dir / "weak_score_tests_real.npz",
        t=t, se=np.sqrt(np.diag(cov)), z=zs, p_uni=p_uni, family_Q=Q,
        boot_iid=boot["iid"], boot_cluster=boot["angular_cluster"],
        boot_block=boot["radial_block"],
        bands_json=json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "t" or True}
                               for k, v in band_rows.items()}, default=lambda o: o.tolist()))
    (out_dir / "metrics_real.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True))
    print(f"[t2.real] EXPLORATORY global: max|z|={np.max(np.abs(zs)):.2f} "
          f"familyQ_S={Q[0]:.1f} (chi2_21 95%={crit_chi2:.1f}) "
          f"maxT p(iid/cluster/block)={maxt_p['iid']:.3f}/{maxt_p['angular_cluster']:.3f}/"
          f"{maxt_p['radial_block']:.3f} flagged={metrics['flagged']}")
    print("[t2.real] EXPLORATORY done (grade: exploratory; no confirmation-set status)")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cal = sub.add_parser("calibrate", help="mock calibration with gates")
    p_cal.add_argument("--out-dir", default="runs/nf-score-audit/t2-weak-score")
    p_cal.add_argument("--replicates", type=int, default=N_REPLICATES)
    p_cal.add_argument("--budget-s", type=int, default=7200)
    p_real = sub.add_parser("real", help="exploratory real-heldout estimates "
                          "(skipped cleanly if the certified cache is not local)")
    p_real.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p_real.add_argument("--registry", default=T1_REGISTRY)
    p_real.add_argument("--out-dir", default="runs/nf-score-audit/t2-weak-score")
    p_real.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    p_real.add_argument("--budget-s", type=int, default=3600)
    sub.add_parser("library", help="print the frozen test-library summary")
    args = parser.parse_args()
    if args.cmd == "library":
        print(json.dumps(library_summary(), indent=1))
        return 0
    if args.cmd == "calibrate":
        _budget_alarm(args.budget_s, "calibration")
        try:
            metrics = calibrate(args.out_dir, n_replicates=args.replicates)
        finally:
            _clear_alarm()
        return 0 if metrics["gates"]["all_ok"] else 3
    _budget_alarm(args.budget_s, "real stage")
    try:
        metrics = real_estimate(args.cache_dir, args.registry, args.out_dir,
                                n_bootstrap=args.bootstrap)
    finally:
        _clear_alarm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
