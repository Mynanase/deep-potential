"""Analytic tests for the T2 weak Stein score machinery (card T2)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "auriga"))

import nf_weak_score_tests as t2  # noqa: E402


def test_window_bump_properties():
    r = np.linspace(2.0, 8.0, 4001)
    w, dw = t2.window(r)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert np.all(w >= 0.0) and np.all(w <= 1.0 + 1e-12)
    assert abs(w[np.argmin(abs(r - 5.0))] - 1.0) < 1e-3
    assert np.all(w[~((r >= 3.0) & (r <= 7.0))] == 0.0)
    inside = (r > 3.05) & (r < 6.95)
    fd = np.gradient(w, r)
    assert np.max(np.abs(dw[inside] - fd[inside])) < 2e-3


def test_u_basis_derivative_matches_finite_differences():
    r = np.linspace(3.05, 6.95, 2001)
    u, du = t2.u_basis(r)
    for m in range(7):
        fd = np.gradient(u[:, m], r)
        assert np.max(np.abs(du[:, m] - fd)) < 2e-2 * max(1.0, np.max(np.abs(du[:, m])))


def test_mock_score_is_gradient_of_logF():
    rng = np.random.default_rng(11)
    z = t2.sample_mock_window(rng)[0]
    z = z[::40]
    s = t2.mock_score(z)
    h = 1e-5
    for comp in range(6):
        zp, zm = z.copy(), z.copy()
        zp[:, comp] += h
        zm[:, comp] -= h
        fd = (t2.mock_logF(zp) - t2.mock_logF(zm)) / (2 * h)
        rel = np.abs(s[:, comp] - fd) / np.maximum(np.abs(fd), 0.1)
        assert np.median(rel) < 1e-5, f"component {comp}: median rel {np.median(rel)}"


def test_divergence_formula_matches_finite_differences():
    rng = np.random.default_rng(12)
    z = t2.sample_mock_window(rng)[0][::97]
    zero_score = np.zeros_like(z)
    G = t2.statistic_matrix(z, zero_score)  # divergence part only
    h = 1e-5
    fams = [("S", j, m) for j in range(3) for m in (0, 3)] + [("V", 0, 0), ("W", 1, 2)]
    for fam, j, m in fams:
        fam_idx = {"S": 0, "V": 1, "W": 2}[fam]
        col = fam_idx * t2.N_FAMILY + j * 7 + m
        num = np.zeros(len(z))
        for k in range(6):
            zp, zm = z.copy(), z.copy()
            zp[:, k] += h
            zm[:, k] -= h
            num += (t2.h_field(zp, fam, j, m)[:, k] - t2.h_field(zm, fam, j, m)[:, k]) / (2 * h)
        rel = np.abs(num - G[:, col]) / np.maximum(np.abs(num), 0.05)
        assert np.median(rel) < 5e-4, f"{fam} j={j} m={m}: {np.median(rel)}"


def test_plummer_cdf_roundtrip_and_band_fractions():
    x = np.array([0.1, 0.5, 1.0, 3.0, 10.0])
    m = t2._plummer_mass_fraction(x)
    back = t2._invert_mass_fraction(m, 0.0, 60.0)
    assert np.max(np.abs(back - x)) < 1e-8
    rng = np.random.default_rng(t2.SEED_MOCK)
    z = t2.sample_full_mock(rng, n=200000)
    r_kpc = np.linalg.norm(z[:, :3], axis=1) * t2.L_KPC
    got = [np.mean((r_kpc >= lo) & (r_kpc < hi)) for lo, hi in ((30, 45), (45, 60), (60, 70))]
    want = np.array(t2.BAND_FRACTIONS)
    se = np.sqrt(want * (1 - want) / len(z))
    assert np.all(np.abs(np.array(got) - want) <= 0.25 * want + 4 * se)


def test_stein_null_all_statistics_zero_mean():
    rng = np.random.default_rng(21)
    z, w = t2.sample_mock_window(rng)
    G = t2.statistic_matrix(z, t2.mock_score(z))
    t, cov = t2.weighted_stats(G, w)
    zs = t2.z_scores(t, cov)
    worst = np.max(np.abs(zs))
    assert worst < 6.0, f"max |z| under the exact-score null = {worst}"


def test_weighted_stats_manual_and_weight_layers():
    G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    w = np.array([0.5, 1.0, 1.5])
    t, cov = t2.weighted_stats(G, w)
    assert np.allclose(t, [11.0 / 3.0, 14.0 / 3.0])
    manual = np.array([[56.0 / 81.0, 56.0 / 81.0], [56.0 / 81.0, 56.0 / 81.0]])
    assert np.allclose(cov, manual, atol=1e-12)
    t_eq, _ = t2.weighted_stats(G, np.ones(3))
    assert np.allclose(t_eq, G.mean(axis=0))
    rng = np.random.default_rng(22)
    z, _ = t2.sample_mock_window(rng)
    Gm = t2.statistic_matrix(z, t2.mock_score(z))
    wr = np.exp(0.3 * rng.standard_normal(len(z)))
    wr /= wr.mean()
    tw, covw = t2.weighted_stats(Gm, wr)
    zsw = t2.z_scores(tw, covw)
    assert np.max(np.abs(zsw)) < 6.0


def test_injection_mean_shifts_match_quadrature():
    e_s, e_v1, e_w = t2.quadrature_expectations()
    rng = np.random.default_rng(23)
    z, w = t2.sample_mock_window(rng)
    s_true = t2.mock_score(z)
    G0 = t2.statistic_matrix(z, s_true)
    t_null, _ = t2.weighted_stats(G0, w)
    for fam, m, unit, col in [("S", 0, e_s[0], 0), ("V1", 0, e_v1, t2.N_FAMILY),
                              ("V2", 0, e_w, 2 * t2.N_FAMILY)]:
        eps = 0.05
        G = t2.statistic_matrix(z, s_true + t2.injection_delta(z, fam, m, eps))
        t_inj, cov = t2.weighted_stats(G, w)
        shift_mc = t_inj[col] - t_null[col]
        se = np.sqrt(cov[col, col])
        assert abs(shift_mc - eps * unit) < 6 * se, f"{fam}: {shift_mc} vs {eps * unit}"


def test_nongradient_pressure_has_nonzero_projection():
    rng = np.random.default_rng(24)
    z, w = t2.sample_mock_window(rng)
    s_true = t2.mock_score(z)
    eps = 2.0
    G = t2.statistic_matrix(z, s_true + t2.injection_delta(z, "NG", 1, eps))
    t, cov = t2.weighted_stats(G, w)
    col = 1  # S_{x,1}
    se = np.sqrt(cov[col, col])
    assert t[col] / se > 4.0


def test_family_Q_null_is_chi2_like():
    rng = np.random.default_rng(25)
    qs = []
    for _ in range(30):
        z, w = t2.sample_mock_window(rng)
        G = t2.statistic_matrix(z, t2.mock_score(z))
        t, cov = t2.weighted_stats(G, w)
        qs.append(t2.family_Q(t, cov))
    qs = np.array(qs)
    assert np.all(qs.mean(axis=0) > 0.6 * t2.N_FAMILY)
    assert np.all(qs.mean(axis=0) < 1.6 * t2.N_FAMILY)


def test_angular_clusters_and_radial_blocks_partition():
    rng = np.random.default_rng(26)
    z, _ = t2.sample_mock_window(rng)
    ca = t2.angular_cluster_ids(z)
    n_cells = t2.ANGULAR_CLUSTERS[0] * t2.ANGULAR_CLUSTERS[1]
    assert ca.min() >= 0 and ca.max() < n_cells
    assert np.min(np.bincount(ca, minlength=n_cells)) > 100
    rb = t2.radial_block_ids(z)
    assert rb.min() >= 0 and rb.max() < t2.RADIAL_BLOCKS


def test_bootstrap_iid_matches_direct_resampling():
    rng = np.random.default_rng(27)
    G = rng.standard_normal((500, 4)).astype(np.float32)
    w = np.ones(500)
    bt = t2.bootstrap_t(G, w, 300, rng)
    assert np.max(np.abs(bt.mean(axis=0) - G.mean(axis=0))) < 0.02
    expected_sd = G.std(axis=0, ddof=1) / np.sqrt(500)
    assert np.allclose(bt.std(axis=0), expected_sd, rtol=0.15)


def test_maxT_statistic_is_distance_from_null_not_from_bootstrap_mean():
    rng = np.random.default_rng(28)
    n_boot, k = 4000, 5
    boot = rng.standard_normal((n_boot, k)) * 0.01
    t_null = np.zeros(k)
    t_far = np.full(k, 0.03)  # 3 sd from the null value
    _, stat_null = t2.maxT_from_bootstrap(boot, t_null)
    _, stat_far = t2.maxT_from_bootstrap(boot, t_far)
    zboot, _ = t2.maxT_from_bootstrap(boot, t_null)
    crit = np.quantile(np.max(zboot, axis=1), 0.95)
    # regression: the observed statistic must be |t|/sd (null value 0), not
    # |t - mean(boot)|/sd -- the vacuous form that produced p = 1.0 on real data
    assert np.max(stat_null) < 0.5
    assert np.max(stat_far) > 2.8
    assert np.max(stat_far) > crit
    assert np.max(stat_null) < crit


def test_real_stage_refuses_hash_mismatch_and_skips_when_absent(tmp_path):
    assert t2.real_estimate(tmp_path / "empty", tmp_path / "reg.json",
                            tmp_path / "out") is None
    reg = {"files_sha256": {"points_heldout.h5": "0" * 64,
                            "arrays_heldout.h5": "0" * 64},
           "point_order_sha256": {"heldout": "0" * 64}}
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "points_heldout.h5").write_bytes(b"not an h5 file")
    (cache / "arrays_heldout.h5").write_bytes(b"not an h5 file")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        t2.real_estimate(cache, tmp_path / "reg.json", tmp_path / "out")


def test_real_stage_end_to_end_on_synthetic_cache(tmp_path):
    import h5py
    rng = np.random.default_rng(31)
    n = 4000
    r = rng.uniform(3.0, 7.0, n)
    direction = rng.standard_normal((n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    q = r[:, None] * direction
    p = 0.4 * rng.standard_normal((n, 3))
    z = np.concatenate([q, p], axis=1)
    ids = np.arange(n, dtype=np.int64)
    weights = np.exp(0.2 * rng.standard_normal(n))
    weights /= weights.mean()
    order = t2.point_order_hash("heldout", ids, z)
    cache = tmp_path / "cache"
    cache.mkdir()
    with h5py.File(cache / "points_heldout.h5", "w") as f:
        f.attrs["point_order_sha256"] = order
        for key, val in [("particle_id", ids), ("eta", z), ("weights", weights)]:
            f.create_dataset(key, data=val)
    with h5py.File(cache / "arrays_heldout.h5", "w") as f:
        f.attrs["point_order_sha256"] = order
        f.create_dataset("score_f32", data=t2.mock_score(z))
    reg = {"files_sha256": {
               "points_heldout.h5": t2.sha256_file(cache / "points_heldout.h5"),
               "arrays_heldout.h5": t2.sha256_file(cache / "arrays_heldout.h5")},
           "point_order_sha256": {"heldout": order},
           "run": "synthetic", "commit": "synthetic"}
    (tmp_path / "reg.json").write_text(json.dumps(reg))
    metrics = t2.real_estimate(cache, tmp_path / "reg.json", tmp_path / "out", n_bootstrap=120)
    assert metrics is not None
    assert metrics["grade"].startswith("EXPLORATORY")
    assert metrics["n_window"] == n
    assert len(metrics["t"]) == t2.K_ALL
    assert set(metrics["maxT_p_values"]) == {"iid", "angular_cluster", "radial_block"}
    assert len(metrics["family_Q_p_bootstrap"]) == 3


def test_point_order_hash_matches_t1_canonical_form():
    ids = np.array([3, 1, 2], dtype=np.int64)
    eta = np.array([[0.5, 0.0, 0.0, 0.1, 0.0, 0.0]], dtype=np.float64)
    hh = hashlib.sha256()
    hh.update(b"heldout")
    hh.update(ids.astype(np.int64).tobytes())
    hh.update(eta.astype(np.float64).tobytes())
    assert t2.point_order_hash("heldout", ids, eta) == hh.hexdigest()
