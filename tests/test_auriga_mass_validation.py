#!/usr/bin/env python
"""Tests for the Halo12 enclosed-mass validation machinery.

These tests validate the VERIFIER (angular quadrature, radial integration,
unit conversion, truth-edge binding, inner-boundary subtraction) on analytic
examples.  They do not test the trained models.

Run from the repo root:  python -m pytest tests/ -q
CPU only, x64 forced where the machinery under test uses jax.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import validate_enclosed_mass as vem  # noqa: E402

L_KPC, V_KMS = 10.0, 100.0

TRUTH_EXISTS = (REPO / "data/auriga/halo_12_total_density.hdf5").exists()
needs_truth = pytest.mark.skipif(not TRUTH_EXISTS, reason="truth HDF5 not present")


@pytest.fixture(scope="module", autouse=True)
def _jax_x64():
    import jax
    jax.config.update("jax_enable_x64", True)
    yield


# ---------------------------------------------------------------- unit conversion

def test_rho_factor_units():
    """rho = V^2/(4 pi G L^2) * lap must convert dimensionless laplacian to
    Msun/kpc^3: check against the analytic Plummer central density."""
    phi, M, b = vem.plummer_phi_func()
    q = np.array([[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]])
    rho = vem.rho_from_phi(phi, q, L_KPC, V_KMS)
    rho_true = vem.plummer_density(0.0)
    assert np.allclose(rho, rho_true, rtol=1e-6)


def test_flux_mass_matches_analytic_total_at_large_radius():
    """M_flux(<r) -> M for r >> b; checks the r^2 V^2/(G L) prefactor."""
    phi, M, b = vem.plummer_phi_func()
    dirs = vem.sobol_directions(1024, scramble_seed=7)
    m = vem.m_flux_at_radius(phi, 500.0, dirs, L_KPC, V_KMS)
    assert abs(m / M - 1.0) < 1e-3


# ---------------------------------------------------------------- Plummer path

def test_plummer_volume_and_flux_within_threshold():
    phi, M, b = vem.plummer_phi_func()
    r_inner, r_outer = 0.5, 70.0
    r_nodes = vem.make_radial_nodes(r_inner, r_outer, per_interval=200)
    dirs = vem.sobol_directions(256, scramble_seed=0)

    import jax.numpy as jnp
    q = dirs[None, :, :] * (r_nodes[:, None, None] / L_KPC)
    lap = np.asarray(vem.laplacian_phi_batch(
        phi, jnp.asarray(q.reshape(-1, 3)))).reshape(q.shape[:2])
    rho_mean = lap.mean(axis=1) * V_KMS ** 2 / (
        4.0 * np.pi * vem.G_KPC_KMS2_MSUN * L_KPC ** 2)
    dM = vem.cumulative_volume_mass(r_nodes, rho_mean, r_inner)
    dM_true = vem.plummer_mass_inside(r_nodes) - vem.plummer_mass_inside(r_inner)
    body = dM_true > 0
    assert np.abs(dM[body] / dM_true[body] - 1.0).max() < vem.TH_PLUMMER_REL

    m_flux = np.array([vem.m_flux_at_radius(phi, r, dirs, L_KPC, V_KMS)
                       for r in r_nodes])
    dMf = vem.cumulative_flux_mass(m_flux)
    assert np.abs(dMf[body] / dM_true[body] - 1.0).max() < vem.TH_PLUMMER_REL


def test_inner_boundary_subtraction_consistency():
    """DeltaM(r;0.5) - DeltaM(2;0.5) == DeltaM(2;70) computed independently.
    The subtraction itself is an algebraic identity; the test's value is that
    the node construction places the inner boundary exactly and the direct
    integration from a=2 agrees with the differenced curve."""
    r_nodes = vem.make_radial_nodes(0.5, 70.0, per_interval=100,
                                    boundaries=(2.0,))
    rho = vem.plummer_density(r_nodes)
    dA = vem.cumulative_volume_mass(r_nodes, rho, 0.5)
    i2 = int(np.flatnonzero(np.isclose(r_nodes, 2.0))[0])
    dA_to_2 = dA[i2]
    dB = dA - dA_to_2          # DeltaM(r;2) on the shared node set
    # independent: integrate 2->70 directly
    r_nodes_b = vem.make_radial_nodes(2.0, 70.0, per_interval=100)
    rho_b = vem.plummer_density(r_nodes_b)
    dB_direct = vem.cumulative_volume_mass(r_nodes_b, rho_b, 2.0)
    assert np.abs(dB[-1] - dB_direct[-1]) / dB_direct[-1] < 1e-10


# ---------------------------------------------------------------- anisotropic

def test_anisotropic_mean_recovers_total_median_does_not():
    comps = ((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))
    m_true = vem.anisotropic_gauss_mass(comps)
    r_nodes = vem.make_radial_nodes(0.5, 70.0, per_interval=200)
    dirs = vem.sobol_directions(512, scramble_seed=0)
    q = dirs[None, :, :] * (r_nodes[:, None, None] / L_KPC)
    rho_dirs = vem.anisotropic_gauss_density(
        q.reshape(-1, 3), L_KPC, comps).reshape(q.shape[:2])
    dM = vem.cumulative_volume_mass(r_nodes, rho_dirs.mean(axis=1), 0.5)
    assert abs(dM[-1] / m_true - 1.0) < vem.TH_NONSPH_REL

    u = np.log(r_nodes)
    g = 4.0 * np.pi * r_nodes[None, :] ** 3 * rho_dirs.T
    inc = 0.5 * np.diff(u)[None, :] * (g[:, 1:] + g[:, :-1])
    m_end = np.cumsum(inc, axis=1)[:, -1]
    median_err = abs(np.median(m_end) / m_true - 1.0)
    mean_err = abs(m_end.mean() / m_true - 1.0)
    # the median of per-direction enclosed mass is NOT guaranteed to equal the
    # total; with this axis ratio it must be visibly biased
    assert median_err > 5 * mean_err


def test_anisotropic_independent_reference_agrees():
    comps = ((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))
    m_true = vem.anisotropic_gauss_mass(comps)
    from numpy.polynomial.legendre import leggauss
    s_nodes, ws = leggauss(200)
    s_kpc = 0.5 * (70.0 - 0.5) * s_nodes + 0.5 * (70.0 + 0.5)
    w_s = 0.5 * (70.0 - 0.5) * ws
    mean_ref = vem.anisotropic_gauss_sphere_mean(s_kpc, comps, n_mu=100)
    m_ref = float(np.sum(4.0 * np.pi * s_kpc ** 2 * mean_ref * w_s))
    assert abs(m_ref / m_true - 1.0) < vem.TH_NONSPH_REL


# ---------------------------------------------------------------- invariances

def test_constant_linear_terms_invariant():
    import jax.numpy as jnp
    phi0, M, b = vem.plummer_phi_func()

    def phi_pert(q):
        return phi0(q) + 123.0 + 0.3 * q[0] - 0.7 * q[1] + 0.15 * q[2]

    rng = np.random.default_rng(3)
    q = rng.normal(size=(256, 3)) * 3.0
    lap0 = np.asarray(vem.laplacian_phi_batch(phi0, jnp.asarray(q)))
    lap1 = np.asarray(vem.laplacian_phi_batch(phi_pert, jnp.asarray(q)))
    assert np.abs(lap1 - lap0).max() / np.abs(lap0).max() < 1e-12

    dirs = vem.sobol_directions(512, scramble_seed=0)
    lin = np.array([
        np.mean(np.asarray(vem.grad_phi_dot_n_batch(
            lambda qq: 0.3 * qq[0] - 0.7 * qq[1] + 0.15 * qq[2],
            jnp.asarray(dirs * (r / L_KPC)), jnp.asarray(dirs))))
        * r ** 2 * V_KMS ** 2 / (vem.G_KPC_KMS2_MSUN * L_KPC)
        for r in (1.0, 10.0, 70.0)])
    assert np.abs(lin).max() < vem.TH_PLUMMER_REL * vem.plummer_mass_inside(70.0)


# ---------------------------------------------------------------- truth binding

@needs_truth
def test_truth_mass_binds_to_outer_edges():
    truth = vem.load_truth()
    m_shell_reconstructed = np.diff(
        vem.plummer_mass_inside(truth["r_edges"]))
    m_cum = np.cumsum(m_shell_reconstructed)
    # pairing with outer edges reproduces the analytic cumulative mass:
    # m_cum[i] + M(<r_edges[0]) == M(<r_edges[i+1])
    assert np.allclose(
        m_cum + vem.plummer_mass_inside(truth["r_edges"][0]),
        vem.plummer_mass_inside(truth["r_edges"][1:]), rtol=1e-12)
    # pairing the same values with shell centres is a detectable mispairing:
    # |M(<center) - M(<edge)| / M(<edge) must exceed the threshold
    i = 20
    edge, center = truth["r_edges"][i + 1], truth["r_center"][i]
    rel = abs(vem.plummer_mass_inside(center)
              - vem.plummer_mass_inside(edge)) / vem.plummer_mass_inside(edge)
    assert rel > vem.TH_SHELL_MISMATCH


@needs_truth
def test_truth_delta_mass_inner_edge_required():
    truth = vem.load_truth()
    r, dM = vem.truth_delta_mass(truth, 0.5, truth["r_edges"][1:30])
    assert r[0] == pytest.approx(truth["r_edges"][1])
    assert dM[0] == pytest.approx(truth["M_shell_total"][0])
    with pytest.raises(ValueError):
        vem.truth_delta_mass(truth, 0.53, truth["r_edges"][1:5])


# ---------------------------------------------------------------- step 3 machinery

def test_sobol_directions_prefix_nested():
    """Nested-prefix contract (plan step 3): the first n points of a scramble
    must not change when more points are drawn from the same engine."""
    a = vem.sobol_directions(256, scramble_seed=0)
    b = vem.sobol_directions(1024, scramble_seed=0)
    assert a.shape == (256, 3)
    assert np.allclose(a, b[:256])
    # and independent scrambles differ
    c = vem.sobol_directions(256, scramble_seed=1)
    assert not np.allclose(a, c)


def test_grid_edge_indices_exact():
    edges = np.array([0.5, 1.3, 5.0, 20.0])
    nodes = vem.make_radial_nodes(0.5, 20.0, per_interval=3, boundaries=edges[1:])
    idx = vem.grid_edge_indices(nodes, edges)
    assert np.allclose(nodes[idx], edges, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        vem.grid_edge_indices(nodes, np.array([1.31]))


def test_per_direction_eval_and_prefix_stats_plummer():
    """End-to-end check of the step-3 evaluator on the spherical Plummer:
    every direction sees the same density; prefix flux mass matches the
    analytic M(<r) = M r^3/(r^2+b^2)^{3/2}."""
    import jax.numpy as jnp
    phi, M, b = vem.plummer_phi_func()
    r_nodes = vem.make_radial_nodes(0.5, 70.0, per_interval=100,
                                    boundaries=(5.0, 20.0))
    dirs = vem.sobol_directions(64, scramble_seed=0)
    rho_dir, dn_dir = vem.per_direction_rho_and_dn(phi, r_nodes, dirs, L_KPC,
                                                   V_KMS, chunk_nodes=3)
    assert rho_dir.shape == (64, r_nodes.size)
    # spherical: all directions identical up to autodiff noise
    assert np.abs(rho_dir - rho_dir[0][None, :]).max() < 1e-8 * np.abs(
        rho_dir).max()
    stats = vem.mass_stats_at_prefix(r_nodes, 0.5, rho_dir, dn_dir, L_KPC,
                                     V_KMS, n_dir=16)
    dM_true = vem.plummer_mass_inside(r_nodes) - vem.plummer_mass_inside(0.5)
    body = dM_true > 0        # dM is exactly 0 at r_inner (0/0 otherwise)
    # volume tolerance = TH_PLUMMER_REL (log-r trapezoid floor at this grid;
    # the quadrature itself is validated by the step-1 checks); flux uses
    # only first derivatives and is far more accurate
    assert np.abs(stats["dM_rho"][body] / dM_true[body] - 1.0).max() < vem.TH_PLUMMER_REL
    assert np.abs(stats["dM_flux"][body] / dM_true[body] - 1.0).max() < 1e-6


# ---------------------------------------------------------------- step 4 machinery

def test_anchor_selection_contract():
    """Anchors are real truth edges (or the grid start) sitting exactly on
    the radial grid -- the plan's 'only actual boundaries' rule."""
    edges = np.array([0.5, 0.7, 1.3, 2.128006, 3.0, 5.188616, 9.0, 20.0])
    nodes = edges.copy()            # grid that contains every edge exactly
    eidx = np.arange(1, edges.size)  # node index of edges_out[j] = j+1
    a0 = vem.anchor_edge_and_node(edges, eidx, nodes, 0.5)
    assert a0["node_idx"] == 0 and a0["m_truth_at_anchor"] == 0.0
    a2 = vem.anchor_edge_and_node(edges, eidx, nodes, 2.0)
    assert a2["r_kpc"] == pytest.approx(2.128006, abs=1e-12)
    assert a2["edge_pos"] == 2      # position among the OUTER edges
    assert a2["node_idx"] == 3
    a5 = vem.anchor_edge_and_node(edges, eidx, nodes, 5.0)
    assert a5["r_kpc"] == pytest.approx(5.188616, abs=1e-12)
    with pytest.raises(ValueError):
        vem.anchor_edge_and_node(edges, eidx, nodes, 25.0)
    # an anchor edge that is not an exact grid node must be rejected
    bad_nodes = nodes.copy()
    bad_nodes[3] += 1e-6
    with pytest.raises(ValueError):
        vem.anchor_edge_and_node(edges, eidx, bad_nodes, 2.0)


def test_anchor_telescoping_identity_plummer():
    """DeltaM(r;a) from the 0.5-anchored cumulative equals a direct
    integration from the anchor node on the same grid -- the identity the
    step-4 anchored volume curves rely on."""
    r_nodes = vem.make_radial_nodes(0.5, 70.0, per_interval=40,
                                    boundaries=(2.128006, 5.188616))
    rho = vem.plummer_density(r_nodes)
    cum05 = vem.cumulative_volume_mass(r_nodes, rho, 0.5)
    for a in (2.128006, 5.188616):
        ia = int(np.flatnonzero(np.isclose(r_nodes, a, atol=1e-12))[0])
        shifted = cum05 - cum05[ia]
        direct = vem.cumulative_volume_mass(r_nodes[ia:], rho[ia:], a)
        # both arrays live on the nodes >= a; both are exactly 0 at a itself
        assert np.allclose(shifted[ia:], direct, rtol=1e-12, atol=0.0)


def test_shell_increments_constant_density():
    """Per-shell increments of a constant-density field recover the exact
    shell masses 4 pi rho / 3 (e_i^3 - e_{i-1}^3)."""
    edges = np.array([0.5, 1.0, 2.0, 4.0])
    r_nodes = vem.make_radial_nodes(0.5, 4.0, per_interval=200,
                                    boundaries=edges[1:])
    rho = np.full(r_nodes.size, 3.0)
    cum = vem.cumulative_volume_mass(r_nodes, rho, 0.5)
    vals = np.concatenate(
        [[0.0], cum[vem.grid_edge_indices(r_nodes, edges[1:])]])
    shells = vem.shell_increments(vals)
    assert np.allclose(shells,
                       4.0 * np.pi * 3.0 / 3.0 * np.diff(edges ** 3),
                       rtol=1e-4)


def test_longest_true_run():
    assert vem._longest_true_run([True, True, False, True]) == 2
    assert vem._longest_true_run([False, False]) == 0
    assert vem._longest_true_run([]) == 0


@needs_truth
def test_step4_anchors_on_truth_grid():
    """On the real truth file, the step-4 anchors are the first edges >= 2
    and >= 5 kpc and lie exactly on the step-3 radial grid (p=4)."""
    truth = vem.load_truth()
    edges_out = truth["r_edges"][1:]
    nodes = vem.make_radial_nodes(0.5, 70.0, 4, boundaries=edges_out)
    eidx = vem.grid_edge_indices(nodes, edges_out)
    a2 = vem.anchor_edge_and_node(truth["r_edges"], eidx, nodes, 2.0)
    assert a2["r_kpc"] == float(edges_out[edges_out >= 2.0].min())
    assert nodes[a2["node_idx"]] == pytest.approx(a2["r_kpc"], abs=1e-9)
    a5 = vem.anchor_edge_and_node(truth["r_edges"], eidx, nodes, 5.0)
    assert a5["r_kpc"] == float(edges_out[edges_out >= 5.0].min())
    assert nodes[a5["node_idx"]] == pytest.approx(a5["r_kpc"], abs=1e-9)


# ---------------------------------------------------------------- step 5 machinery

def test_sph_harm_power_constant_and_quadrupole():
    """A_l of a constant field: A_0 = <c>/sqrt(4 pi) exactly; A_l>0 do NOT
    vanish exactly -- they are the Sobol quadrature bias floor of the
    estimator (~1e-3 at N=1024, recorded as the floor in step 5).  An
    axisymmetric centred anisotropic field: measured dipole stays at the
    floor while the quadrupole is far above it and matches an independent
    Gauss-Legendre reference."""
    dirs = vem.sobol_directions(1024, scramble_seed=0)
    const = np.full(dirs.shape[0], 2.5)
    A = vem.sph_harm_power(const, dirs, 4)
    assert A[0] == pytest.approx(2.5 / np.sqrt(4.0 * np.pi), rel=1e-12)
    assert np.all((A / A[0])[1:] < 5e-3)

    comps = ((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))
    r = 10.0
    vals = vem.anisotropic_gauss_density(dirs * (r / L_KPC), L_KPC, comps)
    A = vem.sph_harm_power(vals, dirs, 4)
    assert A[1] / A[0] < 5e-3                    # centred: dipole at the floor
    assert A[1] < 1e-2 * A[2]                    # ...far below the quadrupole
    assert A[2] / A[0] > 1e-2                    # flattened: quadrupole
    # independent reference: for an axisymmetric field only m=0 contributes,
    # a_l0 = sqrt((2l+1) pi) * int_-1^1 rho(mu) P_l(mu) dmu at fixed r
    from numpy.polynomial.legendre import leggauss
    from scipy.special import eval_legendre
    mu, w = leggauss(200)
    s_sin = r * np.sqrt(1.0 - mu ** 2)
    zz = r * mu
    rho_mu = np.zeros_like(mu)
    for M, a, c in comps:
        norm = M * (2.0 * np.pi) ** (-1.5) / (a ** 2 * c)
        rho_mu += norm * np.exp(-0.5 * (s_sin ** 2 / a ** 2 + zz ** 2 / c ** 2))
    a0 = np.sqrt(np.pi) * np.sum(rho_mu * w)                    # P_0 = 1
    a2 = np.sqrt(5.0 * np.pi) * np.sum(rho_mu * eval_legendre(2, mu) * w)
    assert abs(A[2] / A[0] - a2 / a0) / (a2 / a0) < 1e-3


def test_equal_area_map_grid_and_stats():
    """Equal-area centre grid: unit vectors; the plain cell mean of an
    axisymmetric field (cells have equal area by construction) matches the
    Gauss-Legendre sphere mean; a positive field has zero negative
    fraction."""
    lat, lon = vem.equal_area_map_grid(180, 360)
    dirs = vem.directions_from_grid(lat, lon)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
    comps = ((5.0e10, 3.0, 6.0), (2.0e10, 8.0, 2.0))
    r = 10.0
    vals = vem.anisotropic_gauss_density(dirs * (r / L_KPC), L_KPC,
                                         comps).reshape(lat.size, lon.size)
    mean, neg = vem.map_stats(vals)
    ref = vem.anisotropic_gauss_sphere_mean(np.array([r]), comps, n_mu=200)[0]
    assert abs(mean / ref - 1.0) < vem.TH_NONSPH_REL
    assert neg == 0.0


def test_radial_force_mean_matches_plummer():
    """Mean radial acceleration -(V^2/L) <n.grad phi> = -G M(<r)/r^2 for the
    spherical Plummer (exact through the flux theorem)."""
    phi, M, b = vem.plummer_phi_func()
    r_nodes = np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    dirs = vem.sobol_directions(512, scramble_seed=0)
    _, dn_dir = vem.per_direction_rho_and_dn(phi, r_nodes, dirs, L_KPC, V_KMS)
    a_mean = -(V_KMS ** 2 / L_KPC) * dn_dir.mean(axis=0)
    a_true = -vem.G_KPC_KMS2_MSUN * vem.plummer_mass_inside(r_nodes) / r_nodes ** 2
    assert np.allclose(a_mean, a_true, rtol=1e-8)


def test_pooled_direction_band_contract():
    """Pooled percentiles/mean of the direction band are plain statistics of
    the concatenated per-direction ratio arrays; repeatability envelope is
    the min/max over per-seed mean curves."""
    rng = np.random.default_rng(0)
    n_dir, n_edge = 64, 5
    seeds = (0, 1)
    truth = np.linspace(1.0, 2.0, n_edge)
    d, base = {}, []
    for s in seeds:
        arr = truth[None, :] * (1.0 + 0.1 * rng.normal(size=(n_dir, n_edge)))
        d[f"m__seed{s}__dM_per_dir_edges"] = arr
        base.append(arr)
    out = vem.pooled_direction_band(d, "m", seeds, truth)
    pooled = np.concatenate(base) / truth[None, :]
    assert out["n_directions_pooled"] == 2 * n_dir
    assert np.allclose(out["p16"], np.percentile(pooled, 16, axis=0))
    assert np.allclose(out["p50"], np.percentile(pooled, 50, axis=0))
    assert np.allclose(out["p84"], np.percentile(pooled, 84, axis=0))
    assert np.allclose(out["mean"], pooled.mean(axis=0))
    assert np.allclose(out["seed_means"],
                       np.stack([b.mean(axis=0) / truth for b in base]))
    assert np.allclose(out["rep_min"], out["seed_means"].min(axis=0))
    assert np.allclose(out["rep_max"], out["seed_means"].max(axis=0))


def test_density_map_positive_field_constant_radius():
    """Map of a spherical field is constant across the sky (Plummer through
    the potential->density path), so the weighted mean equals the analytic
    density and the negative fraction is zero."""
    phi, M, b = vem.plummer_phi_func()
    lat, lon = vem.equal_area_map_grid(60, 120)
    r = 8.0
    rho_map = vem.density_map(phi, r, lat, lon, L_KPC, V_KMS)
    rho_true = vem.plummer_density(r, M, b)
    assert np.abs(rho_map / rho_true - 1.0).max() < 1e-6
    mean, neg = vem.map_stats(rho_map)
    assert abs(mean / rho_true - 1.0) < 1e-6
    assert neg == 0.0


def test_angular_truth_inventory_schemas(tmp_path):
    """Both position schemas are detected (Gadget-style Coordinates and the
    dpjax scalar x/y/z export); only a file with ALL of PartType0/1/4
    positions qualifies as a candidate angular truth."""
    import h5py
    star = tmp_path / "star_only.h5"
    with h5py.File(star, "w") as f:
        g = f.create_group("PartType4")
        for ax in "xyz":
            g.create_dataset(ax, data=np.zeros(4))
    full = tmp_path / "all_sources.h5"
    with h5py.File(full, "w") as f:
        for c in ("PartType0", "PartType1", "PartType4"):
            f.create_group(c).create_dataset("Coordinates",
                                             data=np.zeros((4, 3)))
    inv = vem.angular_truth_inventory(tmp_path, include_raw_snapshot=False)
    star_entry = inv["files_scanned"][str(star)]["components_with_positions"]
    assert star_entry == {"PartType4": ["x", "y", "z"]}
    full_entry = inv["files_scanned"][str(full)]["components_with_positions"]
    assert set(full_entry) == {"PartType0", "PartType1", "PartType4"}
    assert inv["qualified_files"] == [str(full)]
    # the chunked raw snapshot is checked as ONE aggregate entry when present
    if all(Path(str(vem.RAW_SNAP_TEMPLATE).format(i=i)).exists()
           for i in range(vem.RAW_SNAP_N_CHUNKS)):
        inv2 = vem.angular_truth_inventory(tmp_path,
                                           include_raw_snapshot=True)
        agg = [k for k in inv2["files_scanned"] if "8 chunks" in k]
        assert len(agg) == 1
        assert inv2["files_scanned"][agg[0]]["n_chunks_present"] == 8
        assert set(inv2["files_scanned"][agg[0]][
            "components_with_positions"]) == {"PartType0", "PartType1",
                                              "PartType4"}
        assert agg[0] in inv2["qualified_files"]


# --------------------------------------------- step 5b machinery (truth side)

def test_kabsch_det_recovers_rotation_and_reflection():
    """The raw->model transform turned out to be IMPROPER (det = -1); the
    determinant-free Kabsch must recover rotations and roto-reflections
    exactly, while a wrong-sign fit of mirrored data must fail loudly."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(400, 3))
    th, ph = 0.9, -0.4
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th), np.cos(th)]])
    Rz = np.array([[np.cos(ph), -np.sin(ph), 0],
                   [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    for target, extra in ((+1, np.eye(3)), (-1, np.diag([1.0, 1.0, -1.0]))):
        T = extra @ Rz @ Rx
        B = (T @ (A - A.mean(0)).T).T
        R, c, res = vem.kabsch_det(A, B, target)
        assert res.max() < 1e-10
        assert np.sign(np.linalg.det(R)) == target
        # wrong-sign fit cannot align mirrored/rotated data
        R_wrong, _, res_wrong = vem.kabsch_det(A, B, -target)
        assert np.median(res_wrong) > 0.1


def test_shell_cell_assign_and_power_quadrupole():
    """Particle shell -> equal-area cell-mass map -> A_l/A_0 chain on a
    synthetic shell with quadrupole-weighted masses: A_2/A_0 must match the
    closed form eps*sqrt(5)/5 (for f = 1 + eps*P_2(mu)) well above shot
    noise; uniform weights must give noise-level ratios."""
    rng = np.random.default_rng(11)
    n = 400_000
    r = 10.0 * (1 + 0.2 * (rng.random(n) - 0.5))       # +/-10% shell
    mu = rng.uniform(-1, 1, n)
    lon = rng.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(mu)
    pos = np.column_stack([r * np.cos(lat) * np.cos(lon),
                           r * np.cos(lat) * np.sin(lon),
                           r * np.sin(lat)])
    eps = 0.3
    w = 1.0 + eps * 0.5 * (3 * mu ** 2 - 1)
    a = vem.shell_cell_assign(pos, w, 10.0, 1.1, 45, 90)
    ratio = vem.map_angular_power_ratio(a["mass_map"],
                                        *vem.equal_area_map_grid(45, 90))
    assert abs(ratio[2] - eps * np.sqrt(5) / 5) < 0.01
    assert ratio[1] < 0.02 and ratio[3] < 0.02 and ratio[4] < 0.02
    # shot noise on the same shell
    a2 = vem.shell_cell_assign(pos, np.ones(n), 10.0, 1.1, 45, 90)
    ratio2 = vem.map_angular_power_ratio(a2["mass_map"],
                                         *vem.equal_area_map_grid(45, 90))
    assert (ratio2[1:] < 0.02).all()
    boot = vem.bootstrap_shell_power(a2, 8, 42, 45, 90)
    assert boot.shape == (8, vem.STEP5_LMAX)
    # bootstrap std must be small relative to the injected quadrupole
    assert boot[:, 1].std() < 0.2 * eps * np.sqrt(5) / 5


def test_rebin_map_block_mean():
    a = np.arange(36, dtype=float).reshape(6, 6)
    b = vem.rebin_map(a, 3)
    assert b.shape == (2, 2)
    assert b[0, 0] == a[:3, :3].mean()
    assert b[1, 1] == a[3:, 3:].mean()
    with pytest.raises(ValueError):
        vem.rebin_map(a, 4)                      # 4 does not divide 6


# ---------------------------------------------------------------- non-finite

def test_nonfinite_values_fail_loudly():
    with pytest.raises(ValueError):
        vem.finite_or_fail([1.0, np.nan], "unit-test")
    with pytest.raises(ValueError):
        vem.finite_or_fail([1.0, np.inf], "unit-test")
    assert vem.finite_or_fail([1.0, 2.0], "unit-test").sum() == 3.0
