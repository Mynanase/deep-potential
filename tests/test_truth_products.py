#!/usr/bin/env python
"""Tests for the truth-side preprocessing collection (truth_products).

Validate the MACHINERY on analytic examples: shell masses from a synthetic
uniform-ball grid product (M(<r) proportional to r^3), r-bin tables from a
synthetic eta h5 (exact digitize counts, histogram sums), and the lineage
idempotency contract (reuse on match, refuse on mismatch).  No trained
models, no real Auriga data.

Run from the repo root:  python -m pytest tests/test_truth_products.py -q
CPU only.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "auriga"))

import truth_products as tp  # noqa: E402


R_BALL = 9.0
N_GRID = 48
EXTENT = 10.0


def _write_ball_grids(path, rho=2.0):
    """Synthetic grids product: uniform density rho inside r <= R_BALL."""
    edges = np.linspace(-EXTENT, EXTENT, N_GRID + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    xx, yy, zz = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    rho3d = np.where(r <= R_BALL, rho, 0.0)
    cell = (2 * EXTENT / N_GRID) ** 3
    m_total = float(rho3d.sum() * cell)
    with h5py.File(path, "w") as f:
        f.attrs.update({
            "schema": tp.GRIDS_SCHEMA,
            "n_particles": 1_000_000,   # effective-count scale for the CI
            "M_total_msun": m_total,
        })
        f["density/rho3d"] = rho3d
        f["density/rho3d_edges_kpc"] = np.stack([edges] * 3)
    return m_total


class TestShellMass:
    def test_uniform_ball_cumulative(self, tmp_path):
        grids = tmp_path / "grids.h5"
        m_total = _write_ball_grids(grids)
        edges = np.array([0.0, 2.0, 5.0, R_BALL])
        nodes = np.array([1.0, 3.0, 7.0, R_BALL])
        tab = tp.shell_mass_from_grids(grids, edges, nodes)

        # analytic M(<r) = M_total * (r/R)^3 for r inside the ball
        expect = m_total * (nodes / R_BALL) ** 3
        rel = np.abs(tab["M_cum"] / expect - 1.0)
        # cell-centre binning: generous at the ball edge, tight well inside
        assert rel[0] < 0.05 and rel[1] < 0.05 and rel[2] < 0.02
        assert rel[3] < 0.03  # full ball: only boundary cells lost/gained

        # shell masses sum to the cumulative at the last edge
        assert abs(tab["M_shell"].sum() / tab["M_cum"][-1] - 1.0) < 5e-3

        # Poisson CI is positive; relative error shrinks with cumulative N_eff
        assert np.all(tab["M_cum_err"] > 0)
        ratio = tab["M_cum_err"] / tab["M_cum"]
        assert np.all(np.diff(ratio) < 0)  # 1/sqrt(N_cum) decreasing
        assert np.isclose(ratio[-1], 1.0 / np.sqrt(tab["N_eff_shell"].sum()),
                          rtol=0.05)

    def test_schema_guard(self, tmp_path):
        bad = tmp_path / "bad.h5"
        with h5py.File(bad, "w") as f:
            f["density/rho3d"] = np.zeros((2, 2, 2))
        with pytest.raises(SystemExit):
            tp.shell_mass_from_grids(bad, np.array([0.0, 5.0]), np.array([1.0]))


def _write_eta(path, n=4000, seed=0, L=10.0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    pos *= (rng.random(n)[:, None] ** (1 / 3)) / np.linalg.norm(pos, axis=1)[:, None] * 0.7
    vel = rng.normal(scale=0.3, size=(n, 3))
    eta = np.concatenate([pos, vel], axis=1)
    w = rng.uniform(0.5, 2.0, size=n)
    with h5py.File(path, "w") as f:
        f["eta"] = eta
        f["weights"] = w
        f.attrs["length_scale_kpc"] = L
    return eta, w


class TestRadialBinTable:
    def test_counts_and_histogram_sums(self, tmp_path):
        src = tmp_path / "in.h5"
        eta, w = _write_eta(src)
        edges = np.array([0.0, 2.0, 4.0, 7.0])
        tab = tp.radial_bin_table(src, edges, weighting="mass", split="all")
        r = np.linalg.norm(eta[:, :3], axis=1) * 10.0
        idx = np.digitize(r, edges) - 1
        inside = (r >= edges[0]) & (r < edges[-1])
        assert tab["n_used"] == int(inside.sum())
        for b in range(3):
            m = inside & (idx == b)
            assert tab["bins"][b]["count"] == int(m.sum())
            assert np.isclose(tab["bins"][b]["mass"], w[m].sum())
            assert int(tab["bins"][b]["r_hist"].sum()) == int(m.sum()) or \
                np.isclose(tab["bins"][b]["r_hist"].sum(), w[m].sum())
            for ci in range(3):
                assert np.isclose(tab["bins"][b]["v_hists"][ci].sum(),
                                  w[m].sum())

    def test_number_weighting_and_val_split(self, tmp_path):
        src = tmp_path / "in.h5"
        eta, w = _write_eta(src, n=1000, seed=1)
        edges = np.array([0.0, 10.0])
        tab = tp.radial_bin_table(src, edges, weighting="number", split="val")
        # val split = first quarter
        n_val = 250
        assert tab["bins"][0]["count"] == n_val
        assert np.isclose(tab["bins"][0]["mass"], float(n_val))  # ones


class TestLineageCache:
    def test_reuse_and_refuse(self, tmp_path):
        src = tmp_path / "in.h5"
        _write_eta(src)
        out = tmp_path / "rh.h5"
        edges = "0,4,7"
        argv_like = dict(input=src, output=out, edges=edges,
                         weighting="mass", split="val", force=False)
        rc = tp.cmd_build_radial_hists(type("A", (), argv_like)())
        assert rc == 0 and out.is_file()
        # second identical call: cache hit, still rc 0, file untouched
        stamp = out.stat().st_mtime_ns
        rc = tp.cmd_build_radial_hists(type("A", (), argv_like)())
        assert rc == 0 and out.stat().st_mtime_ns == stamp
        # different lineage: refused
        argv_like["edges"] = "0,3,7"
        with pytest.raises(SystemExit, match="DIFFERENT lineage"):
            tp.cmd_build_radial_hists(type("A", (), argv_like)())
        # force overrides
        argv_like["force"] = True
        rc = tp.cmd_build_radial_hists(type("A", (), argv_like)())
        assert rc == 0
