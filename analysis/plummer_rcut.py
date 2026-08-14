import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    # 确保项目根目录在 sys.path 中（marimo 的 cwd 不一定在项目根）
    _project_root = Path("/localdisk/kosmos/my-deep-potential")
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from experiments.datasets.phase_space import load_eta_h5
    from experiments.diagnostics.evaluation import (
        conditional_velocity_diagnostics,
        cylindrical_rz_density_by_phi,
    )
    from experiments.paths import resolve_path

    return (
        colors,
        conditional_velocity_diagnostics,
        cylindrical_rz_density_by_phi,
        json,
        load_eta_h5,
        mo,
        np,
        plt,
        resolve_path,
    )


@app.cell
def _(mo):
    mo.md("""
    # Plummer r-cut 实验分析

    **动机**：Auriga 数据用全部 stars 训练 DF 时，中心 stars 主导贡献、外围被忽视。这里用 Plummer 球做受控实验：全量数据（full）vs 去心数据（cut = full 中 r≥1.0 的子集）。

    - **Part A**：原始数据 DF 分布对比（R-z by φ + 三个速度 marginals + 径向质量占比）——训练前即可运行。
    - **Part B**：DF / Phi 拟合后评估对比（依赖 `run_df`/`run_phi`/`run_eval` 产物）。
    """)
    return


@app.cell
def _(np):
    DATA_PATHS = {
        "full": "data/plummer_n524288_train.h5",
        "cut": "data/plummer_n524288_rcut1.0_train.h5",
    }
    RUN_DIRS = {
        "full": "runs/plummer_rcut/full",
        "cut": "runs/plummer_rcut/cut",
    }
    # 与 configs/runs/plummer_rcut_{full,cut}.yaml 的 evaluation 配置保持一致
    RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    THETA_EDGES = np.arccos(np.linspace(1.0, -1.0, 7))  # 6 bins
    PHI_EDGES = np.linspace(-np.pi, np.pi, 9)  # 8 bins
    N_VELOCITY_BINS = 64
    SPATIAL_R_EDGES = np.linspace(0.0, 10.0, 49)  # 48 bins
    SPATIAL_Z_EDGES = np.linspace(-10.0, 10.0, 49)
    MIN_CELL_COUNT = 5
    return (
        DATA_PATHS,
        MIN_CELL_COUNT,
        N_VELOCITY_BINS,
        PHI_EDGES,
        RADIAL_EDGES,
        RUN_DIRS,
        SPATIAL_R_EDGES,
        SPATIAL_Z_EDGES,
        THETA_EDGES,
    )


@app.cell
def _(DATA_PATHS, load_eta_h5, mo, np, resolve_path):
    eta_full = load_eta_h5(resolve_path(DATA_PATHS["full"]))
    eta_cut = load_eta_h5(resolve_path(DATA_PATHS["cut"]))
    r_full = np.linalg.norm(eta_full[:, :3], axis=1)
    _r_cut = np.linalg.norm(eta_cut[:, :3], axis=1)
    info = mo.md(
        f"**full**: N={eta_full.shape[0]}, "
        f"r∈[{r_full.min():.3f}, {r_full.max():.3f}]（含 r<1.0 共 {(r_full < 1.0).sum()} 颗）\n\n"
        f"**cut**: N={eta_cut.shape[0]}, r∈[{_r_cut.min():.3f}, {_r_cut.max():.3f}]"
    )
    return eta_cut, eta_full, r_full


@app.cell
def _(
    MIN_CELL_COUNT,
    PHI_EDGES,
    SPATIAL_R_EDGES,
    SPATIAL_Z_EDGES,
    colors,
    cylindrical_rz_density_by_phi,
    eta_cut,
    eta_full,
    np,
    plt,
):
    """Part A｜R-z by φ：full（数据）vs cut（模型）的密度与 log 比值。"""
    _rz = cylindrical_rz_density_by_phi(
        eta_full[:, :3],
        eta_cut[:, :3][None, :, :],
        reference_weights=None,
        phi_edges=PHI_EDGES,
        cylindrical_radius_edges=SPATIAL_R_EDGES,
        z_edges=SPATIAL_Z_EDGES,
        min_cell_count=MIN_CELL_COUNT,
    )
    _reference_density = _rz["reference_density"]  # full
    _model_density = _rz["model_median_density"]  # cut
    _reference_count = _rz["reference_count"]
    _model_count = _rz["model_count"].reshape(_reference_count.shape)
    _n_phi = PHI_EDGES.size - 1

    _positive = np.concatenate(
        [
            _reference_density[_reference_density > 0],
            _model_density[_model_density > 0],
        ]
    )
    _density_vmin, _density_vmax = np.percentile(_positive, [5.0, 99.5])
    if _density_vmax <= _density_vmin:
        _density_vmax = _density_vmin * 10.0
    _density_norm = colors.LogNorm(
        vmin=max(float(_density_vmin), np.finfo(float).tiny),
        vmax=float(_density_vmax),
    )

    _valid = (
        (_reference_count >= MIN_CELL_COUNT)
        & (_model_count >= MIN_CELL_COUNT)
        & (_reference_density > 0)
        & (_model_density > 0)
    )
    _log_ratio = np.full_like(_reference_density, np.nan)
    _log_ratio[_valid] = np.log10(
        _model_density[_valid] / _reference_density[_valid]
    )
    _finite_ratio = np.abs(_log_ratio[np.isfinite(_log_ratio)])
    _ratio_limit = (
        max(float(np.percentile(_finite_ratio, 98.0)), 0.1)
        if _finite_ratio.size
        else 1.0
    )
    _ratio_norm = colors.TwoSlopeNorm(
        vmin=-_ratio_limit, vcenter=0.0, vmax=_ratio_limit
    )

    rz_fig, _axes = plt.subplots(
        3,
        _n_phi,
        figsize=(max(14.0, 3.0 * _n_phi), 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    _axes = np.asarray(_axes).reshape(3, _n_phi)
    _density_mappable = None
    _ratio_mappable = None
    for _phi_index in range(_n_phi):
        _phi_left = np.degrees(PHI_EDGES[_phi_index])
        _phi_right = np.degrees(PHI_EDGES[_phi_index + 1])
        _density_mappable = _axes[0, _phi_index].pcolormesh(
            SPATIAL_R_EDGES,
            SPATIAL_Z_EDGES,
            np.ma.masked_less_equal(_reference_density[_phi_index].T, 0.0),
            cmap="magma",
            norm=_density_norm,
            shading="auto",
        )
        _axes[1, _phi_index].pcolormesh(
            SPATIAL_R_EDGES,
            SPATIAL_Z_EDGES,
            np.ma.masked_less_equal(_model_density[_phi_index].T, 0.0),
            cmap="magma",
            norm=_density_norm,
            shading="auto",
        )
        _ratio_mappable = _axes[2, _phi_index].pcolormesh(
            SPATIAL_R_EDGES,
            SPATIAL_Z_EDGES,
            np.ma.masked_invalid(_log_ratio[_phi_index].T),
            cmap="coolwarm",
            norm=_ratio_norm,
            shading="auto",
        )
        _axes[0, _phi_index].set_title(
            rf"${_phi_left:.0f}^\circ\leq\phi<{_phi_right:.0f}^\circ$"
        )
        _axes[2, _phi_index].set_xlabel("R")
    _axes[0, 0].set_ylabel("full\nz")
    _axes[1, 0].set_ylabel("cut\nz")
    _axes[2, 0].set_ylabel(r"$\log_{10}(\rho_{cut}/\rho_{full})$" "\nz")
    rz_fig.colorbar(
        _density_mappable,
        ax=_axes[:2, :].ravel().tolist(),
        label="normalized density",
        shrink=0.85,
    )
    rz_fig.colorbar(
        _ratio_mappable,
        ax=_axes[2, :].ravel().tolist(),
        label="log10 density ratio",
        shrink=0.85,
    )
    rz_fig.suptitle(
        "Plummer R-z density by azimuth wedge — full vs cut (r≥1.0)",
        fontsize=14,
    )
    return


@app.cell
def _(np, plt, r_full):
    """Part A｜径向累积质量占比：展示中心主导的动机。"""
    _r_sorted = np.sort(r_full)
    _cum_frac = np.arange(1, len(_r_sorted) + 1) / len(_r_sorted)
    cumfrac_fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(_r_sorted, _cum_frac, color="tab:blue", lw=2)
    _ax.axvline(
        1.0, ls="--", color="tab:red", lw=1.5, label=r"$r_{\rm cut}=1.0$"
    )
    _ax.axhline(
        0.354, ls=":", color="k", alpha=0.6, label="M(<1.0)/M=0.354"
    )
    _ax.set_xlabel("r")
    _ax.set_ylabel("cumulative particle fraction")
    _ax.set_title("Full data: cumulative mass (center-dominated)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    return


@app.cell
def _(
    N_VELOCITY_BINS,
    PHI_EDGES,
    RADIAL_EDGES,
    THETA_EDGES,
    conditional_velocity_diagnostics,
    eta_cut,
    eta_full,
    mo,
    np,
    plt,
):
    """Part A｜三个速度 marginals by r / theta / phi：full（黑实线）vs cut（红虚线）。"""
    _conditional = conditional_velocity_diagnostics(
        eta_full,
        eta_cut[None, :, :],
        reference_weights=None,
        conditioning_edges={
            "r": RADIAL_EDGES,
            "theta": THETA_EDGES,
            "phi": PHI_EDGES,
        },
        n_velocity_bins=N_VELOCITY_BINS,
    )
    _velocity_labels = (r"$v_r$", r"$v_\theta$", r"$v_\phi$")
    _velocity_edges = _conditional["velocity_edges"]
    _figures = []
    for _coordinate_name in ("r", "theta", "phi"):
        _values = _conditional[_coordinate_name]
        _edges = _values["edges"]
        _reference_hist = _values["reference_hist"]
        _model_hist = _values["model_hist"][0]
        _n_rows = _edges.size - 1
        _fig, _axes = plt.subplots(
            _n_rows,
            3,
            figsize=(13.5, max(3.5, 1.9 * _n_rows)),
            sharex="col",
            squeeze=False,
            constrained_layout=True,
        )
        _axes = np.asarray(_axes).reshape(_n_rows, 3)
        for _row in range(_n_rows):
            for _velocity_index in range(3):
                _ax = _axes[_row, _velocity_index]
                _ax.stairs(
                    _reference_hist[_row, _velocity_index],
                    _velocity_edges[_velocity_index],
                    color="black",
                    lw=1.8,
                    label="full",
                )
                _ax.stairs(
                    _model_hist[_row, _velocity_index],
                    _velocity_edges[_velocity_index],
                    color="tab:red",
                    lw=1.4,
                    ls="--",
                    label="cut",
                )
                _ax.grid(True, alpha=0.2)
                if _row == 0:
                    _ax.set_title(_velocity_labels[_velocity_index])
                if _row == _n_rows - 1:
                    _ax.set_xlabel("velocity")
            _left = _edges[_row]
            _right = _edges[_row + 1]
            if _coordinate_name == "r":
                _interval = f"{_left:.2g} ≤ r < {_right:.2g}"
            else:
                _interval = (
                    f"{np.degrees(_left):.0f}° ≤ {_coordinate_name} "
                    f"< {np.degrees(_right):.0f}°"
                )
            _axes[_row, 0].set_ylabel(f"{_interval}\nPDF")
        _axes[0, -1].legend(loc="upper left", fontsize=7)
        _fig.suptitle(
            f"Velocity marginals conditioned on {_coordinate_name} "
            "— full vs cut",
            fontsize=13,
        )
        _figures.append(_fig)
    velocity_figs = mo.vstack(
        [
            mo.md("## 速度 marginals by r / θ / φ（full 黑实线 vs cut 红虚线）"),
            *_figures,
        ]
    )
    return


@app.cell
def _(RUN_DIRS, json, mo, resolve_path):
    """Part B｜读取两个 case 的 DF 评估 metrics。"""
    metrics_by_case = {}
    for _case in ("full", "cut"):
        _metrics_path = (
            resolve_path(RUN_DIRS[_case])
            / "trial_00"
            / "eval"
            / "df"
            / "auriga_df_metrics.json"
        )
        if _metrics_path.exists():
            metrics_by_case[_case] = json.loads(
                _metrics_path.read_text(encoding="utf-8")
            )
    _missing = [c for c in ("full", "cut") if c not in metrics_by_case]
    mo.stop(
        _missing,
        mo.callout(
            "尚未找到 DF 评估产物（需完成 run_df / run_eval）。"
            f"缺失 case: {_missing}",
            kind="warn",
        ),
    )
    return (metrics_by_case,)


@app.cell
def _(metrics_by_case, mo):
    """Part B｜汇总表：径向密度误差 + 速度 marginal W1 中位数。"""
    _rows = []
    for _case, _m in metrics_by_case.items():
        _density = _m["density_profile"]
        _conditional = _m["conditional_velocity"]
        _row = {
            "case": _case,
            "n_data": _m["n_data"],
            "density log10_rmse_dex": round(
                float(_density["log10_rmse_dex"]), 4
            ),
        }
        for _coordinate_name in ("r", "theta", "phi"):
            _w1 = _conditional[_coordinate_name][
                "median_wasserstein_by_velocity"
            ]
            for _velocity_name in ("v_r", "v_theta", "v_phi"):
                _value = _w1[_velocity_name]
                _row[f"W1 {_coordinate_name}.{_velocity_name}"] = (
                    f"{float(_value):.3g}" if _value is not None else "n/a"
                )
        _rows.append(_row)
    table = mo.ui.table(_rows, label="full vs cut DF 拟合质量")
    return


@app.cell
def _(RUN_DIRS, mo, resolve_path):
    """Part B｜DF 评估图（density profile / R-z by φ / 速度 marginals）。"""
    _galleries = {}
    for _case in ("full", "cut"):
        _plots_dir = resolve_path(RUN_DIRS[_case]) / "plots" / "df"
        _image_paths = (
            sorted(_plots_dir.glob("*.png")) if _plots_dir.exists() else []
        )
        _galleries[_case] = (
            mo.ui.tabs({_p.name: mo.image(src=_p) for _p in _image_paths})
            if _image_paths
            else mo.callout(f"{_case}: 暂无 DF 图（等待 run_eval）", kind="info")
        )
    df_gallery = mo.vstack(
        [
            mo.md("## DF 拟合后评估图"),
            mo.hstack([mo.md("### full"), _galleries["full"]], widths=[1, 3]),
            mo.hstack([mo.md("### cut"), _galleries["cut"]], widths=[1, 3]),
        ]
    )
    return


@app.cell
def _(RUN_DIRS, mo, resolve_path):
    """Part B｜Phi 评估图。"""
    _galleries = {}
    for _case in ("full", "cut"):
        _plots_dir = resolve_path(RUN_DIRS[_case]) / "plots" / "phi"
        _image_paths = (
            sorted(_plots_dir.glob("*.png")) if _plots_dir.exists() else []
        )
        _galleries[_case] = (
            mo.ui.tabs({_p.name: mo.image(src=_p) for _p in _image_paths})
            if _image_paths
            else mo.callout(f"{_case}: 暂无 Phi 图（等待 run_eval）", kind="info")
        )
    phi_gallery = mo.vstack(
        [
            mo.md("## Phi 拟合后评估图"),
            mo.hstack([mo.md("### full"), _galleries["full"]], widths=[1, 3]),
            mo.hstack([mo.md("### cut"), _galleries["cut"]], widths=[1, 3]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
