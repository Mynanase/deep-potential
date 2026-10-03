# archive/ — 归档脚本(Phase-1 与 2026-09-22 绘图层重构的退役件)

**定位变化**:本目录前身是 `keep/`(Phase-1 长期保留诊断)。2026-09-22 绘图层
重构后,活跃资产已提升到 `scripts/` 通用层与 `scripts/auriga/truth_products.py`,
本目录成为**纯归档**:不再接入任何流程,保留出处与替代者,可随时按 git 历史或
本表追溯。原 keep 清单见 `docs/history/phase2-premerge-survey.md` §D(历史账,不改写)。

## 退役→替代 映射(2026-09-22 重构)

| 归档件 | 原用途 | 替代者 |
|---|---|---|
| `plot_particle_truth_2d.py` | 真值网格三图(密度/切片/残差) | `scripts/plot_potential_2d.py --truth` |
| `plot_pt_potential_2d.py` | 真值 vs 模型 2D 势切片 | 同上(合并) |
| `plot_pt_density_resid_lambda.py` | λ 轮三模型密度图 | 同上 rho 模式(裁决按原任务回溯) |
| `plot_potential.py` | 轴向加速度/带符号密度剖面 | `scripts/plot_potential_2d.py`(模型单独模式) |
| `plot_enclosed_mass_verifier_panels.py` | 封闭质量验证器 11 面板调试图 | `scripts/plot_enclosed_mass.py`(通用双面板 + CI) |
| `plot_df_constraints.py` | CBE 约束审计显示 | 计算侧 `auriga/audit_df_constraints.py` 保留;显示件退役 |
| `df_phase1_figures.py` | Phase-1 DF 审计图版 | 一次性事件图,无继任(讲稿引用历史产物) |
| `df_radial_rbin{,_own,_mass}.py` | r-bin 计数/自身真值/质量直方图 | `scripts/plot_radial_marginals.py` |
| `df_radial_velocity_marginals.py` | 三速度分量 r-bin 边际 | 同上 velocity 模式 |
| `build_particle_truth_grids.py` + `particle_truth.py` | 网格产品构建 + 共享库 | `auriga/truth_products.py build-grids`(逐字吸收) |
| `build_total_matter_truth.py` | 星系架粒子资产构建 | 独立重活,原样保留此处(尚未子命令化) |
| `run_total_matter_inventory.sh` | 服务器数据盘点 runner | 原样保留 |
| `orx_figstyle.py` | keep 内样式库副本 | 活跃副本在 `auriga/orx_figstyle.py` |

## 来源谱系(Phase-1,自 keep/README 原样保留)

| 文件 | 来源(分支 @ commit) | 参考 run |
|---|---|---|
| `df_radial_velocity_marginals.py` | `orx/radial-velocity-marginals-…` @ `0b4282b` | `fig_radial_marginals_{strict,h12val_outer}.png` |
| `df_radial_rbin.py` | `orx/radial-marginals-with-clump-covering-r-bins` @ `199830c` | `fig_radial_rbins_{full,outer}.png` |
| `df_radial_rbin_own.py` | `orx/radial-r-bins-per-model-own-population-truth` @ `ad44e32` | `fig_radial_rbin_own_{full,outer}.png` |
| `df_radial_rbin_mass.py` | `orx/radial-r-bins-unnormalized-mass-histograms` @ `aef6f40` | `fig_radial_rbin_mass_{full,outer}.png` |
| `plot_pt_potential_2d.py` | cap-w512 @ `e197fea` | `0e812d65` 系 |
| `plot_pt_density_resid_lambda.py` | `orx/density-maps-…` @ `f80d415` | `553b68f8` |
| `build_total_matter_truth.py` / `build_particle_truth_grids.py` / `particle_truth.py` / `run_total_matter_inventory.sh` | cap-w512 @ `e197fea` | 数据构建 |
| `orx_figstyle.py` | osc-spectra @ `5f80c54` | 样式库 |

**共同改动**:keep→archive 目录名变更(深度不变,REPO/parents 路径无需改);
自 `auriga/` 根移入的 5 件已把 `parents[1..2]` 修正为归档深度,保持原地可编译。

## 已结束研究阶段的 runner

`run_nf_score_cache.sh`、`run_nf_weak_score_tests.sh`、`run_round1_osc_adjudication.sh`、
`run_resolution_convergence.sh` 是旧阶段的命令记录，不在当前流程中调用；其路径与输入反映当时快照。
可复用的计算实现仍在 `scripts/auriga/`。当前任务入口见根 README 与 `docs/progress.md`。
