# keep/ — Phase-1 长期保留的诊断与绘图脚本

**收录日期**：2026-09-22 ｜ **决策依据**：`docs/phase2-premerge-survey.md` §D
**状态**：暂不接入主流程；每件标注来源，供后续按需提升为标准流程组件。
**共同改动**：脚本从原位置（`scripts/auriga/`）拷入本目录（深一级），
`df_radial_*` 的 `REPO = parents[2]` 已改为 `parents[3]`；两个 plot 脚本补了
`HERE.parent.parent`（scripts/）的 sys.path 插入。其余内容与来源提交逐字一致。

## #1 径向 r-bin：质量 + 速度分布

| 文件 | 用途 | 来源（分支 @ commit） | 输出 |
|---|---|---|---|
| `df_radial_velocity_marginals.py` | 三个速度分量在 r-bin 上的边际分布（模型对比） | `orx/radial-velocity-marginals-w1024full-vs-w128-vs-w` @ `0b4282b` | `fig_radial_marginals_{strict,h12val_outer}.png` |
| `df_radial_rbin.py` | r-bin 计数/分布（固定 bin 边界 0/10/20/30/45/64/75 kpc） | `orx/radial-marginals-with-clump-covering-r-bins` @ `199830c` | `fig_radial_rbins_{full,outer}.png` |
| `df_radial_rbin_own.py` | 各模型对**自身种群**真值的 r-bin 对比 | `orx/radial-r-bins-per-model-own-population-truth` @ `ad44e32` | `fig_radial_rbin_own_{full,outer}.png` |
| `df_radial_rbin_mass.py` | 未归一化质量直方图版 | `orx/radial-r-bins-unnormalized-mass-histograms` @ `aef6f40` | `fig_radial_rbin_mass_{full,outer}.png` |

依赖：`pandas`、`fit_all`（repo `scripts/`）；需要服务器上的训练产物 run 目录。

## #2 真值 vs 模型：2D 势 / 密度对比（两个主力）

| 文件 | 用途 | 来源 | 参考 run |
|---|---|---|---|
| `plot_pt_potential_2d.py` | 粒子真值 vs 模型的 2D 势切片 + 残差 | `orx/cap-w512-three-point-width-trend-on-particle-tru` @ `e197fea` | `0e812d65` 系 |
| `plot_pt_density_resid_lambda.py` | λ 轮三模型（base/innerA/λ10）密度图 + 残差 | `orx/density-maps-base-innera-lambda10-drop-lambda0-1` @ `f80d415` | `553b68f8` |

依赖：`validate_enclosed_mass`（在 `scripts/auriga/`，随基线自带）、本目录的
`particle_truth.py` 与 `orx_figstyle.py`、函数体内延迟 `import fit_all`。

## #5 粒子真值（density-2d 真值底座）

| 文件 | 用途 | 来源 |
|---|---|---|
| `build_total_matter_truth.py` | 从 Auriga 原始粒子构建总物质真值（star frame，权威 gate） | cap-w512 @ `e197fea` |
| `build_particle_truth_grids.py` | 真值网格数据产品（density + potential → 单个 h5） | 同上 |
| `particle_truth.py` | 共享 loader / `phi_direct`（含 XLA slow-fusion 优化） | 同上 |
| `run_total_matter_inventory.sh` | 服务器侧数据盘点 runner（heredoc 自包含；含 `/localdisk/kosmos` 服务器路径） | 同上 |

## 共享依赖

| 文件 | 说明 | 来源 |
|---|---|---|
| `orx_figstyle.py` | 统一 matplotlib 样式库（osc-spectra 线 vendored 最新版） | `orx/density-oscillation-spectra-5-frozen-phis-vs-par` @ `5f80c54` |

## 未收录（survey §D 记录）

- #3 `clump_smooth65` corner preview：session 一次性脚本从未入库，按裁决**取消**。
- #4 数据准备（`prepare_data.py` / `run_w1024_csmooth.sh` / `orx_data_path.txt`）与
  #6 enclosed-mass 上下格式评估图（`plot_pt_adjudication.py`）：**已在主线基线上**，无需收录本目录。
- 其余图族（六模型裁决全套、phi-profiles、osc 谱诊断等）：留在原冻结分支，见 survey §B 索引。
