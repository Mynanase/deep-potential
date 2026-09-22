# Phase-2 合并前产物清点（Pre-merge Survey）

**日期**：2026-09-22 ｜ **用途**：在建立 phase-2 基线分支**之前**，盘点全部实验产物，供逐项决定去留。
**扫描方法**：全分支 `git log` + 尖端包含关系测试（脚本存档于 `/tmp/survey_artifacts.py` 的逻辑，可复现）。

---

## 0. 合并集定义（拓扑已验证）

Phase-2 基线 = 主线尖端 + 4 条工具线尖端，全部共享 inner-band-A(5f76faa) 祖先：

```
S1(af42efb) ── grid-prior(cc5d87c) ── innerA(5f76faa) ── λ=10(b9a981f) ── 六模型裁决(b5b5737)
                    │                                      ├─ 密度图三模型(f80d415)      ← merge 1
                    │                                      └─ BASE: shell-error+pt-gens(af32ee9)
                    └─(经 grid-prior)─ phi链: 内带裁决→paired-shell→…→phi-profiles(2b26cbc) ← merge 2
                                   └─ 谱诊断线: osc-spectra(5f80c54)                       ← merge 3
S1 ── 封闭质量裁决 ── particle-truth产品 ── flux ── cap-w512(e197fea)                        ← merge 4
```

> 注：phi-profiles(2b26cbc) 与 osc-spectra(5f80c54) 是 inner-band-A 下的**平行**分叉，互不包含；
> 密度图(f80d415) 与 BASE(af32ee9) 是六模型裁决下的**平行**分叉。

---

## A. 代码修复账（问题 → 修法 → 去向）

### A1. 已在主线，随 BASE 自动进入 phase-2 ✅

| commit | 日期 | 问题 | 修法 |
|---|---|---|---|
| `58dbd30` | 09-13 | 新版 JAX 移除了 `jnp.clip(a_min=…)` 用法，上游代码报错 | 替换为 `jnp.maximum`；补依赖 |
| `af42efb` | 09-19 | S1 runner 期望配额总数硬编码 `n_samples`，与 run options 脱节 | 从 run options 推导配额总数 |
| `cc5d87c` | 09-20 | grid-prior 证据里对网格 Laplacian 的求值方式在 jit/vmap 下不正确 | 改为闭包 + bare vmap 求值 |
| `8ad776b` | 09-20 | vmapped Laplacian 少传了 `phi_model` | 补传参 |

（另有 2020–2025 上游历史 fix 4 条：进度条、默认文件名、v_theta 计算、边界绘图，随 baseline 带入，无需决策。）

### A2. 随 4 条 merge 尖端带入 ✅（无需单独动作）

| commit | 日期 | 问题 | 修法 | 载体 |
|---|---|---|---|---|
| `d30601e` | 09-21 | `run_phi_profiles.sh` 在 pipefail 下 awk 提前退出触发 SIGPIPE | 规避早退 | merge 2 (phi链) |
| `109fdbb` | 09-21 | 固定 `CUDA_VISIBLE_DEVICES` 会撞占用的 GPU | 未设置时自动挑空闲 GPU | merge 2 (phi链) |
| `0d83871` | 09-21 | 密度谱诊断各自加载真值网格，版本漂移 | 统一用已提交的 particle-truth grid 产品 + 共享 loader | merge 3 (谱诊断) |

### A3. 第 6 轮未决实验的修复（失败 run → 修复 commit 已就绪）⏳

这些**不进合并基线**，作为新项目 round-1 的 cherry-pick 素材：

| commit | 日期 | 失败 run | 问题（来自 traceback） | 修法 | 分支 |
|---|---|---|---|---|---|
| `7de9703` | 09-22 | `651f025e` (44min) | `_unpack_phi_batch` 对 5 元 batch 解包 7 值 → `ValueError` | len==5/else 分支补 `q_pair = None` | osc-pair |
| `002dcdd` | 09-21 | `e18f6d6d` (cancelled) | `_linear_layers` 误收非 Linear 叶子 | 树遍历只过滤 Linear 模块 | spectral-norm |
| `51fd27e` | 09-22 | `65ab84cb` (44min) | anneal 后 `lambda_` 是 traced array，`(lambda_ != 0)` 触发 `TracerBoolConversionError` | 保持 `lambda_` 为 Python float；修 midpoint 测试容差 | cosine-anneal |
| （未提交） | 09-21 | 同 `002dcdd` | 与 `7de9703` 同源的 `q_pair=None` 2 行修复 | **需先提交**（在 chat_58d7329a worktree） | spectral-norm |

> ⚠️ 另有非 fix 前缀但值得记录的修复：`aab0da4`（XLA slow-fusion，单 kernel 拆小 + 粒子循环移出 jit，已在 merge 4 路径上）、cap-w512 线上的 phi_direct 单位校准 debug commit（同在 merge 4 路径上）。

---

## B. 图表资产账（按图族，全部取各线**最新版**）

### B1. 粒子真值产品与真值图（工具库，强候选进长期流程）

| 脚本 | 最新版载体 | 去向 | 最新渲染 run |
|---|---|---|---|
| `build_total_matter_truth.py` | cap-w512 | merge 4 | —(数据构建) |
| `build_particle_truth_grids.py` | cap-w512 | merge 4 | —(数据构建) |
| `particle_truth.py`（共享库，含 jit 优化） | cap-w512 | merge 4 | — |
| `plot_pt_potential_2d.py` | cap-w512 | merge 4 | 0e812d65 系 |
| `plot_pt_potential_norm.py`（flux dM + 外区负密度） | cap-w512 | merge 4 | 76ac071b |
| `plot_particle_truth_2d.py` | osc-spectra（比 cap-w512 新） | merge 3 | e4d4db5d |
| `plot_s1_potential_2d.py` / `plot_s1_density_angular.py` / `plot_s1_potential_diagnostics.py` | cap-w512 | merge 4 | 0e812d65 / 76ac071b 系 |

**建议**：整族进长期流程 —— 这是所有裁决图的真值底座；两处 `plot_*_2d` 并入时以 osc-spectra 版为最新。

### B2. 裁决套件（每轮实验后的标准验收图，强候选）

| 脚本 | 最新版载体 | 去向 | 最新渲染 run |
|---|---|---|---|
| `plot_pt_adjudication.py`（六模型 vs 锚点） | BASE | 自动 | 369e3eae |
| `plot_shell_error_pairs.py` | BASE | 自动 | 780137d7 |
| `plot_pt_generations.py` | BASE | 自动 | 780137d7 |
| `plot_pt_density_resid_lambda.py`（密度图三模型版） | density-maps 线 | **merge 1** | 553b68f8 |
| `plot_enclosed_mass.py` / `plot_df_constraints.py`（Phase-1 审计） | BASE | 自动 | — |
| `compare_phi_truth.py` + `plot_s1_density_2d.py`（3 模型旧裁决） | grid-prior-adjudication 线 | **OUTSIDE** | e8c61f04 / 2b4af62c |

**建议**：前五个进长期流程；`compare_phi_truth`/`plot_s1_density_2d` 已被六模型裁决实质取代（同一功能更新版），归档即可。

### B3. 振荡诊断（第 6 轮的直接工具）

| 脚本 | 最新版载体 | 去向 | 最新渲染 run |
|---|---|---|---|
| `plot_density_spectrum.py` + `run_density_spectrum.sh` | osc-spectra | merge 3 | e4d4db5d |
| `orx_figstyle.py`（统一样式库） | osc-spectra（vendored 自 paired-shell） | merge 3（merge 2 也带同源副本，应无冲突） | — |

**建议**：进长期流程 —— round-1 的 osc-pair 校准直接依赖它。

### B4. 角向结构图版

| 脚本 | 最新版载体 | 去向 | 最新渲染 run |
|---|---|---|---|
| `plot_phi_profiles.py`（divmod 面板配对修正版） | phi-profiles | **merge 2** | c2d66343 |
| （谱系内含 shell-error 累积版、per-model 轴标、paired-shell） | phi 链祖先 | 随 merge 2 | ca216614 / b727ec6a / 03ab1a32 |

**建议**：merge 2 整链带入（最新版即链条尖端）。

### B5. 清洗时代专用图（2026-09-17 前后，第 1 轮数据决策期）—— 不建议并入

| 脚本 | 载体 | 性质 |
|---|---|---|
| `df_clump_fig_reference.py` / `df_clump_fig_uniform.py` / `df_clean_truth_pid.py` / `df_clean_truth_comparison.py` / `df_phase1_vt_diagnostic.py` | clump 系分支 | clump/清洗决策期一次性诊断，结论已固化进冻结数据集 |
| `df_phase1_radial_cv_figures.py`（+ metrics） | radial-cv 线 | Phase-1 DF 审计时代产物，报告素材已入 talk |

### B6. Pre-orx 旧线（codex 时代）—— 不属于 phase-2

`plot_data_corner.py`、`plot_flow_projections.py`、`plot_gaia.py`、旧 `plot_potential.py`（2026-07/08，`halo12-outer-clump-removal` 等分支）：属另一谱系，留在分支上即可。

---

## C. 待你拍板的决策点

1. **合并集确认**：BASE + 4 尖端（f80d415 / 2b26cbc / 5f80c54 / e197fea）—— B1–B4 全部覆盖，是否同意？
2. **OUTSIDE 两件**：`compare_phi_truth.py` / `plot_s1_density_2d.py` 按"已被取代"归档，不并入 —— 是否同意？
3. **B5 清洗时代 + B6 旧线**不并入 —— 是否同意？
4. **长期流程化**：B1/B2/B3 进 phase-2 基线后，是否同时把"每轮实验后必跑的验收图"（pt-adjudication + shell-error + density-maps + density-spectrum）写成一个统一 runner 脚本（如 `run_phase2_adjudication.sh`），作为 phase-2 的标准验收步骤？（这是"把图表加到长期流程"的落点，可在基线上做一个提交。）
5. **A3 的未提交 2 行 fix**：现在提交到 spectral-norm 分支？（无损，且是 round-1 cherry-pick 的前提。）

---

## D. 裁决记录（2026-09-22）：图表不整体合并，改为指定目录收存 keep 清单

**裁决**：B1–B4 各图族不随 merge 并入；仅把下列 6 项放进基线分支的 `scripts/auriga/keep/` 目录（暂不接入主流程，每件标注来源与用法）。

| # | 需求描述 | 定位结果 | 最新版载体 | 备注 |
|---|---|---|---|---|
| 1 | radial_rbin_velocity：三个速度的 r-bin 分布 | `df_radial_velocity_marginals.py`（三速度边际 vs r-bin，模型对比）；同族另有 `df_radial_rbin{,_own,_mass}.py`（质量/计数直方图） | 各 radial 分支（09-17/18） | 确认：只要 velocity 版，还是整族？ |
| 2 | density-2d / potential-2d：真值 vs 模型对比 | `plot_pt_potential_2d.py`、`plot_particle_truth_2d.py`、`plot_s1_density_2d.py`、`plot_pt_density_resid_lambda.py`（λ轮三模型密度图） | cap-w512 / osc-spectra / density-maps 线 | 依赖 `particle_truth.py` + `orx_figstyle.py` |
| 3 | clump_smooth65 corner preview（处理前看数据形状） | **不在 git**（session 一次性脚本，未提交；产物应在服务器 run 目录）；codex 时代另有 `plot_data_corner.py`（用途类似） | — | 需补：服务器找回，或重写一个 ~40 行 corner 预览 |
| 4 | 数据准备 + 处理好的数据 | `prepare_data.py`（OLD `halo12_all_mass_clean_outer_clump_smooth.h5` → 冻结 `halo12-clean-smooth.h5`）+ `run_w1024_csmooth.sh` + `orx_data_path.txt` | **已在主线上（BASE 自带）** ✅ | 数据本体在服务器 DATA_ROOT，smoke 副本在 `data/auriga/halo12-smoke.h5` |
| 5 | auriga 数据做 density-2d 真值的构建脚本 | `build_total_matter_truth.py` + `build_particle_truth_grids.py` + `particle_truth.py`（loader）+ `run_total_matter_inventory.sh` | cap-w512 尖端 | 即粒子真值产品族，为 #2 提供真值 |
| 6 | enclosed-mass 上下格式评估图 | `plot_pt_adjudication.py`（计算侧 `validate_enclosed_mass.py`）；你所见的图为 gridprior 轮版本 `pt-adjudication-gridprior.svg`（run `e8c61f04`），最新版为 λ轮六模型 | **已在 BASE** ✅ | 若指 2 行审计图则为 `plot_enclosed_mass.py`（也在 BASE） |

**最终裁决（同日补记）**：#1 质量与速度四件全收；#2 只收两个主力（`plot_pt_potential_2d` + `plot_pt_density_resid_lambda`）；#3 取消。
已落地：基线分支 `codex/phase2-baseline-2026-09-22`（自 af32ee9），keep 件收入 `scripts/auriga/keep/`（含 README 标注来源），路径层级已修补。

**附带效应**：合并集因此从「BASE + 4 尖端」缩小为「BASE + 按需拷贝」——
- 基线 = `orx/lambda-sweep-10…`（纯主线）或 `af32ee9`（BASE，自带 #4/#6）；
- 其余 keep 项从各载体分支**拷贝**进 `keep/`（不做 merge，不动冻结分支）。

---

## 附：渲染图的查看路径

渲染图不入库，在服务器 run 目录：`/home/qiutao/.orx/runs/<runId>/`（如 `e4d4db5d…` 为密度谱图、`c2d66343…` 为 phi-profiles、`369e3eae…` 为六模型裁决、`553b68f8…` 为三模型密度图）。日志经 `orx logs <runId>` 查看。
