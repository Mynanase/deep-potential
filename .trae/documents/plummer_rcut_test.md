# Plummer r-cut 测试：中心 stars 切除对 DF / Phi 拟合的影响

## Summary

模拟 Auriga 场景（全部 stars 训练时 DF 贡献被中心 stars 主导、外围被忽视），用 Plummer 球做一次受控实验：

1. 生成全量数据 `r ∈ [0, 10]`（N=524288, seed=2024）；**切心数据 = 全量数据中 r ≥ 1.0 的子集（同一实现，非独立采样）**，规模以过滤结果为准（期望 ≈33.9 万，即 Plummer 中 M(<1)/M_total=1/(2√2)≈0.354）。
2. 在 marimo 中绘制两份数据的 DF 分布（**R-z by φ** 与 **三个速度 marginals by r/θ/φ**），对比全量与外围分布。
3. 对两种情况（full / cut）分别做 **DF 拟合**（FFJORD）与 **Phi 拟合**（MLP），模型配方完全一致，仅数据不同。
4. 用现有的 auriga 风格 DF 评估（`evaluate_auriga_df` + `plot_auriga_df_ensemble`）产出 rz-by-phi 与速度 marginals 的 data-vs-model 图，用现有 plummer Phi 评估产出径向剖面图；最终对比图在 marimo 中汇总展示。

GPU 训练在用户终端执行（沙盒无 GPU）。

## 当前状态分析（已探明）

- 数据生成：[gendata_plummer.py](file:///localdisk/kosmos/my-deep-potential/experiments/gendata_plummer.py) + [plummer.py](file:///localdisk/kosmos/my-deep-potential/experiments/datasets/plummer.py) 已存在；`sample_radius` 只支持 `max_dist` 截断。**切心通过 gendata 层对采样结果按 r 过滤实现，`datasets/plummer.py` 无需改动**。
- 训练入口：`experiments.run_df` / `run_phi` / `run_eval`，各只接收一个 `configs/runs/*.yaml`（见 [server_agent_run_guide.md](file:///localdisk/kosmos/my-deep-potential/docs/server_agent_run_guide.md)）。
- 模型配方已存在：`configs/models/df/plummer_ffjord_v1.yaml`、`configs/models/phi/plummer_mlp_v1.yaml`。
- 用户要的 DF 分布图（`df_spatial_rz_by_phi`、`velocity_marginals_by_{r,theta,phi}`）已由 [evaluation/auriga_df.py](file:///localdisk/kosmos/my-deep-potential/experiments/workflows/evaluation/auriga_df.py) 的 `evaluate_auriga_df` + [plotting/diagnostics.py](file:///localdisk/kosmos/my-deep-potential/experiments/plotting/diagnostics.py) 的 `plot_auriga_df_ensemble` 产出；`load_auriga_snapshot` 对只有 `eta` 数据集的 H5（即 Plummer 文件）直接可用。
- `run_eval` 的 `system: halo` 路径（`_run_halo_df`）调用上述流程；`radial_edges` 目前硬编码为 Auriga 默认（到 75），对 Plummer（r≤10）不适配，需小改暴露为配置项。
- phi 评估 `run_eval_phi` 用 `system` 决定 G：`halo→4.3e-6`、其余→`1.0`；支持 `gravitational_constant` 显式覆盖。
- marimo 模式参考 [analysis/halo12.py](file:///localdisk/kosmos/my-deep-potential/analysis/halo12.py)：只读 artifact，不训练。

## 决策（Assumptions & Decisions）

- **切心方式**：cut 数据 = full 数据按 `r ≥ 1.0` 过滤后的子集（**同一实现**）。生成方式：`gendata_plummer` 以相同 `--seed` 采样与 full 完全相同的 N 个样本后，丢弃 `r < 1.0` 的粒子再保存——故 cut 与 full 中的外围粒子逐颗一致。cut 规模随采样浮动（期望 ≈33.9 万），与 Auriga 场景（cut 后数据变少）一致。
- **参数**：`r_cut=1.0`（Plummer 尺度半径）、`N=2^19=524288`、`max_dist=10`、数据 seed=2024（与历史 Plummer 数据一致）、模型 seed=42。
- **评估方式**：`evaluation.system: halo`（得到 rz-by-phi + 速度 marginals），同时 `evaluation.phi.gravitational_constant: 1.0` 保证 Phi 评估用 Plummer 无量纲单位。
- **日志**：`logging.backend: csv`（本测试不依赖 wandb）。
- **公平性**：full 与 cut 两份 run config 使用完全相同的模型配方与超参，仅 `data.path` 不同。
- **命名**：run 名 `plummer_rcut`，output_dir 分别为 `runs/plummer_rcut/full` 与 `runs/plummer_rcut/cut`（`data.path` 是 run 级配置，两种数据必须拆成两个 run config，各含 1 个 `trial_00`）。
- 两 case 数据独立采样、独立训练；`resume: false`，重复执行需换新目录（受 `validate_stage_start` 保护）。

## 具体改动

### 1. `experiments/datasets/plummer.py` — **无需改动**

切心在 gendata 层用过滤实现（采样 → 算 r → mask），采样器保持只支持 `max_dist`，向后零风险。

### 2. `experiments/gendata_plummer.py` — 新增 `--min-dist`（采样后过滤）

- 增加 `--min-dist`（默认 0.0），校验 `0 ≤ min_dist < max_dist`。
- 采样逻辑不变（`sample_plummer(total_n, max_dist=..., rng=rng)` 仍生成 `total_n` 个样本）；随后若 `min_dist > 0`：`r = norm(eta[:, :3], axis=1)`，`eta = eta[r >= min_dist]`，打印过滤前后数量。
- 注意：`--total-n` 语义为"采样总数"，实际保存数 ≤ `total_n`（过滤后），打印实际数量，避免用户误以为数据缺失。
- `--seed` 语义不变 → **cut 与 full 用相同 seed 生成的样本逐颗一致**，cut 即 full 的过滤子集。

### 3. `experiments/run_eval.py` — `_run_halo_df` 暴露 `radial_edges`

- `from experiments.workflows.evaluation.auriga_df import DEFAULT_RADIAL_EDGES`（或内联默认列表）。
- `_run_halo_df` 中 `radial_edges = np.asarray(config.get("radial_edges", DEFAULT_RADIAL_EDGES), dtype=np.float64)` 并传给 `evaluate_auriga_df`。不配置时行为与现状完全一致（向后兼容）。

### 4. 新 run 配置（两个文件）

`configs/runs/plummer_rcut_full.yaml` 与 `configs/runs/plummer_rcut_cut.yaml`，结构：

```yaml
schema: dpjax.run.v1
name: plummer_rcut_full          # cut 版本: plummer_rcut_cut
output_dir: runs/plummer_rcut/full   # cut 版本: runs/plummer_rcut/cut
data:
  path: data/plummer_n524288_train.h5            # cut: data/plummer_n524288_rcut1.0_train.h5
  dataset: eta
  split: { validation_fraction: 0.25, seed: 2024 }
  preprocessing:
    clip_sigma: 0.0
    transform: { type: none, dims: [] }
trials:
  trial_00:
    df:
      model: configs/models/df/plummer_ffjord_v1.yaml
      seed: 42
      model_overrides: { train: { epochs: 64 } }   # 配方默认 32，本测试统一提到 64，两 case 一致
    phi:
      model: configs/models/phi/plummer_mlp_v1.yaml
      seed: 42
logging: { backend: csv, project: deep-potential }
execution:
  resume: false
  stages:
    df:  { multi_gpu: true, log_every: 25, checkpoint_every: 500, checkpoints_to_keep: 3 }
    phi: { multi_gpu: true, log_every: 50, checkpoint_every: 500, checkpoints_to_keep: 3 }
evaluation:
  system: halo        # 触发 rz-by-phi + 速度 marginals 评估
  df:
    enabled: true
    radial_edges: [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    n_samples_per_model: 262144
    n_score_points: 32768
    score_batch_size: 1024
    seed: 42
    n_theta_bins: 6
    n_phi_bins: 8
    n_velocity_bins: 64
    spatial_r_bins: 48
    spatial_z_bins: 48
    spatial_r_max: 10.0
    spatial_z_max: 10.0
    spatial_min_cell_count: 5
  phi:
    enabled: true
    gravitational_constant: 1.0    # 覆盖 system=halo 的 G，保持 Plummer 无量纲
    n_eval: 32768
    batch_size: 4096
    seed: 0
    r_min: 0.001
    r_max: 10.0
    n_r: 256
    r_ref: 1.0
plots:
  formats: [png]
  df:  { dpi: 150 }
  phi: { dpi: 180, plot_overview: true, slice_grid: 128, slice_rmax: 10.0 }
```

### 5. 新 marimo 文件 `analysis/plummer_rcut.py`

只读数据/artifact，不训练。结构：

- **Cell 0（导入）**：`mo`、`numpy`、`matplotlib`、`load_eta_h5`、`resolve_path`、`cartesian_to_spherical_phase_space`、`conditional_velocity_diagnostics`、`cylindrical_rz_density_by_phi`、`plot_auriga_df_ensemble`、`Path`、`json`。
- **Cell 1（路径）**：`DATA_PATHS = {"full": ..., "cut": ...}`、`RUN_DIRS = {"full": "runs/plummer_rcut/full", "cut": ...}`（marimo 文本框可改）。
- **Cell 2（Part A｜原始数据 DF 分布，训练前即可用）**：
  - 加载两 H5；打印 N 与 r 范围。
  - **R-z by φ**：用 `cylindrical_rz_density_by_phi` 计算两 case 的 reference 密度（phi/r/z 边界与 run 配置一致），并排渲染 2×n_phi 热图（共享色标）。
  - **速度 marginals by r/θ/φ**：用 `conditional_velocity_diagnostics` 计算两 case 的 reference 直方图，同一子图内 full(solid) vs cut(dashed) 叠画（rows=各坐标 bin，cols=3 速度分量）。
  - 附一张**径向累积质量占比**小图，直接展示"中心主导"这一动机。
- **Cell 3（Part B｜拟合后对比，依赖 run_eval 产物）**：
  - 对两 case 分别调用 `plot_auriga_df_ensemble(metrics_json, diagnostics_npz, fig_dir=...)`，把生成的 `density_profile.png`、`df_spatial_rz_by_phi.png`、`velocity_marginals_by_{r,theta,phi}.png` 用 `mo.image` 展示（data vs model）。
  - 展示 phi 评估产物（`plots/phi/` 下已由 run_eval_phi 生成的 `density_profile.png`、`potential_density_overview.png`、`phi_slice_xy.npz` 渲染图）。
  - 汇总表：两 case 的 `density_profile.log10_rmse_dex`、速度 marginal W1 中位数等数值对比（读 `auriga_df_metrics.json`）。

## 执行步骤（用户终端，GPU）

```bash
conda activate dp-jax

# 1) 生成数据（CPU，两条命令；--min-dist 为采样后过滤，cut 是 full 的逐颗子集）
python -m experiments.gendata_plummer --total-n 524288 --max-dist 10.0 --seed 2024 \
  --train-out data/plummer_n524288_train.h5
python -m experiments.gendata_plummer --total-n 524288 --max-dist 10.0 --min-dist 1.0 --seed 2024 \
  --train-out data/plummer_n524288_rcut1.0_train.h5

# 2) DF 拟合（8 卡，两 case 可先后跑）
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_df configs/runs/plummer_rcut_full.yaml > logs/plummer-rcut-full-df.log 2>&1 < /dev/null &
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_df configs/runs/plummer_rcut_cut.yaml  > logs/plummer-rcut-cut-df.log  2>&1 < /dev/null &

# 3) Phi 拟合（DF 完成后）
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_phi configs/runs/plummer_rcut_full.yaml > logs/plummer-rcut-full-phi.log 2>&1 < /dev/null &
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_phi configs/runs/plummer_rcut_cut.yaml  > logs/plummer-rcut-cut-phi.log  2>&1 < /dev/null &

# 4) 评估（DF+Phi 完成后）
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_eval configs/runs/plummer_rcut_full.yaml > logs/plummer-rcut-full-eval.log 2>&1 < /dev/null &
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_eval configs/runs/plummer_rcut_cut.yaml  > logs/plummer-rcut-cut-eval.log  2>&1 < /dev/null &

# 5) 绘图
marimo edit analysis/plummer_rcut.py
```

数据生成、marimo 绘图可由 Agent 在沙盒完成；DF/Phi 训练与评估需在用户终端（GPU）执行。

## 验证

- `pytest -q tests/test_plummer.py` — ✅ 7 passed（`datasets/plummer.py` 未改）。
- 数据自检 — ✅ full N=524288 含 187989 颗 r<1；cut N=**336299** 全部 r≥1.0，且 **cut == full[r≥1.0] 逐行一致**（同一 seed 采样 + 过滤）。
- `marimo check --strict analysis/plummer_rcut.py` — ✅ 通过。
- run 配置加载 + `prepare_run` 快照 — ✅ 两个 run 均生成 `run.yaml` 快照。
- 训练：两 case `metrics.csv` 收敛、`ckpt/` 存在；`run_phi`/`run_eval` 依赖前置 stage 完成（失败时会明确报错）。⚠️ 需 GPU（用户终端执行）。
- 评估产物：`runs/plummer_rcut/{full,cut}/trial_00/{eval/df/auriga_df_metrics.json, auriga_df_diagnostics.npz, plots/df/*.png, eval/phi/*, plots/phi/*.png}` 存在。⚠️ 需 GPU。
- 最终在 marimo 中对比 full vs cut 的 DF 分布与拟合质量。
