# 架构与功能扩展操作指南

本项目采用三层结构：`dpjax` 只处理数组和数值模型，`experiments` 负责数据、
训练、产物和绘图，Jupyter 只读取保存结果并定位单张图。

## 配置边界

`configs/models` 保存可复用的网络、loss、optimizer 和训练超参数；
`configs/runs` 中每个 YAML 只描述一次具体实验。一次实验只有一个输出目录，DF
与 Phi 在该目录下配对，不存在 `trials` 或 `trial_00`：

```yaml
schema: dpjax.run.v2
name: cut-baseline
case: cut
output_dir: runs/plummer_rcut/cut-baseline

data:
  path: data/plummer_n524288_rcut1.0_train.h5
  dataset: eta

df:
  model: configs/models/df/plummer_ffjord_v1.yaml
  seed: 42

phi:
  model: configs/models/phi/plummer_mlp_v1.yaml
  seed: 42
```

需要更换 seed、模型或超参数时复制一份 run YAML，并使用新的 `name` 与
`output_dir`，例如 `cut-seed43`。`case` 是用于分析分组的元数据，产物位置以
`output_dir` 为准。

## 输出约定

```text
runs/<experiment>/
├── run.yaml
├── logs/
├── df/                 # checkpoint、模型状态、训练指标
├── phi/                # checkpoint、模型状态、训练指标
└── results/
    ├── data/             # flat JSON/NPZ 与评估配置
    ├── figures/          # 正式 PNG/PDF
    ├── debug/            # Jupyter 调试图
    ├── manifest.json
    └── report.md
```

新运行不创建 `eval/`、`plots/` 或多层 validation；读取器仍兼容这些旧目录。
`results/data/df_*` 与 `results/data/phi_*` 是从不可变 `run.yaml`、输入数据和
checkpoint 派生的评估产物，可以独立重算，不需要重新训练。模拟 potential、
acceleration 等 truth 不进入 run YAML 或日常评估入口。

## 诊断与绘图

DF 产物按需读取：

```python
from experiments.diagnostics import (
    load_df_diagnostics,
    load_df_metrics,
    load_df_samples,
)

metrics = load_df_metrics(eval_dir)
diagnostics = load_df_diagnostics(eval_dir)
samples = load_df_samples(eval_dir)  # 仅在确实需要大数组时调用
```

缺失文件会抛出 `FileNotFoundError`。绘图函数位于 `experiments.plotting`：

```python
plot_density_profile(diagnostics)
plot_cylindrical_rz_density(diagnostics)
plot_velocity_marginals(diagnostics, "r")
plot_score_distribution(diagnostics)

plot_potential_profile(radius, model_potential, truth_potential=truth)
plot_radial_acceleration_profile(radius, model_acceleration)
plot_mass_density_profile(radius, model_density, truth_density=truth_density)
plot_mass_density_residual(radius, model_density, truth_density)
plot_potential_slice(x, y, model_potential)
plot_mass_density_slice(x, y, model_density)
```

Figure builder 只接收内存数据并返回 `Figure`，不读文件、不保存、不调用
`plt.close()`。`FigureRegistry` 负责按名字装配数据和 builder，`FigureWriter`
统一写入格式、manifest 和关闭 Figure。日常批量入口是 `plot_df` 与
`plot_phi`；它们只消费本阶段已保存的评估产物，缺少必需文件时立即报错，
不会隐式训练或评估。Jupyter 每个单元只调用一个 `render_figure`。

## 修改位置

- 模型、方程、导数：`dpjax/`
- 数据格式、筛选、预处理：`experiments/datasets/`
- 训练和评估执行：`experiments/workflows/`
- 持久化产物读取与诊断计算：`experiments/diagnostics/`
- 图表：`experiments/plotting/`
- 模拟 truth：`experiments/validation/`
- 一次性数值验证：`analysis/`

新增数据最终只需提供有限的 `(N, 6)` 数组，列顺序为
`[x, y, z, vx, vy, vz]`；文件读取和源数据特有逻辑不进入 `dpjax`。

## 执行与验证

```bash
conda activate dp-jax

# 日常推荐：每一阶段都可单独重跑
python -m experiments.run_df configs/runs/plummer_rcut_cut.yaml
python -m experiments.eval_df configs/runs/plummer_rcut_cut.yaml
python -m experiments.plot_df configs/runs/plummer_rcut_cut.yaml
python -m experiments.run_phi configs/runs/plummer_rcut_cut.yaml
python -m experiments.eval_phi configs/runs/plummer_rcut_cut.yaml
python -m experiments.plot_phi configs/runs/plummer_rcut_cut.yaml

# 前台组合入口
python -m experiments.run df configs/runs/plummer_rcut_cut.yaml
python -m experiments.run phi configs/runs/plummer_rcut_cut.yaml
python -m experiments.run all configs/runs/plummer_rcut_cut.yaml

# 后台组合入口
python -m experiments.launch all configs/runs/plummer_rcut_cut.yaml

tail -f runs/plummer_rcut/cut-baseline/logs/all.log
jupyter lab notebooks/figure_debug.ipynb
```

组合入口不会把所有 JAX 工作装进同一进程：每个 worker 都在新的子进程中运行，
使 GPU/JAX 状态在阶段之间隔离。它采用 fail-fast 语义：第一个非零退出码会停止
后续阶段，已经完成的产物保留。组合入口不会自动跳过已有训练；失败后应先用对应
的 `run_df`、`eval_df`、`plot_df`、`run_phi`、`eval_phi` 或 `plot_phi` 单阶段命令
恢复。只有在配置与已有 checkpoint 兼容时才设置 `execution.resume: true`。
`df`/`phi` 组合要求相应评估已启用；`phi` 要求已有 DF checkpoint，`all` 还要求
Phi 使用本次 run 的 DF。故意关闭评估或通过 `phi.df_run` 组合不同 run 时，应使用
适合的单阶段入口或 `phi` 组合，而不是 `all`。

launcher 负责后台脱离、日志和 PID；单阶段名称为 `df`、`eval-df`、`plot-df`、
`phi`、`eval-phi`、`plot-phi`，组合名称为 `df-pipeline`、`phi-pipeline`、`all`。
launcher 返回成功仅表示后台 worker 已启动，最终结果以对应日志为准。项目默认
设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。GPU 可见性由启动 launcher 的父
环境控制。

`experiments.run_eval` 与 `experiments.run_plot --only` 只保留为旧聚合流程的
兼容入口；新脚本不要使用不存在的 `--stage` 选择器，而应直接调用四个独立的
eval/plot 模块。launcher 的旧名称 `eval`、`plot` 也仅用于这些兼容入口。

可选真值检查不使用配置文件。直接修改一次性脚本顶部的实验路径和数值参数：

```bash
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
```

脚本只在 `results/data/` 生成 flat validation JSON/NPZ；通用图不依赖解析真值。
提交前运行：

```bash
pytest -q
ruff check dpjax experiments analysis tests
```
