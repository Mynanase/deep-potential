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
模拟 potential、acceleration 等 truth 不进入 run YAML 或 `run_eval.py`。

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
统一写入格式、manifest 和关闭 Figure。批量入口是 `run_plot`；Jupyter 每个
单元只调用一个 `render_figure`。

## 修改位置

- 模型、方程、导数：`dpjax/`
- 数据格式、筛选、预处理：`experiments/datasets/`
- 训练和评估执行：`experiments/workflows/`
- 持久化产物读取与诊断计算：`experiments/diagnostics/`
- 图表：`experiments/plotting/`
- 模拟 truth：`experiments/validation/`
- 交互组合：`analysis/`

新增数据最终只需提供有限的 `(N, 6)` 数组，列顺序为
`[x, y, z, vx, vy, vz]`；文件读取和源数据特有逻辑不进入 `dpjax`。

## 执行与验证

```bash
conda activate dp-jax

python -m experiments.launch df configs/runs/plummer_rcut_cut.yaml
python -m experiments.launch phi configs/runs/plummer_rcut_cut.yaml
python -m experiments.launch eval configs/runs/plummer_rcut_cut.yaml
python -m experiments.launch plot configs/runs/plummer_rcut_cut.yaml

tail -f runs/plummer_rcut/cut-baseline/logs/phi.log
jupyter lab notebooks/figure_debug.ipynb
```

launcher 负责后台脱离、日志和 PID；项目默认设置
`XLA_PYTHON_CLIENT_PREALLOCATE=false`。GPU 可见性由启动 launcher 的父环境控制。

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
