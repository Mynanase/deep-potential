# 架构与功能扩展操作指南

本项目采用三层结构：`dpjax` 只处理数组和数值模型，`experiments` 负责数据、
训练、产物和绘图，`analysis` 中的 Marimo 只读取结果并组合展示。

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
├── eval/
│   ├── df_config.yaml
│   ├── df_metrics.json
│   ├── df_diagnostics.npz
│   ├── df_samples.npz
│   ├── phi_config.yaml
│   ├── phi_metrics.json
│   └── phi_diagnostics.npz
└── plots/
```

`eval/` 不再按 DF/Phi 建子目录。模拟 potential、acceleration 等 truth 不进入
通用诊断文件，也不进入 run YAML 或 `run_eval.py`。

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

这些函数只接收内存数据并返回 `Figure`，不读文件、不保存、不调用
`plt.close()`。Marimo 决定加载哪些实验、如何组合以及展示哪些图。当前没有
“一次生成全部默认图”的聚合入口；未来如需增加，应由 workflow 根据配置调用
独立函数，而不是放入 plotting 包。

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

tail -f runs/plummer_rcut/cut-baseline/logs/phi.log
marimo edit analysis/plummer_rcut.py
```

launcher 负责后台脱离、日志和 PID；项目默认设置
`XLA_PYTHON_CLIENT_PREALLOCATE=false`。GPU 可见性由启动 launcher 的父环境控制。

可选真值检查不使用配置文件。直接修改一次性脚本顶部的实验路径和数值参数：

```bash
python analysis/validate_plummer_truth.py
python analysis/validate_auriga_truth.py
```

脚本只生成 `validation/<kind>/metrics.json` 与 `diagnostics.npz`；绘图仍由
Marimo 调用独立 Figure 函数组合完成。
提交前运行：

```bash
pytest -q
marimo check --strict analysis/halo12.py analysis/plummer_rcut.py
ruff check dpjax experiments analysis tests
```
