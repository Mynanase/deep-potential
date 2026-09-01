# 服务器 Agent 操作指南

训练和高成本评估使用独立后台进程；Jupyter 只读取保存后的结果并调试单张图。

## 准备

```bash
git fetch origin
git switch codex/halo12-outer-clump-removal
git pull --ff-only

conda activate dp-jax
pip install -e ".[operations,notebook,tracking]"
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## 创建一次实验

复制最接近的配置，并为这次尝试设置唯一目录：

```yaml
schema: dpjax.run.v2
name: halo12-static-baseline
case: halo12
output_dir: runs/halo12/static-baseline

df:
  model: configs/models/df/halo12_ffjord_compact_v1.yaml
  seed: 42

phi:
  model: configs/models/phi/halo12_static_mlp_v1.yaml
  seed: 42
```

一个 YAML 只对应一次具体尝试。换 seed 或模型时复制配置并更换
`name/output_dir`，不要增加 `trials` 层。

## 后台执行

完整运行推荐使用组合 launcher：

```bash
python -m experiments.launch all configs/runs/halo12_static_v1.yaml
```

只运行一条链时使用 `df-pipeline` 或 `phi-pipeline`；需要人工恢复、重算评估或
重画图时，使用对应的独立 launcher：

```bash
python -m experiments.launch df-pipeline configs/runs/halo12_static_v1.yaml
python -m experiments.launch phi-pipeline configs/runs/halo12_static_v1.yaml

python -m experiments.launch df configs/runs/halo12_static_v1.yaml
python -m experiments.launch eval-df configs/runs/halo12_static_v1.yaml
python -m experiments.launch plot-df configs/runs/halo12_static_v1.yaml
python -m experiments.launch phi configs/runs/halo12_static_v1.yaml
python -m experiments.launch eval-phi configs/runs/halo12_static_v1.yaml
python -m experiments.launch plot-phi configs/runs/halo12_static_v1.yaml
```

launcher 返回成功仅表示后台 worker 已经启动；训练或评估是否成功必须查看对应
日志，例如：

```bash
tail -f runs/halo12/static-baseline/logs/all.log
```

无需手写 `nohup`、日志重定向或 XLA 设置。未设置
`CUDA_VISIBLE_DEVICES` 时使用服务器/容器暴露的 GPU；需要限制 GPU 时在启动
launcher 的父环境中设置。

组合 worker 会为每个高成本阶段启动独立子进程，阶段结束后不会把同一个 JAX/GPU
状态带入下一阶段。它在第一个失败阶段停止，保留之前已经完成的产物，也不会自动
跳过已有 checkpoint。修复原因后，用失败阶段的独立命令恢复；训练 checkpoint
只有在保存配置兼容时才能通过 `execution.resume: true` 继续。

同一 run 中，活动 pipeline 会阻止其覆盖的单阶段任务；所有 eval 与 plot worker
也互斥，避免绘图读取半写评估文件或多个绘图进程同时覆盖 manifest/report。

`eval` 与 `plot` launcher 名称仍映射到 `experiments.run_eval` 和
`experiments.run_plot` 的兼容聚合流程；命令行直接调用 `run_plot` 时仍可使用
`--only`。新任务应使用上面的独立名称，不使用 `--stage`。

## W&B

```bash
wandb login
```

然后在 run YAML 中设置：

```yaml
logging:
  backend: wandb
  project: deep-potential
  mode: online
```

离线服务器使用 `mode: offline`，之后运行 `wandb sync`。不要把 API key 写入
YAML、日志或 Git。

## 产物

```text
runs/halo12/static-baseline/
├── run.yaml
├── logs/
├── df/
├── phi/
└── results/
    ├── data/
    ├── figures/
    ├── debug/
    ├── manifest.json
    └── report.md
```

旧的 `eval/`、`plots/` 和 `validation/` 仍可读取，但不会由新运行创建，旧结果
不需要迁移。

`results/data/df_*` 与 `results/data/phi_*` 是由 `run.yaml`、输入数据和已保存
checkpoint 生成的派生评估产物。可以分别用 `eval_df` 或 `eval_phi` 重算；
`plot_df` 与 `plot_phi` 只读取这些文件，缺失必需 artifact 时会立即失败，不会
隐式重新评估或训练。

真值验证不属于服务器 run 配置或日常评估入口。如需临时验证，修改并运行
`analysis/validate_plummer_truth.py` 或 `analysis/validate_auriga_truth.py`；脚本
会写入 `results/data/validation_<kind>_*.{json,npz}`。

## 分析与检查

```bash
jupyter lab notebooks/figure_debug.ipynb
pytest -q
```

Agent 不得提交数据、凭据、日志或 checkpoint，也不得把数据适配、绘图和 truth
验证重新放进 `dpjax`。
