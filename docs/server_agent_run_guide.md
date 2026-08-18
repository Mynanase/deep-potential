# 服务器 Agent 操作指南

训练和高成本评估使用独立后台进程；Marimo 只读取保存后的结果。

## 准备

```bash
git fetch origin
git switch codex/standalone-run-architecture
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

```bash
python -m experiments.launch df configs/runs/halo12_static_v1.yaml
python -m experiments.launch phi configs/runs/halo12_static_v1.yaml
python -m experiments.launch eval configs/runs/halo12_static_v1.yaml
```

按 DF、Phi、eval 的顺序，在前一阶段完成后启动下一阶段。日志位于：

```bash
tail -f runs/halo12/static-baseline/logs/df.log
```

无需手写 `nohup`、日志重定向或 XLA 设置。未设置
`CUDA_VISIBLE_DEVICES` 时使用服务器/容器暴露的 GPU；需要限制 GPU 时在启动
launcher 的父环境中设置。

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
├── eval/
│   ├── df_metrics.json
│   ├── df_diagnostics.npz
│   ├── df_samples.npz
│   ├── phi_metrics.json
│   └── phi_diagnostics.npz
└── plots/
```

旧的 `trial_00/`、`eval/df/`、`eval/phi/` 和 `auriga_df_*` 产物不属于当前
结构。已有重要结果需显式迁移或重新运行 evaluation。

真值验证不属于服务器 run 配置或 `run_eval.py`。如需临时验证，修改并运行
`analysis/validate_plummer_truth.py` 或 `analysis/validate_auriga_truth.py`；脚本
会另外写入 `validation/<kind>/metrics.json` 和 `diagnostics.npz`。

## 分析与检查

```bash
marimo edit analysis/halo12.py
pytest -q
marimo check --strict analysis/halo12.py analysis/plummer_rcut.py
```

Agent 不得提交数据、凭据、日志或 checkpoint，也不得把数据适配、绘图和 truth
验证重新放进 `dpjax`。
