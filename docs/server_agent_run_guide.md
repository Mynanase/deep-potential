# 服务器 Agent 操作指南

本文是服务器执行指南。训练和评估由独立进程完成；Marimo 只读取结果。
新增模型、数据、诊断或绘图时，先查阅 `docs/architecture_operation_guide.md`。

## 1. 架构约束

- `experiments.run_df`、`run_phi`、`run_eval` 是三个前台 worker；
- `experiments.launch` 是服务器后台执行的统一入口；
- 每个入口只接收一个 `configs/runs/*.yaml` 路径；
- `dpjax` 只接受 `(N, 6)` 相空间数组，不读取文件或生成图表；
- 数据适配、训练编排、评估、日志和绘图实现在 `experiments`；
- DF 和 Phi 必须属于同一 run/trial；
- `analysis/halo12.py` 不加载训练流程，也不写 checkpoint；
- `configs/models` 只保存模型结构与训练配方；
- `configs/runs` 保存数据、运行、评估、绘图和模型引用；
- 不重新引入旧的多参数 CLI、独立绘图脚本或每个实验一份 shell wrapper。

## 2. 服务器准备

```bash
git fetch origin
git switch codex/standalone-run-architecture
git pull --ff-only

conda activate dp-jax
pip install -e ".[operations,notebook,tracking]"

python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

数据和 checkpoint 不进入 Git，需要单独同步。

## 3. 创建运行配置

```bash
cp configs/runs/halo12_static_v1.yaml configs/runs/halo12_server_v1.yaml
```

编辑复制后的文件，至少检查：

```yaml
schema: dpjax.run.v1
name: halo12_server_v1
output_dir: runs/halo12_server_v1

data:
  path: data/auriga/halo12_all_mass.h5
  dataset: eta
  weight_dataset: tracer_weight
  split:
    validation_fraction: 0.25
    seed: 1042
  preprocessing:
    clip_sigma: 4.5
    jitter_std: 0.002
    transform:
      type: none
      dims: []

trials:
  trial_00:
    df:
      model: configs/models/df/halo12_ffjord_compact_v1.yaml
      seed: 42
    phi:
      model: configs/models/phi/halo12_static_mlp_v1.yaml
      seed: 42

logging:
  backend: wandb
  project: deep-potential
  mode: online
  # entity: your-user-or-team

execution:
  resume: false
  stages:
    df:
      multi_gpu: true
      log_every: 25
      checkpoint_every: 500
      checkpoints_to_keep: 3
    phi:
      multi_gpu: true
      log_every: 50
      checkpoint_every: 500
      checkpoints_to_keep: 3
```

需要修改模型参数时，在 `configs/models/{df,phi}` 复制一个新模型配方；仅针对
本次 run 的小型消融可使用 `model_overrides`。比较多组配对时增加 `trial_01` 等，
不要分别维护互不关联的 DF/Phi 输出路径。

## 4. 后台执行

以下命令可直接用于 fish，不需要手写 `nohup`、GPU 列表、XLA 设置或日志重定向：

```bash
python -m experiments.launch df configs/runs/halo12_server_v1.yaml
```

DF 成功后运行 Phi：

```bash
python -m experiments.launch phi configs/runs/halo12_server_v1.yaml
```

Phi 成功后运行评估：

```bash
python -m experiments.launch eval configs/runs/halo12_server_v1.yaml
```

项目默认设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。未设置
`CUDA_VISIBLE_DEVICES` 时使用服务器或容器暴露的全部 GPU；如需限制设备，应由
调度器或启动 launcher 的父环境负责。评估包含采样、score、梯度和 Hessian，
也可能需要 GPU。查看进度：

```bash
tail -f runs/halo12_server_v1/logs/df.log
```

launcher 会同时写入 `df.pid`、`phi.pid` 或 `eval.pid`，并拒绝重复启动仍在运行的
同一阶段。

### 启用 W&B

首次在服务器账户中执行：

```bash
pip install -e ".[operations,tracking]"
wandb login
```

然后在 run YAML 中设置 `logging.backend: wandb` 和 `mode: online`。没有网络时
使用 `mode: offline`，之后对阶段目录中的 `wandb/offline-run-*` 执行
`wandb sync`。密钥只保存在 W&B 用户配置或受保护的 `WANDB_API_KEY` 环境变量，
不要写入 Git、run YAML 或日志。

## 5. 输出结构

```text
runs/halo12_server_v1/
├── run.yaml
├── logs/
│   ├── df.log / phi.log / eval.log
│   └── df.pid / phi.pid / eval.pid
├── trial_00/
│   ├── df/
│   │   ├── config.yaml
│   │   ├── model_summary.yaml
│   │   ├── metrics.csv
│   │   └── ckpt/
│   ├── phi/
│   │   ├── config.yaml
│   │   ├── model_summary.yaml
│   │   ├── metrics.csv
│   │   └── ckpt/
│   ├── eval/
│   │   ├── df/
│   │   └── phi/
│   ├── plots/
│   └── validation/
│       └── auriga_truth/  # 仅模拟数据且显式启用时存在
└── summary/
```

首次执行会生成完整解析后的 `run.yaml`。已有 checkpoint 且
`execution.resume: false` 时程序会拒绝覆盖；续训时改为 `true`，但保存的配置必须
完全一致。科学参数发生变化时创建新的 `name/output_dir`。

## 6. Marimo 分析

```bash
marimo edit analysis/halo12.py
```

在页面中选择 `runs/halo12_server_v1`。页面展示已保存的训练曲线、DF/Phi 指标和
图片；没有结果时先检查 `run_eval` 是否完成，不要在 Marimo 中启动训练。

## 7. 验证和 Agent 边界

```bash
pytest -q
marimo check --strict analysis/halo12.py
```

Agent 操作时：

1. 先读根目录 `AGENTS.md`；
2. 模型参数修改 `configs/models`，数据和运行参数修改 `configs/runs`；
3. 不混用不同 trial 的模型；
4. 不绕过配置快照与 checkpoint 保护；
5. 不提交数据、凭据、日志和 checkpoint；
6. 新数据、图表和 truth 验证只修改 `experiments`，不要放进 `dpjax`；
7. 修改数值核心时同步更新数组级 `dpjax` 测试；
8. 保留工作区中与本任务无关的用户文件。
