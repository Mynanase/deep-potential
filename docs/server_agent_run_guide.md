# 服务器 Agent 操作指南

本文是当前架构的唯一操作指南。训练和评估由独立进程完成；Marimo 只读取结果。

## 1. 架构约束

- `experiments.run_df`、`run_phi`、`run_eval` 是唯一支持的训练/评估入口；
- 每个入口只接收一个 `configs/runs/*.yaml` 路径；
- 训练、评估、配置与日志实现在 `dpjax.workflows`；
- DF 和 Phi 必须属于同一 run/trial；
- `analysis/halo12.py` 不加载训练流程，也不写 checkpoint；
- 不重新引入旧的多参数 CLI、独立绘图脚本或每个实验一份 shell wrapper。

## 2. 服务器准备

```bash
git fetch origin
git switch codex/standalone-run-architecture
git pull --ff-only

conda activate dp-jax
pip install -e ".[notebook,tracking]"

python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

数据和 checkpoint 不进入 Git，需要单独同步。

## 3. 创建运行配置

```bash
cp configs/runs/halo12_static_v1.yaml configs/runs/halo12_server_v1.yaml
```

编辑复制后的文件，至少检查：

```yaml
name: halo12_server_v1
output_dir: runs/halo12_server_v1

data:
  path: data/auriga/halo12_all_mass.h5
  dataset: eta

trials:
  trial_00:
    df:
      config: configs/df_halo12_ffjord_v23_mass.yaml
      seed: 42
    phi:
      config: configs/phi_halo12_static_v1.yaml
      seed: 42

logging:
  backend: wandb
  project: deep-potential

execution:
  resume: false
```

需要修改模型参数时，优先改所引用的 stage YAML；仅针对本次 run 的小改动放入
`df.overrides` 或 `phi.overrides`。比较多组配对时增加 `trial_01` 等，不要分别维护
互不关联的 DF/Phi 输出路径。

## 4. 后台执行

以下命令可直接用于 fish：

```bash
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_df configs/runs/halo12_server_v1.yaml \
  > logs/halo12-server-v1-df.log 2>&1 < /dev/null &
```

DF 成功后运行 Phi：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_phi configs/runs/halo12_server_v1.yaml \
  > logs/halo12-server-v1-phi.log 2>&1 < /dev/null &
```

Phi 成功后运行评估：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_eval configs/runs/halo12_server_v1.yaml \
  > logs/halo12-server-v1-eval.log 2>&1 < /dev/null &
```

评估包含采样、score、梯度和 Hessian，也可能需要 GPU。查看进度：

```bash
tail -f logs/halo12-server-v1-df.log
```

W&B 打开时也可用对应 run 名监控指标。

## 5. 输出结构

```text
runs/halo12_server_v1/
├── run.yaml
├── trial_00/
│   ├── df/
│   │   ├── config.yaml
│   │   ├── metrics.csv
│   │   └── ckpt/
│   ├── phi/
│   │   ├── config.yaml
│   │   ├── metrics.csv
│   │   └── ckpt/
│   ├── eval/
│   │   ├── df/
│   │   ├── phi/
│   │   └── truth/
│   └── plots/
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
2. 先修改 run/stage YAML，再启动独立进程；
3. 不混用不同 trial 的模型；
4. 不绕过配置快照与 checkpoint 保护；
5. 不提交数据、凭据、日志和 checkpoint；
6. 修改数值实现时同步更新 `dpjax.workflows` 测试；
7. 保留工作区中与本任务无关的用户文件。
