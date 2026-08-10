# 架构与功能扩展操作指南

本文用于快速判断一个新功能应该修改哪里，并给出配置、运行和验证步骤。

## 1. 先判断修改属于哪一层

| 需求 | 修改位置 | 配置位置 |
|---|---|---|
| 新增 DF 或 Phi 网络 | `dpjax/flows` 或 `dpjax/models` | `configs/models` |
| 修改 CBE、导数或核心数学 | `dpjax/physics` | `configs/models`（如有参数） |
| 更换数据或文件格式 | `experiments/datasets` | `configs/runs` 的 `data` |
| 修改筛选、切分或坐标预处理 | `experiments/datasets` | `configs/runs` 的 `data` |
| 修改训练编排、日志或 checkpoint | `experiments/workflows` | `configs/runs` 的 `execution/logging` |
| 新增数值诊断 | `experiments/diagnostics` 或 `workflows/evaluation` | `configs/runs` 的 `evaluation` |
| 新增图表 | `experiments/plotting` | `configs/runs` 的 `plots` |
| 新增模拟 truth 检查 | `experiments/validation` | `configs/runs` 的 `validation` |
| 修改 Marimo 展示 | `analysis/halo12.py` | 通常读取已有 run 产物 |

判断标准：只有直接处理六维数组和数学模型的可复用算法进入 `dpjax`。文件、
目录、数据集、实验编排、绘图和 truth 都留在 `experiments`。

## 2. 配置文件只有两类

```text
configs/
├── models/
│   ├── df/
│   └── phi/
└── runs/
```

### 2.1 新建模型配方

复制最接近的模型文件：

```bash
cp configs/models/df/halo12_ffjord_compact_v1.yaml \
  configs/models/df/my_df_v1.yaml
```

模型文件中只修改以下内容：

- `flow` 或 `potential`：网络结构和数值求解器；
- `normalizer`：DF 归一化的数值稳定参数；
- `train.optimizer`：目前支持 `adam` 和 `radam`；
- `train.lr`、loss、正则化、batch size 和 epochs。

参数量由层数、宽度和具体实现共同决定，不要在 YAML 中手填。模型初始化后会把
真实 `parameter_count` 写入对应的 `df/model_summary.yaml` 或
`phi/model_summary.yaml`。

不要在模型文件中加入：

- `data`、数据路径或字段名；
- `seed`、`multi_gpu`、日志和 checkpoint；
- evaluation、plots、validation 或 output directory。

模型配置边界由 `experiments.workflows.model_config` 自动校验，字段放错层时会在
启动前报错。

### 2.2 新建一次运行

```bash
cp configs/runs/halo12_static_v1.yaml configs/runs/my_run.yaml
```

至少修改：

```yaml
schema: dpjax.run.v1
name: my_run
output_dir: runs/my_run

data:
  path: data/my_data.h5
  dataset: eta
  split:
    validation_fraction: 0.1
    seed: 100
  preprocessing:
    clip_sigma: 0.0
    jitter_std: 0.0
    transform:
      type: none
      dims: []

trials:
  trial_00:
    df:
      model: configs/models/df/my_df_v1.yaml
      seed: 42
    phi:
      model: configs/models/phi/halo12_static_mlp_v1.yaml
      seed: 42
```

一次参数对比增加新的 trial。需要少量模型消融时可用
`model_overrides`，但稳定且可复用的配方应另存到 `configs/models`：

```yaml
trials:
  trial_01:
    df:
      model: configs/models/df/my_df_v1.yaml
      seed: 43
      model_overrides:
        train:
          optimizer: adam
```

运行层参数集中在：

```yaml
logging:
  backend: wandb
  project: deep-potential

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

评估参数放在 `evaluation`；纯绘图参数放在 `plots`；仅模拟数据存在的真实势能或
加速度检查放在 `validation`。

## 3. 新增一套数据

1. 在 `experiments/datasets` 新增读取或转换函数；
2. 最终产出有限的 `float32 (N, 6)` 数组，列顺序为
   `[x, y, z, vx, vy, vz]`；
3. 如需中间准备步骤，在 `experiments/prepare_*.py` 实现；
4. 将数据路径、字段名、筛选和预处理参数写入 run YAML；
5. 为适配器增加测试，不修改 `dpjax`。

如果新数据已经整理成包含 `eta` 的 HDF5，通常只需修改 run YAML 的 `data`。

## 4. 新增一个诊断或图表

诊断计算和渲染分开：

1. 在 `experiments/diagnostics` 或 `experiments/workflows/evaluation` 计算并保存
   JSON/NPZ；
2. 在 `experiments/plotting` 新增接收数组或产物路径的绘图函数；
3. 在 `experiments/run_eval.py` 编排调用；
4. 将计算参数放进 `evaluation`，将 DPI、格式、切片网格等放进 `plots`；
5. Marimo 只读取保存后的结果，不在页面里启动训练或重计算高成本诊断。

只有完全与文件和绘图无关、可用于任意 `(N, 6)` 数据的统计量，才考虑放进
`dpjax/diagnostics`。

## 5. 执行完整实验

```bash
conda activate dp-jax

python -m experiments.run_df configs/runs/my_run.yaml
python -m experiments.run_phi configs/runs/my_run.yaml
python -m experiments.run_eval configs/runs/my_run.yaml
```

服务器后台运行示例：

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m experiments.run_df configs/runs/my_run.yaml \
  > logs/my-run-df.log 2>&1 < /dev/null &
```

DF 成功后再启动 Phi，Phi 成功后再启动 evaluation。用日志或 W&B 监控，不依赖
SSH 会话持续连接。

## 6. 输出与恢复

```text
runs/<run>/
├── run.yaml
├── logs/
├── summary/
└── <trial>/
    ├── df/
    ├── phi/
    ├── eval/
    ├── plots/
    └── validation/
```

`run.yaml` 保存解析后的完整快照，包括模型来源、数据参数和运行时参数。已有
checkpoint 时默认拒绝覆盖。只有配置完全一致时才能设置
`execution.resume: true`；科学参数变化时应使用新的 `name/output_dir`。
每个模型目录还包含 `model_summary.yaml`，记录由实际参数树计算出的参数量。

## 7. 查看结果与验证代码

```bash
marimo edit analysis/halo12.py
```

提交修改前执行：

```bash
pytest -q
marimo check --strict analysis/halo12.py
ruff check dpjax experiments analysis scripts tests
```

GPU 训练需要在用户终端或服务器环境运行；本地测试只验证 CPU 可执行路径和配置
边界。
