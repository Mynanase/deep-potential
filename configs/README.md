# 配置文件约定

本目录只包含两类 YAML：模型配方和运行配置。

```text
configs/
├── models/
│   ├── df/       # DF 网络、loss、optimizer 与训练超参数
│   └── phi/      # Phi 网络、loss、optimizer 与训练超参数
└── runs/         # 一次实验的数据、预处理、运行、评估与绘图
```

## 模型配方

模型文件必须声明：

```yaml
schema: dpjax.model.v1
kind: df  # 或 phi
```

允许放置网络结构、normalization 数值参数、loss、optimizer、学习率、
batch size、epochs 和正则化。不得放置数据路径、数据筛选、seed、日志、GPU、
checkpoint、绘图或输出目录。加载器会检查这些边界。

参数量不手工配置；程序根据实际初始化后的参数树计算，并写入训练目录中的
`model_summary.yaml`，避免声明值与真实网络不一致。

## 运行配置

运行文件必须声明：

```yaml
schema: dpjax.run.v2
```

它负责引用模型文件，并保存一次具体实验的数据、预处理、seed、运行环境、评估和
绘图设置。真值验证不属于 run schema。一个 YAML 对应一个输出目录，不包含
`trials`。三个正式
入口都只接收一个 run YAML：

```bash
python -m experiments.run_df configs/runs/<run>.yaml
python -m experiments.run_phi configs/runs/<run>.yaml
python -m experiments.run_eval configs/runs/<run>.yaml
```

Plummer oracle 实验可在 `phi.model_overrides` 中显式选择解析 score：

```yaml
phi:
  model_overrides:
    score:
      source: plummer_analytic  # 默认值为 flow
```

该设置同时控制 Phi/CBE 训练和 residual 评估；解析 truth 不进入 `dpjax` 数值核心。
若 oracle 需要与已有 baseline 严格共用 DF support 和 normalizer，可在 `phi` 下设置
`df_run: runs/<baseline>/df`。该依赖会写入不可变的 `run.yaml` 快照。

服务器后台执行使用统一 launcher；它自动写入 run 目录中的日志：

```bash
python -m experiments.launch phi configs/runs/<run>.yaml
```

日志后端也属于 run 配置：`backend: csv` 只保存本地指标，`backend: wandb`
启用 W&B；可用 `mode: online` 或 `mode: offline` 控制同步方式。凭据不写入 YAML。

完整修改路线见 `docs/architecture_operation_guide.md`。
