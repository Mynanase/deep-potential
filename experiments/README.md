# Experiment entry points

`experiments/` 只放参数解析和流程编排；可复用的数据、模型、物理与绘图逻辑
放在 `dpjax/`，因此可以独立进行单元测试。

## 推荐：run-level 配置

完整的 `DF -> Phi -> evaluation` 流程优先使用一个运行配置，例如
`configs/runs/halo12_static_v1.yaml`。模型超参数仍保留在原有 DF/Phi YAML 中，
运行配置只负责把数据、trial、输出路径、日志和评估选项连接起来：

```bash
python -m experiments.run_df configs/runs/halo12_static_v1.yaml
python -m experiments.run_phi configs/runs/halo12_static_v1.yaml
python -m experiments.run_eval configs/runs/halo12_static_v1.yaml
```

输出采用固定结构：

```text
runs/<run-name>/
├── run.yaml
├── logs/
├── trial_00/
│   ├── df/
│   ├── phi/
│   ├── eval/
│   └── plots/
└── summary/
```

`run.yaml` 是首次启动时写入的完整训练配置快照。同一输出目录的后续阶段会
校验该快照；若模型配置已经变化，必须使用新的 `name/output_dir`。已有 checkpoint
默认禁止覆盖；需要续训时在运行配置中设置 `execution.resume: true`。

远程运行时只需把这一条短命令放入后台：

```bash
nohup python -m experiments.run_df configs/runs/halo12_static_v1.yaml \
  > logs/halo12-df.log 2>&1 &
```

原有多参数入口继续保留，用于兼容旧作业和临时调用。

从仓库根目录使用模块方式运行：

```bash
python -m experiments.gendata_plummer --help
python -m experiments.train_df --help
python -m experiments.eval_df --help
python -m experiments.train_phi --help
```

数据文件的快速检查入口：

```bash
python -m experiments.inspect_data --data data/halo_12_train.h5
```
