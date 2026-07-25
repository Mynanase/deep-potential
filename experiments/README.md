# Experiment entry points

`experiments/` 只放参数解析和流程编排；可复用的数据、模型、物理与绘图逻辑
放在 `dpjax/`，因此可以独立进行单元测试。

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
