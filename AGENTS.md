我的配置：
- 终端使用 fish

虚拟环境使用
```shell
conda activate dp-jax
```

## GPU 环境

- 8x NVIDIA A100-PCIE-40GB (GPU 0-7)
- 驱动 570.158.01 / CUDA 12.8
- JAX 0.9.0 可正常使用 GPU（`jax.default_backend()` 返回 `gpu`）
- 注意：Trae 沙盒环境无法访问 GPU，GPU 相关任务需在用户终端运行

## Changelog 提醒规则

仅在以下情况下，在回复末尾加一句提醒：
- 执行了 `git push`
- TODO list 中所有任务变为 completed

提醒内容（固定格式，不展开）：
> 📝 需要更新 CHANGELOG 吗？运行 `/log` 自动生成。

**不触发**的场景：debug、问答、调参、小修改、无 git 操作的对话。
