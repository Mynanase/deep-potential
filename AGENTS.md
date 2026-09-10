# Repository instructions

## Environment

- 用户终端使用 fish；命令示例应保持可直接复制，避免依赖 zsh-only 语法。
- 项目虚拟环境为 `dp-jax`：`conda activate dp-jax`。
- 精确机器、GPU、驱动和 CUDA 信息属于本地状态，记录在被忽略的
  `.agent_local/notes/runtime-environments.md`，不要写入共享文档。
- 沙盒不保证 GPU 可用。GPU 训练和真实多卡验证应提供用户终端命令，不能把
  沙盒 CPU 结果描述为 GPU 验证。

## Architecture contracts

- `dpjax` 是只接受内存数组的数值核心；不得导入 `experiments`、HDF5、YAML、
  Matplotlib、Orbax、W&B 或仓库路径工具。
- `experiments` 负责数据文件、运行配置、训练、checkpoint、评估、绘图和 artifacts。
- 保持 `dpjax.run.v2`、模型配置、物理目标、单位、数组顺序和 artifact schema，除非
  用户明确要求修改科学契约。
- 正式结果写入 `runs/<experiment>/results/{data,figures,debug}`；可复用诊断进入
  `experiments/diagnostics` 或 `experiments/plotting`，一次性真值检查留在 `analysis`。

## Workflow contracts

- `run_df`/`run_phi` 只训练，`eval_df`/`eval_phi` 只生成派生数值 artifacts，
  `plot_df`/`plot_phi` 只读取已保存 artifacts 绘图。
- `experiments.run df|phi|all` 只做薄编排，并以独立 Python 进程顺序执行昂贵阶段；
  不要把 DF、Phi 训练和绘图塞入同一个 JAX 进程。
- 训练循环只保存强伴随健康指标；完整采样、score、CBE、Hessian/Laplacian 诊断在
  训练后 eval 中执行。绘图函数接收内存数组并返回 Figure，不读取模型或启动训练。
- 任一组合阶段失败后停止后续阶段，保留已生成科研产物；不要自动跳过、重试或删除。

## Change and validation rules

- 先检查分支、tracked diff 和未跟踪文件；用户的本地研究材料不得移动、删除或纳入提交。
- 优先扩展现有配置、workflow、registry 和 artifact loader，不为每次 debug 新增脚本。
- 科研图改动除测试外还要实际渲染并目视检查；GPU 相关结果明确区分已验证与待验证。
- 常规验证使用 `ruff check dpjax experiments analysis tests`、CPU pytest、
  `compileall`、CLI help、`uv lock --check` 和 `git diff --check`。
