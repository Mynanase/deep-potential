# dpjax-phase2

Auriga Halo12 的引力势推断，基于 JAX / Equinox / Flow Matching。
先拟合恒星相空间分布，再计算 DF score，最后用完整稳态 CBE 拟合神经势场。
这是一个 OpenResearch 项目；实验状态、父子关系和运行记录以 orx 为准。

## 当前入口

- 项目 ID：`686a3ce0-1646-4fe7-9155-9fa5762f4020`。
- 已登记的生产命令：`bash scripts/auriga/run_w1024_csmooth_s1_gridprior.sh`。
- 基线配置：[scripts/auriga/options.json](scripts/auriga/options.json)。
- 数据入口：[scripts/auriga/orx_data_path.txt](scripts/auriga/orx_data_path.txt)。

按任务需要查看 orx，不必每次遍历完整实验树或全部历史：

```sh
orx skill
orx project view 686a3ce0-1646-4fe7-9155-9fa5762f4020
```

运行命令由 orx 管理。代码和配置是实验变更的载体；生产 runner 已包含环境、
数据检查和三阶段调用，不需要从旧文档拼装另一套启动命令。

## 基线事实

| 项目 | 当前配置 |
| --- | --- |
| 训练输入 | `halo12-clean-smooth.h5`，质量加权，数据 shuffle seed 0 |
| 服务器数据目录 | `/localdisk/kosmos/my-deep-potential/data/auriga/` |
| 服务器 Python | `/home/qiutao/miniforge3/envs/dp-jax/bin/python` |
| DF | 两个 flow，MLP width 1024 / depth 3，seed 0 |
| Score 采样 | S1 径向分层及 importance weights，seed 1 |
| Phi | MLP width 1024 / depth 3，seed 2，完整 DF CBE |
| 负密度先验 | 独立半径平衡网格，`lambda_=10` |

参数的完整定义在 `options.json`，数据单位和范围在 HDF5 attrs。
这张表描述对照配置，后续实验的实际配置以其提交和运行产物为准。

## 源码导航

| 任务 | 入口 |
| --- | --- |
| 转换物理单位数据 | [prepare_data.py](scripts/auriga/prepare_data.py) |
| DF 训练 → score 采样 → Phi 训练 | [fit_all.py](scripts/fit_all.py) |
| 采样与 CBE 损失 | [flow_sampling.py](scripts/flow_sampling.py)、[potential.py](scripts/potential.py) |
| 真值预处理 | [truth_products.py](scripts/auriga/truth_products.py) |
| 质量验收与 DF 约束诊断 | [validate_enclosed_mass.py](scripts/auriga/validate_enclosed_mass.py)、[audit_df_constraints.py](scripts/auriga/audit_df_constraints.py) |
| 通用绘图 | [plot_potential_2d.py](scripts/plot_potential_2d.py)、[plot_radial_marginals.py](scripts/plot_radial_marginals.py)、[plot_enclosed_mass.py](scripts/plot_enclosed_mass.py) |
| CPU 解析测试 | [tests/](tests/) |

数值修改的本地检查：`env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false <python> -m pytest tests/ -q`。
`<python>` 使用已有且依赖完整的环境；文档、格式等修改只需相应检查。

## 科学含义

`q=x/L`、`p=v/V`，默认 `L=10 kpc`、`V=100 km/s`；`Phi_phys=V²φ`。
保存的 `dlnf_deta` 对输入 `(q,p)` 求导。DF 使用 75 kpc 内恒星，
`eta.attrs` 的 1–70 kpc 范围限定 Phi 训练区域。
质量加权与粒子数加权对应不同目标；势的 Laplacian 对应总引力源密度。
`lambda_` 抑制负密度，不保证密度平滑。模拟真值用于独立验收。

## 历史与产物

[docs/history/](docs/history/) 保存 Phase-1 决策、分支沿革、迁移记录和汇报材料；
仅在追溯相关结论时阅读。旧路径和旧命令反映当时状态。
[scripts/auriga/archive/](scripts/auriga/archive/) 保存旧脚本；当前入口见上表。
代码版本用 Git/orx 追溯，运行产物在 `runs/` 或 orx run 目录。
