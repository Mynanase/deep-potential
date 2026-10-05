# dpjax-phase2

用 conditional Flow Matching 拟合相空间分布，再通过稳态 CBE 推断引力势。
这是持续研究的工作仓库；主线是 `codex/phase2-baseline-2026-09-22`。
已合入最近的 NF 审计、Plummer oracle 和 T4 基础实现；T4 系列实验在冻结的 orx 分支上，
收线报告归档于 [docs/reports/](reports/t4-plummer-p00-p11-paired-phi/t4-report.md)。

## 从这里开始

[近期进展](docs/progress.md) 记录已完成结果、限制和当前问题。
一次围绕一个问题推进：按 [OpenResearch 工作约定](docs/orx-playbook.md) 管理实验，做相关验证，提交后通过 orx 运行，读日志后留下短记录。
旧计划与执行记录在 [docs/history/](docs/history/)，需要追溯时再读。

## 当前科学设置

- Halo12 control 是 innerA、`lambda_=1`：clean+smooth 质量加权数据、w1024、S1 分层 score 采样、半径平衡负密度先验。[配置](scripts/auriga/options.json) 是参数来源。
- `q=x/10 kpc`、`p=v/100 km/s`；score 对输入 `(q,p)` 求导，势的 Laplacian 对应总引力源密度。
- Halo12 生产 particle truth 是 96³；约 10 kpc 以上**波长**用于定量谱比较，<5 kpc 仅作诊断。这不是对所有径向量的 10 kpc 截断。
- Plummer mock 用完整样本、没有 radial cut。当前 NF 使用项目的 conditional flow，旧 FFJORD 仅为历史参考。

## 常用入口

| 问题 | 代码 / runner |
| --- | --- |
| Halo12 DF → score → Phi | [fit_all.py](scripts/fit_all.py)、[control runner](scripts/auriga/run_w1024_csmooth_s1_gridprior.sh) |
| DF 导数与数值链 | [audit_df_constraints.py](scripts/auriga/audit_df_constraints.py)、[nf_score_cache.py](scripts/auriga/nf_score_cache.py) |
| Stein 弱检验 | [nf_weak_score_tests.py](scripts/auriga/nf_weak_score_tests.py) |
| Plummer 解析 oracle / 局部力 | [plummer_oracle.py](scripts/plummer/plummer_oracle.py)、[local_force_inversion.py](scripts/auriga/local_force_inversion.py) |
| Plummer conditional NF | [t4a_plummer_nf.py](scripts/auriga/t4a_plummer_nf.py)、[T4a runner](scripts/auriga/run_t4a_plummer_mock_nf.sh) |
| P00/P11 Phi 对比 | [t4_plummer_p00_p11.py](scripts/auriga/t4_plummer_p00_p11.py)、[T4 runner](scripts/auriga/run_t4_plummer_p00_p11.sh) |
| 真值与质量评估 | [truth_products.py](scripts/auriga/truth_products.py)、[validate_enclosed_mass.py](scripts/auriga/validate_enclosed_mass.py) |
| 直接绘图 | [plot_potential_2d.py](scripts/plot_potential_2d.py)、[plot_radial_marginals.py](scripts/plot_radial_marginals.py)、[plot_enclosed_mass.py](scripts/plot_enclosed_mass.py) |

## 环境与启动

GPU 使用 `gpu` 的 `/home/qiutao/miniforge3/envs/dp-jax/bin/python`；不自动装包或升级。
本地保留原 T2 工作区已有环境至本仓库 `.venv/`，直接调用 `.venv/bin/python`，无需重建环境。
JAX 显式设 `JAX_PLATFORMS=cpu`（本地）或 `cuda`（服务器），以及 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
数据默认目录是服务器 `/localdisk/kosmos/my-deep-potential/data/auriga/`。

重开 OpenResearch 时从这个主线提交开始，新会话只带当前问题和必要输入。
各研究阶段可以有自己的 runner；同一组对比实验保持一致。orx 自身仍会提供原生提示和 Skills。
旧项目 `686a3ce0-1646-4fe7-9155-9fa5762f4020` 的运行记录、原实验分支和产物保留供追溯；不要恢复旧任务提示作为新实验说明。

代码提交后再运行；已完成结果保留其 source commit。运行数据放 `runs/`，不提交。
本地修改按实际影响验证，无需为普通修复重复完整验收和 hash 清单。
