# 分支说明(Branch Landscape)

维护日期:2026-09-17。所有实验分支均从 `master@c8bc5f7`(上游 README "code is now mostly JAX")分出。

## 谱系

```
master (c8bc5f7)
  └─ test-code-20260303            最早的 JAX/Flax 移植(单提交)……已被吸收
  └─ refactor-jupyter              notebook→脚本迁移、CHANGELOG 规范 ……已被吸收
       └─ experiment/nsf-velocity-transform   FFJORD v10–v21 训练大修
            └─ experiment/nsf-implementation  + RQ-NSF 样条流实现
                 └─ autopsy/score-audit       + score/CBE 可辨识性诊断
       └─ experiment/baseline-b ≡ cleanup/deprecate-old-scripts
            │                      Auriga 质量加权 DF 主实验线(164 文件)
            └─ codex/standalone-run-architecture
                                   core/workflows/configs 分层重构 + Plummer 偏差实验
  └─ codex/halo12-outer-clump-removal          Halo12 密度涨落 / 外团块线(226 文件)
  └─ codex/upstream-sync-2026-09-12 (当前)     上游同步 + Halo12 适配器 + 审计脚本入库
```

## 各分支说明

### orx 实验线（2026-09，OpenResearch 树上分支）

| 分支 | 状态 | 说明 |
|---|---|---|
| orx/freeze-training-data-clean-smooth-union-default | 冻结（2026-09-19） | 训练数据冻结为 halo12-clean-smooth.h5（union 去除版）；整合验证在其子节点 orx/route-closure-integration-and-verification（幂等复用 + 损失复现）。 |
| orx/w1024-on-clean-smooth-removed-population | 已冻结 | 冻结数据上的首个 w1024 三阶段训练（run fb85670e）。 |
| orx/w1024-phase-2-flow-sampling-potential-training | 已冻结 | clean 线势训练（run 6b6e13aa）；w1024-full 对照线见 orx/w1024-on-full-population-substructure-df。 |
| orx/direction-a-* / debris / velocity-only 系 | 关闭 | 碎片剔除改善有限，路线结项；候选名单与级联图保留为参考。 |
| orx/eval-protocol-bias-* / independent-reshuffle-s11-* | 阻塞待修 | strict 偏置修复与 seed-11 重洗牌；失败原因已定位（selftest 索引 bug / 校验断言反），见结项报告。 |

结项报告：项目 artifact df-phase1-audit/route-closure-20260919.md。

| 分支 | tip(日期) | 状态 | 主要内容 |
|---|---|---|---|
| `master` | — | 上游基线 | 上游 Majakas/deep-potential 主线;`c8bc5f7` 是全部实验分支的共同基底。 |
| `test-code-20260303` | 549700d(2026-03-03) | 已吸收,可删 | 单提交:JAX/Flax 核心、RealNVP + CBE 残差。内容已完整包含在 `refactor-jupyter` 及其后所有分支中。 |
| `refactor-jupyter` | 4e7fb87(2026-05) | 已吸收,可删 | notebook→脚本迁移、CHANGELOG 与日志规范、强制 `flow.type` 并废弃 RealNVP、FFJORD 调参与 2D 边际归一化。工作全部进入 `experiment/nsf-*` 一线。 |
| `experiment/nsf-velocity-transform` | e16c92c(2026-05-24) | 已归档 | FFJORD 训练大修 v10–v18(σ 裁剪、坐标变换、LR 调度)、v19–v21 实验、变换 bug 修复与物理评估、shuffle-val 修复、冷盘/外盘实验。 |
| `experiment/nsf-implementation` | 5e684a9(2026-05-24) | 已归档 | 上一分支 + RQ-NSF(有理二次神经样条流)实现,`flow.type` 分发的第一种样条后端。 |
| `autopsy/score-audit` | 88c3d73(2026-07-12) | 已归档 | 上一分支 + score/CBE 数据可辨识性 5 步诊断,以及带半空间速度截断的解析 2D mock;结论用于判断 score 残差是否数据可辨识。 |
| `experiment/baseline-b` | 0e30b52(2026-08-04) | 保留(主线) | Auriga 质量加权 DF 流水线、FFJORD v21/v22 配置、无调度器服务器脚本、势场 `output_scale` 物理单位 + MSE 训练、真值切片诊断、DF 支撑对齐与密度约束、配置驱动 run 工作流。 |
| `cleanup/deprecate-old-scripts` | 0e30b52(2026-08-04) | 与 `baseline-b` 同 tip,可删 | 清理线,最终与 `experiment/baseline-b` 指向同一提交,留一个名字即可。 |
| `codex/standalone-run-architecture` | 0c7e23f(2026-08-20) | 保留 | `baseline-b` 之上的架构收敛:拆掉旧实验层、core/workflows/configs 分层、脱机实验启动器、Plummer r-cut 中心星 DF 偏差实验、Plummer score oracle 对比、统一结果工件。 |
| `codex/halo12-outer-clump-removal` | 886e915(2026-09-13) | 保留(活跃) | Halo12 密度涨落/外团块线:柱坐标速度诊断、R×θ×φ 联合诊断 PDF 报告、密度切片数据支撑掩码与色阶、均匀质量探针 + 中心挖孔、phi 扫描配置 v2–v7(lam/probe、centerpad/noclip)、密度涨落实验计划 R07–R15、总密度真值对比 r1–r4、通用 shell-truth 库与配置驱动 eval_truth 阶段。 |
| `codex/upstream-sync-2026-09-12` | 见下(当前) | 保留(当前) | 上游 develop 合并(PR #12–#16)、新版 JAX 兼容(`jnp.clip(a_min=)`→`jnp.maximum`)、Auriga Halo12 适配器接入上游 `fit_all`;2026-09-17 追加提交见下节。 |

## 当前分支(codex/upstream-sync-2026-09-12)2026-09-17 整理提交

| 提交 | 内容 |
|---|---|
| `feat(benchmarking)` | 物理单位标度贯通:HDF5 文件级 attrs(`length_scale_kpc`/`velocity_scale_kms`)→ flow/potential 全部基准绘图;修正 2D 残差面板泊松显著性公式;非 gaia 密度图稳健 LogNorm;`nanpercentile`。 |
| `feat(auriga)` #1 | Halo12 封闭质量审计 + DF 约束审计(step 6)脚本及其验证器测试(`tests/`,41 项,解析例子上验证求积/积分/单位换算,不测模型)。绘图脚本只读持久化数组、绝不重载模型。 |
| `feat(auriga)` #2 | Phase-1 DF 审计套件:协议固定与 strict 公共验证集(发现 baseline 与容量组验证集互漏 ~41%)、三 checkpoint 条件速度诊断、指标 + bootstrap + 噪声底(v2)、图版 v2、外 bin 双向确认。运行产物在 `runs/halo12-phase1-df-audit-20260916/`(不入库)。 |
| `chore` | `.gitignore` 增加 `.zcode/` 与 `halo12_package.tar.gz`。 |
| `docs` | 本文件。 |

审计脚本说明:这批脚本目前**不接入** `fit_all` 长期流程,作为独立审计线入库以保存出处与复现路径;`scripts/auriga/` 下绘图与计算严格分层(计算脚本持久化数组/JSON,绘图脚本只读),坐标轴范围采用手调 `set_ylim` 常数并在注释中标注数据范围(2026-09-15 约定)。

## 维护约定

- 分支整理只做提交与文档,不删除分支;"可删"标记需人工确认后自行删除。
- 审计/分析类运行产物一律落在 `runs/`(已忽略),代码入库、产物不入库。
- 原始数据快照(如 `halo12_package.tar.gz`)与会话工件(`.zcode/`)不入库。
