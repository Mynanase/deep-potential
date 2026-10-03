# T0 执行记录：来源盘点与执行契约（NF score audit 线）

冻结日期 2026-09-30。方案：`docs/nf-score-audit-plan.md`（v1，commit e0edb21，节点 45b0446d）。
机读 manifest：`t0-manifest.json`；任务卡：`t0-task-cards.md`（均在本目录）。
本轮为只读盘点：未训练、未启动任何服务器诊断、未改动任何冻结分支与用户脏文件（各 worktree 状态已核对）。
**更正（2026-09-30 同日，经用户指正）**：结论 2 初版把『truth 产品现状』误写成『数据上限』——逐点真势存在于源数据（见 manifest `true_potential_assets`），结论 2 与 T3 卡已按核验事实改写。

## 任务 / 状态

**完成**。三层状态：代码＝仅新增本文档三件套，无代码改动；数值＝无新计算，全部结论来自只读核验（run 日志、服务器文件、HDF5 attrs、sha256）；科学＝T0 契约本身，不包含任何 score/力/密度的科学判断。

## 被检验对象（control，唯一）

innerA λ=1：旧项目 dpjax-phase1（07d8ee01，已归档）训练 run `2b32eb04-d8bf-4c03-a3a9-98d0f0db1192`（节点 9788e6de），源码 `orx/inner-band-fix-a-radius-balanced-prior-grid` @ `5f76faa`（run 记录 commit，非当前 HEAD）。服务器工件目录与部署件 sha256 见 manifest：空间流 `flow_pos_only-10` + 条件速度流 `flow-21` + Φ `potential-10`，**单 flow 对，无 ensemble**。锚点：dM 带 1.25/5.95/8.76/8.14%（run 313e0cc7）；f_hi<10/<5 = 10.7%/2.8% vs truth 8.8%/2.3%（run e4d4db5d）；残差谱峰 69.5 kpc、模型残差方差 ~22× truth（run e66ac2dd）。

## 三个明确结论

1. **合格确认集：无。** clean-smooth 前 25% 行（404,903 粒子）未进梯度更新、未用于 checkpoint 选择（固定 256 epoch 末点部署），但 w1024/FlowMatching 架构、S1 配额、innerA 设计经 phase-1/2 在重叠总体上的比较选定，清洗规则（union registry）也在全总体上设计；s11 文件是同成员重洗牌，去除团块与 all-mass 是不同总体。本轮 Auriga Stein 检验只能作探索性；独立确认需要新设计（如其他 snapshot），不能用现有任何划分回溯恢复。
2. **逐点真势存在；逐点真加速度无；现成 3D 场产品无。** clean-smooth 的直接源文件 `halo12_all_mass_clean_outer_clump_smooth.h5` 带逐粒子 `potential`（n=1,619,615，按 `particle_id` join；`prepare_data.py` 生成训练文件时丢弃该列）；raw 快照 `snapdir_127`（8 文件）PartType0/1/4 均有 `Potential` 块、**无 `Acceleration` 块**；96³ truth 产品（sha256 e02379fa…）只用 positions+masses 构建，势侧只有密度网格＋球单极 phi_sphere_mean(24)＋方向散布 phi_dirs(24,2048)＋切片。含义：非球真势信息在真实粒子位置上可得（全物质贡献、模拟软化），但 α*(q) 读出需局部梯度估计且误差必须先传播（外区稀疏恰是最难区）；单位未标定（adapter 自标 `acceleration_unit=unknown`）。附录 A5 由『不可做』升级为**条件可行**（先单位标定＋梯度误差定量）；Auriga 力侧主结论仍是可辨识性/敏感性/与 Φ 的一致性，网格真值口径不变（不小于 10 kpc 定量、小于 5 kpc 诊断）。
3. **兼容 mock NF：无。** 全树无任何 Plummer mock 上训练的 NF；`mock_dust.py` 是无关的 HEALPix 尘埃 mock。归档分支 `codex/standalone-run-architecture`（0c7e23f）有历史 Plummer oracle 代码可参考。T4 需新建 mock DF 训练节点并单独申请预算。

## 契约冻结摘要（全文见 manifest）

- 坐标 q=x/L（L=10 kpc）、p=v/V（V=100 km/s）；score 对输入 (q,p) 求导；**α=−∇qφ**（旧 `local_force_svd` 的 g=∇φ 必须显式转换）。
- 分层：q 边界 [0.1,0.2,1,2,3,4.5,6,7]（= S1 radial_alloc 边界，已核一致）；报告带 2–10/10–30/30–50/50–70 kpc 由成员分层直接聚合（30–50 = [3,4.5)+[4.5,5)）。
- 点集候选：heldout＝clean-smooth 前 404,903 行（PID 保留，探索资格）；velocity_probes＝7 壳×16 角向×96 共享速度（proposal/权重 T1 卡冻结）；spatial_grid＝Sobol 角向固定半径（2:3:30 ∪ 30:5:70 kpc，2048 向量/半径）+ 谱用独立 3D 网格，与训练 q_grid 无关。
- 种子：既有 0/0/1/2 不动；新增诊断层候选 probes=3、bootstrap=4、mock=5。
- Mock（T3/T4）：参考实现＝commit `0c7e23f` 的 Plummer 体系（`experiments/workflows/score_sources.py` 已内置 flow/plummer_analytic 双源切换，即 P00/P11 的入口模式；含 oracle 配置与解析测试；**r-cut 配置不移植**）。参数已按检测区密度匹配冻结：a=6.8 kpc、N_mock=1,619,615，对照 control run 实测的分带质量份额（30–45/45–60/60–70 kpc = 0.032/0.0175/0.0058，对应数密度 0.193/0.054/0.018 /kpc³），解析残差 +22%/−18%/−14%，容差每带 ±25%；mock 只作检测区类比，内区失配（1–2 kpc −86%）已声明。**不使用 r-cut，完整 Plummer 数据**：oracle score 恒为完整解析 score；Φ 阶段保留生产域约定（1–70 kpc）属势训练契约而非数据切割，T4 卡显式声明约束池边界。
- 容差：科学容差 **TBD**（参考尺度＝control 锚点，不以显著性倒推）；数值预算候选约为目标效应 10%（floored-rel 中位<1%/p99<5%、FD 相邻双步长稳定、ODE 默认-严格差<0.5%），T1 验证可实现后方采用。
- 开销（本轮）：无训练；服务器动作一律走编排＋已提交 runner；T1 吞吐试点候选不超过 2048 点/单卡/30 min，全量缓存候选不超过 2 h/单卡，均待派发时授权；T2/T3 解析部分仅本地 CPU。
- 持久化：`runs/nf-score-audit/<stage>/`（manifest/points/arrays/metrics/review/figures）；**每个缓存必须绑 hash**——已确认 control 的 df_gradients.h5 attrs 为空，正是方案 4.2 针对的真实缺口。

## 发现与限制

- **control 模型从未做过数值 score 链审计**：audit_df_constraints 只审过 2026-09-15 旧 baseline；T1 是全新证据，不是复算。
- （更正记录）初版结论 2 的『无非球 3D 势场』混淆了产品现状与数据上限：核验后确认逐点真势在 clean-smooth 源文件与 raw 快照中都存在（manifest `true_potential_assets`）。遗漏原因：初版只核验了 truth 产品与训练文件，未下到源数据逐字段检查。
- Mock 事实：服务器 legacy `plummer_n*.h5` 只有 eta、无任何 attrs（如 n524288_rcut1.0 文件实存 336,299 行），不可作谱系输入，须按新契约重新生成；参考分支的 branch ref 在归档 dpjax-phase1 仓库，commit `0c7e23f` 在本仓库可直接读取。
- S1 样本行序不可交换（首/中部 r_q 中位约 0.66、尾部约 6.43），Φ 的 val（前 65,536 个样本）是空间偏倚子集，phi_val=0.3640 只作监控量解释。
- S1 实际 n=262,145（options 写 262,144；逐 bin max(1,round) 取整，S1 WEIGHTS CHECK PASS）；审计以实际数组为准。
- `plummer_sphere.draw_from_sphere` 方位角用全局 np.random（种子分层违规），T3 前必须修复；1D CDF 采样为 1024 点梯形＋插值，离散精度待校准。
- 限制：真值侧谱收敛正式判据未过（不小于 10 kpc 定量、小于 5 kpc 诊断）；run 快照工件在服务器 `.orx/runs/` 下未做异地备份，T1 起的重要缓存建议登记 sha256 于本仓库 manifest。

## 下一步

唯一优先：派发 **T1（Agent A）**——control score 数值链审计与共享缓存（先吞吐试点，后申请全量）。解析侧 T2/T3 mock 校准可并行开发（本地 CPU）；T3 卡新增『逐点真势单位标定＋主轴系梯度估计误差定量』作为 A5 前置小项。触发证据：control 外区异常锚点（69.5 kpc 谱峰、约 22 倍残差方差）存在，但其 score 链从未被审计。成本候选见上，等待逐段授权。
