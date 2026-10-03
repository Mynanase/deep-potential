# T1 执行记录：数值链与共享 score 缓存（Agent A 交付）

冻结日期 2026-09-30。任务卡：t0-task-cards.md 卡 T1；上游输入 t0-manifest.json（control＝旧项目 run 2b32eb04 @ 5f76faa，四文件 sha256 全部匹配后才进入计算）。本记录按方案第 6 节交接格式书写；机读缓存登记见 t1-cache-registry.json。

## 任务 / 状态

**完成（数值链通过，缓存可供 B/C 科学使用）**。三层状态：代码＝重写 audit_df_constraints.py＋新增 nf_score_cache.py / run_nf_score_cache.sh / nf_score_stage.txt，测试 30 项（本地 CPU 全套 65 passed 3 skipped）；数值＝pilot 与全量两个 run 独立复现同一审计结果（8/8 门 True），缓存 certified=True；科学＝本卡只认证"导数算得对、链可靠"，不认证"模型学得对"。

## 代码

- base 75d470f → 本卡 commits：6301ecb（审计重写＋缓存＋runner＋测试）、49581fb（runner SIGPIPE 修复）、c056973（manifest 路径解析）、55865e0（split 梯度合并）、cd11fd5（pilot 常量名＋points-only 冒烟）、919bf20（eqx 模型 jit 闭包化＋微型模型全流程冒烟）、8948905（stage→full）。
- 节点 45b0446d 命令已按卡重定向为 bash scripts/auriga/run_nf_score_cache.sh（orx CLI 无按节点改命令入口；以带时间戳备份的 orx.db 单行更新完成，orx exp status 复核通过）。
- 停止条件编码在 runner/审计内：sha256 不符立即退出且不建缓存；非有限值即异常；审计 rc≠0 不进缓存阶段；预算由 timeout 硬保护（pilot 600+1200 s，full 900+6300 s）。
- 过程记录：6 次启动中 4 次为实现性即死（SIGPIPE/路径拼接/shape 广播/jit 边界），逐项修复并各附回归测试；pilot(47a26c96) 与 full(f7f2c973) 为有效 run。

## 输入与产物

- 运行：pilot＝run 47a26c96（commit 919bf20），full＝run f7f2c973（commit 8948905），均在 gpu 单卡（自动选 0 号空闲卡）。全量缓存总耗时 670 s（heldout f32 523 s @774 pts/s＋x64 子集 27.5 s；探针 87.4 s；网格 19.7 s @1972 pts/s），远低于 2 h 上限。
- 产物（服务器 /home/qiutao/.orx/runs/f7f2c973-5a1b-4330-ba72-286526b9e3d2/repo/runs/nf-score-audit/t1-cache/，sha256 全表见 t1-cache-registry.json）：
  - heldout＝clean-smooth 前 int(0.25n)=404,903 行（PID/行号/质量权重/分层标记），f32 全量＋x64-strict 分层子集 4,096；点序 hash 357c26ea…。
  - velocity_probes＝7 壳×16 Sobol 角向×96 共享速度（proposal＝总体速度 std 的对角高斯，σ=[1.396,1.398,0.888]），f32＋x64-strict 双精度＋重要性权重；点序 hash e46d5cb2…。
  - spatial_grid＝19 半径×2048 Sobol 角向，phi/grad_phi/acceleration(=−grad_phi)/laplacian 双精度；点序 hash 8cdb2334…。
  - manifest.json 绑定：四 control 文件 sha256、7 个模型源码文件 sha256、全部点序/产物 hash、ODE 双档容差、精度、"无随机 trace（精确 jacfwd/Hessian trace）"记录、单位块。certified=True（依据同 run 内审计 8/8 门）。

## 发现（数值链，control，全半径分层）

- 同点重算（1024 行分层，x64-strict vs 存储 f32）：空间中位 1.38e-3 / p99 1.35e-2；速度中位 8.9e-4 / p99 2.9e-2（门 1e-2 / 5e-2）。最大尾部 0.14（速度分量单点，已记录）。
- ODE 容差敏感性（strict vs default，x64）：中位 ~1.3e-3（门 5e-3）。精度项（stored vs x64-default）：中位 ~2.2e-3。两者合计约占总链误差的绝大部分——存储值与严格参考的差主要由 f32 算术与默认容差贡献。
- FD：六分量均存在跨 (0.03→1e-4) 的相邻步长稳定区间；配对一致 0.004–0.042（门 0.05）。
- 链式恒等式（ln n(q)+ln P(p|q) 值与梯度，含条件归一化 Jacobian）与 p-block 恒等式：差**精确为 0**（同一实现路径拆分评估）。同输入重复评估：GPU f32 逐位一致（确定性成立）。
- 空间分层：链误差向外递增——内带中位 ~7e-4–1e-3，[4.5,6)/[6,7) 带中位 ~2.7e-3、p99 ~0.03–0.038，仍在 T0 预算内但为外区 2–4 倍梯度，B/C 外区结论引用缓存时应带此注记。
- Phi 侧（网格产品）：grad FD 相对误差 ≤6e-6（h=1e-2）、lap 二阶差分 ≤7e-5；acceleration＝−grad_phi 显式命名与符号测试覆盖。探针壳内 ESS（96 共享速度×16 方向）：149/118/103/95/82/30.7/168——壳 5（51.96 kpc）最低 30.7，C 做测度加权读出时须用缓存内重要性权重。
- 记录性检查：control＝单 flow 对（flow-21 内空间子流与 flow_pos_only-10 逐位一致，12 leaves，max|Δ|=0）；混合机制 combine_log_prob_and_derivatives 在 len(flow_list)==1 时为恒等直通；评估链无随机 trace（Hutchinson 仅存在于未使用的训练期 Jacobian 正则）。

## 限制

- "数值链可靠"不等于模型正确：只证明导数/实现忠实于所实现的 lnF。
- 2026-09-15 旧审计的阈值与 checkpoint 编号未沿用；本卡门为 T0 预算候选（中位<1%/p99<5%、ODE<0.5%、FD 相邻双步稳定）在 pilot 验证后采用。
- heldout＝探索资格（T0 结论 1：无合格确认集）；B 的真实样本 Stein 结果仍标 exploratory。
- x64-strict 参考自身未独立验证到机器精度（FD 交叉验证的是 default-strict 实现的 autodiff 正确性；严格档与默认档一致性由 ODE 敏感性门覆盖）。
- 缓存产物仅存于服务器 run 快照，未异地备份；registry 已把全部 sha256 入库以供校验。

## 下一步

唯一优先：**派发 T2 / T3**。T2（Agent B）读取 t1-cache/arrays_heldout.h5（f32 全量＋x64 子集）做解析 mock 校准，真实样本探索性估计前置条件已满足；T3（Agent C）读取 arrays_velocity_probes.h5 与 arrays_spatial_grid.h5（α 约定已在缓存内显式命名），并按卡完成逐点真势单位标定＋梯度误差定量。触发证据：外区链误差 2–4 倍梯度与壳 5 ESS 偏低均已定量，不再有未知数值阻断。
