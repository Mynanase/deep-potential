# T4 Plummer P00 versus P11 paired Phi

## Scope and lineage

- Question: 在完整 Plummer mock 上，把解析 score（P00）换成项目 conditional NF score（P11），下游 Phi 链的误差放大多少？显著性如何？
- Main comparison run: `129a1be2-4310-4463-8e19-c4c3cbfea351`，commit `12b9106`（分支 `orx/t4-plummer-p00-versus-p11-paired-phi-2`）
- Figures + paired stats run: `6b9d2d11-65eb-4528-b296-7ffb477246f7`，commit `1deea74`（子节点 `8486e2dc`，分支 `orx/t4-paired-figures-and-significance-stats`）
- Frozen mock: full Plummer, a=6.8 kpc, N=1,619,615, seed 5, no radial cut; sha256 `7e01175d…e503d`
- P11 flow: 冻结的 T4a checkpoint（`14e2e76a` 资格 run），score 相对误差 median≈3.1%、p99≈20.4%（T4a 记录）
- Constraints: 262,145 条分层配额（封顶到 band 支持量后重分配），per-band 重要性权重；两臂同约束、同权重、同 seed=2、同 MLP（width 1024, depth 3）、各 256 epochs
- Eval grid: 19 radii (2–70 kpc) × 2048 Sobol 方向 = 38,912 点，两臂完全相同 → 逐点配对

## Paired results（run 6b9d2d11）

| band [kpc] | n | Δ force rel. med. (P11−P00) [95% boot CI] | Δ negative-Lap frac. [95% CI] | Wilcoxon p |
|---|---:|---:|---:|---:|
| 2–10 | 6,144 | +0.0398 [+0.0391, +0.0404] | 0.0000 [0, 0] | ~0 |
| 10–30 | 14,336 | +0.0462 [+0.0458, +0.0467] | +0.2243 [+0.2173, +0.2313] | ~0 |
| 30–50 | 8,192 | +0.0688 [+0.0678, +0.0696] | +0.2567 [+0.2427, +0.2704] | ~0 |
| 50–70 | 8,192 | +0.0888 [+0.0876, +0.0902] | −0.0098 [−0.0243, +0.0052] | ~0 |
| all | 38,912 | +0.0572 [+0.0568, +0.0577] | +0.1270 [+0.1217, +0.1322] | ~0 |

Arm-level: P00 force rel. median 0.32%（neg-Lap 15.4%），P11 6.12%（28.1%）；P11 负 Laplacian 绝对均值 ≈ 17× P00。

## Uniform-volume constraint ablation（采样根因消融，run `8e935d0e`）

上方配对比较里 P00（解析 score）外区仍有 20–42% 负密度，提出两种根因：约束采样设计（质量加权 + 外区样本稀疏）或先验/网络容量。子节点 `c4d2ce46`（commit `cd2d805`，分支 `orx/t4-uniform-volume-constraints-ablation`）把约束点改为生成式采样：q 在 [0.1,7] code 体积均匀（r³ 均匀 + 各向同性方向），p 从真 Plummer 条件速度分布采样（与 mock 生成同一原语），262,145 点、seed=23、两臂同点，importance weights 全 1；先验网格、λ、网络、epochs 与配对基线完全一致。质量门：确定性重采 bit 级一致、体积均匀边缘偏差 3.7%、条件速度 KS=0.003、解析 score 下 CBE 恒等式 1.8e-15。

结果（力相对误差中位 / 负 Laplacian 占比，同源球面评估网格）：

| band [kpc] | P00 基线 | P00 均匀 | P11 基线 | P11 均匀 |
|---|---|---|---|---|
| 2–10 | 0.08% / 0 | 0.65% / 0 | 4.06% / 0 | 3.23% / 0 |
| 10–30 | 0.19% / 0.5% | 0.21% / 0 | 4.83% / 22.9% | 4.10% / 20.5% |
| 30–50 | 0.49% / 20.3% | **0.12% / 0** | 7.37% / 45.9% | 7.65% / 48.5% |
| 50–70 | 0.88% / 42.4% | **0.13% / 4.9%** | 9.86% / 41.4% | 10.42% / 45.5% |
| overall | 0.32% / 15.4% | 0.18% / 6.0% | 6.12% / 28.1% | 5.46% / 29.7% |

（基线列：run `6b9d2d11` evaluation arrays；均匀列：run `8e935d0e` manifest。）

结论：

1. **采样根因坐实**。体积均匀 + 无权重把 P00 外区负密度近乎清零（30–50 kpc 20.3%→0%，50–70 kpc 42.4%→4.9%，幅度均值 2.6e-6 可忽略），外区力中位降至 0.12–0.13%（基线 0.49%/0.88%）。内区 2–10 kpc 力中位 0.08%→0.65%——体积均匀下内区点数仅约 0.3%（约 770 点 vs 基线 ~55%），信息量转移的直接后果，在预设 2x 阈值边缘且绝对值小。
2. **score 误差通道被孤立**。同采样下 P11 负密度不降反集中（30–50 kpc 48.5%、50–70 kpc 45.5%），负密度绝对均值比 P11/P00 = 197.6。P00 清零后，P11 的外区负密度与采样设计无关，全部来自 NF score 的系统误差（外区条件速度分布拟合偏差直接传导）：score 误差通道可在 30–70 kpc 制造 ~50% 级负密度。

限制：每臂单次训练（种子方差未覆盖）；P11 在 q>3 code（30 kpc 外）的 score 为弱内插，均匀采样放大其权重；内/外折中权重未扫。

数据：[data/t4_uniform_constraints_stats.json](data/t4_uniform_constraints_stats.json)（设计、两臂分带指标、paired、来源 run/commit）。

## Interpretation and limits

- P00 标定链路噪声底 ~0.3% 中位力误差；P11 的 ~20× 放大与 10–50 kpc 负密度 +22–26 pp 归因于 NF score 误差（唯一被替换的输入）。
- 50–70 kpc 真密度趋零（a=6.8 kpc），符号敏感统计量在两臂同涨（P00 自身 40–45% 负比例）；该波段负密度差不显著、属共享噪声底，判别量看幅度比与 10–50 kpc。
- bootstrap/Wilcoxon 覆盖评估点抽样不确定度；每臂单次训练，训练种子方差未覆盖（效应 ~6 pp vs CI 宽 ~0.1 pp，翻转需种子噪声大两个量级）。
- 确定性检验：两次独立 run 的 force 中位差逐带复现到 0.05% 内；neg-Lap 比例差复现到 ~1–2 pp（符号敏感量）。
- 对 Auriga 的角色：同 machinery（约束构造、`fit_all.train_potential`、评估指标）下量化 score 误差 → Phi 误差通道；Auriga 负密度归因于物理前需先排除该通道。Plummer 为光滑球对称平衡目标，是 NF 的最优情形。
- 更新（采样消融后）：本节所述“P00 噪声底”的外区部分已由子节点 `c4d2ce46` 归因为采样/权重设计并消除（见上节）；在体积均匀采样下，P00 外区负密度归零、噪声底只剩内区 0.65% 力中位一级，P11 的外区负密度则完全来自 score 误差。后续引用本报告的噪声底时应以均匀采样一节为准。

## 后续系列（2026-10-05）：1M 规模、无记忆对照、NF 加权重训否证

接上节采样消融（262k 体积均匀，run `8e935d0e`）。完整机器可读汇总：[data/t4_series_summary.json](data/t4_series_summary.json)。节点链：体积均匀消融（c4d2ce46）→ 1M 规模（86b324ac，run 1460a815，commit e149191）→ NF 加权重训（9e27e5ec，runs 10df043b/a398bc7a/226ed465，e7b934f→67cea34，否定性结论后冻结）→ batch 重采样（bceded5c，run 308c5b75，46aafaf→287ba5f，否定性结论后冻结；首个 run da6a60b0 因未截断权重取消）。

### 全链总览（力误差中位 / 负 Laplacian 占比）

| 设计 | P00 | P11 外区（30-50 / 50-70 kpc）负密度 |
|---|---|---|
| 质量加权 262k（6b9d2d11） | 0.32% / 15.4% | 45.9% / 41.4% |
| 体积均匀 262k（8e935d0e，见上节） | 0.18% / 6.0% | 48.5% / 45.5% |
| 体积均匀 1M（1460a815） | **0.06% / 1.1%（各带全 0）** | 47.4% / 45.6%（纹丝不动） |
| 体积均匀 1M + conditional 流 batch 重采样（308c5b75） | 0.08% / 0.75%（链内同 run） | 48.6% / 48.0%（不变，远外区 force 翻倍） |

1. **偏差型坐实**：约束 ×4 后 P11 外区负密度不变——方差通道排除；同时 P00 全域力中位 0.18%→0.06%、内区（体积均匀下点数占比仅 ~0.3%）0.65%→0.12%（优于质量加权基线 0.32%），回归链在充分约束下噪声底近零，为后续任何 NF 改进提供测量下限。
2. **无记忆对照（本地只读，~17 万点）**：同半径 bin 内 mock 训练样本点与生成点的 NF score 误差相同（1-10: 4.5%/4.4%…60-70: 17.4%/17.7%），误差是半径的单调光滑函数——采样位置无关，排除“生成点落在 NF 未处致差”的替代解释。
3. **损失级加权三次否证**：joint c=1/c=2 把 spatial 边缘变成重尾 w·p(q)，spatial 流训练崩溃（loss ~1e3 vs 控制 ~3）；conditional-only c=1（均匀 batch + loss 权重）不能提升外区样本到达率（每 batch ~20 个 60-70 kpc 样本，w² 梯度噪声主导），s_p 全域劣化（内区 3.6%→84%、外区 12.8%→115%），且加权训练 loss 数值正常掩盖这一切。**reweighting ≠ resampling**；改善外区的杠杆是到达率、容量或架构，不是损失压力。

### batch 重采样否证（2026-10-05，节点 bceded5c，run 308c5b75 / commit 287ba5f）

设计：父节点否证 loss 级加权后，同一权重函数移到采样级——conditional velocity 流每 batch 索引按 p∝w(q)=min((|q|²+1)/1, 50) Gumbel-top-k 抽取（库入口 train_flow_matching_model 新增可选 sample_weights，默认 None 时主线逐位不变），loss 权重全 1，spatial 流与 val 保持原分布，同预算同步数（纯转移版）。每 4096 batch 期望到达：<10 kpc 2316→800、30-50 kpc 186→710、50-70 kpc 54→470。中途发现并修复：mock 无 radial cut（拖尾至 r≈8105 kpc），未截断 w 的加权质量 68.7% 落在 r>70 kpc（49.9% 在 r>300 kpc），首 run（da6a60b0）取消后按设计上界截断到 w=50；同一发现回溯解释 9e27e5ec 的 loss 加权目标实际被拖尾支配（见该节点补充说明）。

结果（vs 1M 基线 1460a815，唯一差异=conditional 流 batch 采样分布；qualify all_ok）：
- score 级：总中位 5.7%（control 3.1%）；s_p 1-10 kpc 7.1%（control 3.6%）、45-60 kpc 25.4%、60-70 kpc 68.3%（外区 control ~12.8%）；s_q 外区持平（60-70 kpc 17.1% vs 17.6%）；60-70 kpc s_p signed_mean −0.25——系统性低估而非噪声（本轮新增 signed-bias 读数的首次应用）。
- Phi 级：P11 force 中位 10-30 kpc 4.45→3.80%、30-50 kpc 7.70→8.51%、50-70 kpc 10.58→18.41%；负密度 30-50 kpc 47.4→48.6%、50-70 kpc 45.6→48.0%（各带 [基线, 重采样]，下同）。

判读：转移斜率为负——中段小赚、远外区大亏，外区负密度纹丝不动；且 30-50 kpc（到达 ×3.8、不邻接拖尾采样质量）同样无改善，到达率假设在干净区域也不成立。|q|²+1 族加权在 loss 级与采样级双双否证后，外区条件速度误差指向容量/参数化（径向特征化、更宽/更长训练、spatial 非 NF 参数化）或进入误差预算；更陡/定向权重与“转移+加预算”未测。

### 对 Auriga 训练的启示

- **第一优先消融**：Halo12 control 的 DF 约束正是质量加权 + S1 分层配额（与 T4 基线同构，外区配额同样受支持量封顶）。T4 证据预测 Auriga 外区负密度有相当部分是采样设计伪影：把约束改为体积均匀/半径平衡 + 无质量权重（同预算）重跑一次 Phi，预期外区负密度大幅下降。这是便宜且高信息量的实验。规模不变性检验（1M）同样可迁移，用于区分方差型/偏差型。
- **NF score 误差需径向分解诊断**：Auriga 的 NF 同样在质量分布上训练，外区欠约束是同构问题；loss 加权与 batch 重采样在 Plummer 上均已否证，剩余方向是容量/径向特征化、spatial 边缘改用非 NF 参数化（spatial 流对重尾目标的脆弱性是独立证据），或把外区 score 偏差作为已知系统误差进入 Auriga 归因误差预算。Plummer 的 signed-bias 读数（60-70 kpc s_p 系统低估 −25%）表明该误差有方向，进预算时可按径向偏置建模。sample_weights 库入口保留，可用于 Halo12 的训练采样消融。
- **负密度判读要带幅度**：真密度趋零的波段符号统计脆弱（Plummer 50-70 kpc 两臂同涨），归因结论要看负密度绝对幅度比与真密度水平，不能只看占比。
- **链路标定方法论可迁移**：P00 解析臂把链路噪声底与 score 误差分离的设计，Auriga 可用替代 score 源（如 Stein 校准/held-out 交换）构造类似标定。

## Figures and data

- [P00 radial profiles](figures/t4_p00_radial.png)（[pdf](figures/t4_p00_radial.pdf)）
- [P11 radial profiles](figures/t4_p11_radial.png)（[pdf](figures/t4_p11_radial.pdf)）
- [Paired comparison](figures/t4_paired_comparison.png)（[pdf](figures/t4_paired_comparison.pdf)）
- 绘图/统计脚本（随图保存）：`figures/t4_plot_p00_p11.py`
- 完整配对统计：`data/t4_paired_stats.json`（含每带 bootstrap CI 与 Wilcoxon）
- 采样消融统计：[data/t4_uniform_constraints_stats.json](data/t4_uniform_constraints_stats.json)
- 全链机器可读汇总（含 1M 与加权重训否证）：[data/t4_series_summary.json](data/t4_series_summary.json)

### 2D 切片与训练前 score 形状（同源 run 6b9d2d11）

- [Score shapes](figures/t4_score_shapes.png)（[pdf](figures/t4_score_shapes.pdf)）：Phi 训练前，262,145 约束点上 NF vs 解析 score 的 |s_q|、|s_p| 径向剖面（16–84 分位带）及逐点相对差。形状趋势一致；相对差内区（<10 kpc）~4–5%，30–70 kpc 升至 ~10–12%，|s_p| 幅值随半径增大而 NF 外区偏差同步放大。
- [Force error slices](figures/t4_slice_force_rel_P00.png) / [P11](figures/t4_slice_force_rel_P11.png)：x–z 平面（y=0）力相对误差 2D map（1%/10% 等值线）；P11 的 1% 线收缩到内区，外区大片超 10%。
- [Density slices](figures/t4_slice_density.png)：上行 (ρ−ρ_t)/ρ_t（对称 symlog）；下行 max(0,−ρ)/ρ_t 负密度占比（log）。P00 负密度集中在 R≳30 kpc，P11 从 10–30 kpc 即出现；负密度呈连贯的角向扇区（环带方位角符号翻转中位数 ~2），是低频角向系统性偏差，非高频斑驳、也非全周壳层。
- [NF score_q slice](figures/t4_slice_score_q.png)：p=0 平面上 NF vs 解析 score_q 相对差（2D）；median ~20%，高于速度平均的 ~5%——NF 误差存在速度维度各向异性，p=0（速度分布峰）处 q-score 偏差更大。
- 切片绘图脚本（随图保存）：`figures/plot_t4_slices.py`；输入为 run 6b9d2d11 的 Phi checkpoint、冻结 NF checkpoint 与 paired_constraints.h5（本地 sha256 与该 run manifest 逐一核对）。

Reproduce: 在 run `6b9d2d11` 的 run 工作区 `runs/nf-score-audit/t4-p00-p11/` 上执行
`python scripts/auriga/t4_plot_p00_p11.py --run-dir runs/nf-score-audit/t4-p00-p11`；
切片图：`python scripts/auriga/plot_t4_slices.py --run-dir runs/nf-score-audit/t4-p00-p11`
（需 P00/P11/nf checkpoint 与 paired_constraints.h5，随该 run 保留）。
日志 `orx logs 6b9d2d11-65eb-4528-b296-7ffb477246f7`。
