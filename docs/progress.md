# 近期研究状态（2026-10-05）

当前问题：T4 Plummer 线已收线。外区负密度完成三层根因分解；NF 外区精度的两类加权修复（loss 级、采样级）均已否证；
下一步转向容量/参数化方向或把外区 score 偏差进误差预算，并把采样消融迁移到 Auriga。
完整报告与图归档在 [docs/reports/t4-plummer-p00-p11-paired-phi/t4-report.md](reports/t4-plummer-p00-p11-paired-phi/t4-report.md)，
源 artifacts 目录为 `t4-plummer-p00-p11-paired-phi/`（含随图保存的绘图脚本；repo 归档只收报告、数据与图）。

## 已收线的结论（全部有 run 证据，报告内有 run/commit 对照）

1. **NF score 误差通道量化**：同约束、同回归链下把解析 score 换成 conditional NF score，
   全域力误差中位 0.32% → 6.12%（逐点配对 Δ 中位 +5.7pp），10–50 kpc 负密度占比 +22–26pp，Wilcoxon ~0（run `6b9d2d11`）。
   50–70 kpc 真密度趋零，符号统计两臂同涨，判读以幅度比与 10–50 kpc 为准。
2. **采样设计伪影坐实**：质量加权 + S1 分层约束在完美 score（P00）下仍产生 20–42% 外区负密度；
   改为体积均匀生成 + 无权重后外区清零（30–50 kpc 20.3%→0%，50–70 kpc 42.4%→4.9%），
   外区力中位降至 0.12–0.13%（run `8e935d0e`）；1M 点下链路噪声底 0.06% 力中位、负密度 1.1%（run `1460a815`）。
3. **NF 外区误差是偏差型**：约束 ×4 后 P11 外区负密度不变（47.4%/45.6%）；同半径 bin 内训练点与生成点
   score 误差相同（4.5%→17.6% 随半径单调），误差是半径的光滑函数、无记忆，排除“生成点落在未处”解释。
4. **损失级加权否证**（节点 `9e27e5ec` 按否定性结论冻结）：joint 加权把 spatial 目标变成重尾 w·p(q)，训练崩溃；
   conditional-only 加权不提升外区样本到达率，w² 梯度噪声主导，s_p 全域劣化（内区 3.6%→84%、外区 12.8%→115%）。**reweighting ≠ resampling**。
5. **采样级重采样否证**（节点 `bceded5c`，run `308c5b75`）：conditional 流每 batch 按 p∝w(q)=min(|q|²+1, 50) Gumbel-top-k 抽样
   （纯转移、同预算；库入口 train_flow_matching_model 加可选 sample_weights，默认 None 主线不变），外区负密度不动（47.4→48.6%、45.6→48.0%），
   远外区 force 误差翻倍（10.6→18.4%），仅 10-30 kpc 小赚；30-50 kpc（到达 ×3.8、无拖尾邻接）同样无改善——外区条件速度误差不是到达率问题，
   且带系统方向（60-70 kpc s_p 低估 ~25%，本轮新增 signed-bias 读数）。
6. **拖尾发现**（重采样节点首次启动时）：mock 无 radial cut（拖尾至 r≈8105 kpc），未截断 |q|²+1 的加权质量 68.7% 落在 r>70 kpc——
   9e27e5ec 的 loss 加权目标实际被拖尾支配，其否证范围应表述为“未截断的 |q|²+1 权重不可行”（该节点已补记）。

## 下一步

- **NF 外区精度**：|q|²+1 族加权在 loss 级与采样级双否证后，剩余方向是容量/径向参数化（径向特征化、更宽/更长训练）、
  spatial 边缘非 NF 参数化，或接受外区 score 偏差进误差预算（signed-bias 读数表明误差有方向，可按径向偏置建模）；
  更陡/定向权重与“转移+加预算”未测。`sample_weights` 库入口保留，可直接用于后续训练采样实验。
- **Auriga 迁移**：第一优先做 Halo12 control 采样消融（约束改体积均匀/半径平衡 + 去质量权重，同预算重跑 Phi）——
  这是 T4 已坐实的采样设计伪影通道，与 NF 训练加权无关；NF 外区 score 偏差按上一条处理。

## 已有证据

| 工作 | 最终 run / source commit | 可以保留的结论与限制 |
| --- | --- | --- |
| T1 数值链与固定点缓存 | `f7f2c973` / `8948905` | 8 项数值检查通过，缓存合格；只说明导数计算可靠，不认证 NF 的物理正确性。 |
| T2 Stein 检验 | `6778ac8c` / `c6d3457` | 校准通过；Auriga 读出仍是探索性。cluster/block maxT p=0.2235/0.1645，不能单独认定 score 偏差。 |
| T3 Plummer oracle / 局部力 | `fade7ea7` / `00f7049` | 完整 mock 与解析 machinery 合格。Auriga 真势局部梯度相对误差中位数约 25%，不能用作可靠真力标签。 |
| T4a Plummer conditional NF | `14e2e76a` / `1d5b18f` | 数值/缓存检查通过，score 相对误差 median≈3.1%、p99≈20.4%；不是下游 Phi 结论。 |
| T4 P00/P11 配对 | `6b9d2d11` / `12b9106` | 结论 1；每臂单次训练，种子方差未覆盖（效应量 vs CI 宽度余量两个量级）。 |
| T4 体积均匀采样消融 | `8e935d0e` / `cd2d805` | 结论 2；内区 2–10 kpc 力中位升至 0.65%（体积均匀下点数仅 ~0.3%，信息量转移）。 |
| T4 均匀约束 1M | `1460a815` / `e149191` | 结论 2 规模项 + 结论 3 方差通道排除。 |
| T4a 半径加权重训 | `10df043b` `a398bc7a` `226ed465` / `67cea34` | 结论 4+6，节点已冻结；未截断权重不可行，截断版由子节点检验。 |
| T4a batch 重采样 | `308c5b75` / `287ba5f` | 结论 5+6，节点 `bceded5c` 已冻结；首个 run `da6a60b0` 因未截断权重取消。 |

T4a 训练 checkpoint 来自 `dac804e7`；资格 run 只复用已完成训练修复下游计算，不当作重新训练。

## Halo12 对照与前一轮

- Phase-2 control 为 innerA、lambda=1，96³ production truth；lambda=10 是历史对照。
- truth 分辨率研究没有建立全尺度收敛；约 10 kpc 以上波长用于定量谱比较，<5 kpc 仅诊断。
- round-1 的 osc-pair、spectral-norm ceiling、lambda anneal 没有改善定量域振荡指标；不作为主线默认设置。
- 相关报告保留在旧项目 artifacts：`particle-truth-spectrum-resolution/`、`osc-suppression-round1/`、`nf-score-audit-t2/`、`plummer-oracle-t3/`。

## 代码与证据的位置

T4 系列代码在实验分支 `orx/t4-plummer-p00-versus-p11-paired-phi-2` → `orx/t4-paired-figures-and-significance-stats`
→ `orx/t4-uniform-volume-constraints-ablation` → `orx/t4-uniform-constraints-at-1m-points`
→ `orx/t4a-radius-reweighted-nf-retrain-with-score-corr` → `orx/t4a-conditional-flow-gumbel-top-k-radius-resampl`
（节点链 bbe6566d → 8486e2dc → c4d2ce46 → 86b324ac → 9e27e5ec → bceded5c，前四个已收线，末两个按否定性结论冻结）。
`scripts/auriga/plot_t4_slices.py` 按约定留在 worktree，未入主线；切片图随 artifacts 报告保留。
运行日志用 `orx logs <run-id>` 查询；本地 CPU run、服务器 checkpoint、项目 artifacts 保持原路径。
重开会话时带上当前问题、相关代码和所需输入即可；完成一次判断后更新本页。
