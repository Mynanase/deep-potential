# Halo12 体积均匀 DF 约束消融

## 问题与谱系

- 问题：Halo12 Phi 推断链其余环节全部不动，只把势回归的约束点采样从
  "flow 生成 + S1 分层配额 + 逐带重要性权重"换成 "q 体积均匀生成 + conditional p|q + 无权重"，
  同预算重跑一次 Phi，外区负密度与力（dM）误差如何变化？
- 来源预测：T4 Plummer 系列的采样消融（节点 c4d2ce46，run `8e935d0e`）——完美 score（P00）下
  体积均匀+无权重把外区负密度 20–42% 清零、外区力中位降至 0.12–0.13%。
- 本节点：`fe48bffb`（分支 `orx/halo12-uniform-volume-df-constraint-ablation-2`），
  run `49af3bb9-e1c0-42d4-a509-a447a9a6f8cc`，commit `382e84d`；
  误建节点 `29c2746c`（继承错 run command）与 palette 崩溃的 run `3f7e33b4` 见节点 desc 执行记录。
- 对照：冻结 control run `2b32eb04`（innerA、lambda=1、半径平衡网格负密度先验）在本 run 内
  同码同网格重新评估，dM 锚点 2-10/10-30/30-50/50-70 = 1.25/5.95/8.76/8.14% **逐位复现**。

## 固定项与唯一变量

固定：DF/NF score 源（复用 `2b32eb04` 训练好的 conditional flow，loss 锚点 val_pos/val_vel 复现
rel ~1e-5）、势网络 w1024d3、预算（262144 约束、256 epochs、batch 1024、同 LR 表）、
半径平衡网格负密度先验（prior_grid 4096、q_max 7、lambda=1）、评估代码与网格。
变量（唯一）：约束点采样——control 为 flow 生成 + S1 radial_alloc 配额 + importance weights；
uniform 臂为 q 体积均匀（[0.1,7] code，r³-CDF + 各向同性）+ 同 flow 的 conditional p|q + 单位权重。
约束径向分布（S1 边，code units [0.1,0.2,1,2,3,4.5,6,7]）：
uniform `[1, 752, 5323, 14426, 48773, 95428, 97441]`（92% 质量在 r>30 kpc）
vs S1 配额 `[41943, 104858, 52429, 18350, 13107, 20972, 10486]`。

## 结果

### dM（Gauss 通量 enclosed mass vs 粒子真值单极，带中位 |rel err|）

| 带 [kpc] | innerA control | uniform |
|---|---:|---:|
| 2–10 | 1.25% | **37.44%** |
| 10–30 | 5.95% | 4.32% |
| 30–50 | 8.76% | **13.41%** |
| 50–70 | 8.14% | **20.79%** |

signed dM（r≥30 kpc 节点）：control −8.3/−8.8/−10.3/−9.5/+6.7%；uniform −11.4/−13.4/−18.0/−23.6/−17.9%
（外区全线更负，质量系统性低估；M_flux(70)/M_true(70) = 0.80 vs control 1.06）。

### 负密度（Sobol 壳层 2048 方向 × 2–70 kpc；两臂同码同网格）

| 指标 | innerA control | uniform |
|---|---:|---:|
| 2–10 kpc 带 | 1.9% | **0.0%** |
| 10–30 kpc 带 | 2.4% | **0.0%** |
| 30–70 kpc 带 | 2.3% | **12.7%** |
| 均匀球探针（R=70 kpc, n=65536）30–70 kpc | 2.3% | **17.0%** |

uniform 臂逐半径：<33 kpc 全 0%；33→60 kpc 单调攀升（1%→5%→11%→17%→23%→30%），70 kpc 回落 6%。

### 判读

1. **T4 预测不成立，方向反转**。体积均匀把内区负密度清零（control 1.9–2.4%→0%），但外区负密度
   12.7–17.0%（control 2.3%）、外区 dM 恶化 1.5–2.6 倍；唯一改善带是 10–30 kpc（5.95→4.32%）。
2. **机制：约束重分配 × score 偏差交互**。92% 约束质量堆到 r>30 kpc，而外区正是 NF score 误差
   最大的壳层（T4 结论 3：偏差型、半径光滑、不可用加权/重采样修复）；负密度峰值在 60 kpc——
   约束最密与 score 最差的交叠处。control 的质量加权 + S1 权重实际起着压低外区坏 score 拉力的
   隐性正则作用。Plummer 的"均匀清零"依赖完美 score（P00 解析臂），不能迁移到 NF score 臂。
3. **内区代价来自约束稀疏**：2–10 kpc 仅 ~6k 约束点（control ~10 万级），质量尺度崩掉
   （dM 37%；总质量低估 20%）——Plummer 上内区代价轻微（0.65% 力中位）不可类比，
   Halo12 内区承载大部分质量。

### 对研究线的含义

- Halo12 control 的外区 dM ~8–9%（signed 约 −10%）**不是约束采样设计伪影主导**；
  改进杠杆在外区 score 质量（容量/径向参数化，与 T4a loss/采样双否证一致）或显式偏差建模/误差预算。
- control 现行约束设计（质量加权 + S1 + 权重）可重新表述为"对 score 偏差的隐性正则"，
  其内区负密度（1.9–2.4%）是可改进项但代价需控制。
- 中间设计（**半径平衡 + 无权重**，内区 ~43% 质量留驻，外区约束 ~4 倍于 S1）未测，
  是自然的后续子节点；判读框架沿用本节点（双指标 dM + 负密度、峰值壳层定位）。

### 限制

单次训练，种子方差未覆盖（方向在 dM 与负密度双指标、多个节点一致）；
外带 dM 中位仅 2–3 个节点（脚本自身提示 band-median 对节点集敏感）；
负密度先验（lambda=1）在两臂同样作用，先验-CBE 平衡的变化是结果的一部分。

## 图与数据

- [dM adjudication（双臂 dM 带 + signed 节点 + 角向负密度）](figures/uniform-ablation/pt-adjudication-lambda.pdf)（[npz](figures/uniform-ablation/pt_adjudication_lambda.npz)）
- [Phi 切片（真值+双臂+残差）](figures/uniform-ablation-slices/2d_slice_phi_truth.png)（[npz](figures/uniform-ablation-slices/2d_slice_phi_truth.npz)）
- [density 切片（log10 rho，真值+双臂+残差）](figures/uniform-ablation-slices/2d_slice_rho_truth.png)（[npz](figures/uniform-ablation-slices/2d_slice_rho_truth.npz)）
- [enclosed mass（M(<r) + 相对误差带）](figures/uniform-ablation-mass/enclosed_mass.png)（[npz](figures/uniform-ablation-mass/enclosed_mass.npz)）
- 汇总 JSON：[data/uniform_ablation_summary.json](data/uniform_ablation_summary.json)；
  负密度剖面：[data/uniform_ablation_negdensity.json](data/uniform_ablation_negdensity.json)；
  uniform 臂配置：[data/options.json](data/options.json)
- 生成脚本：仓库冻结分支 `orx/halo12-uniform-volume-df-constraint-ablation-2`（commit `382e84d`），
  runner `scripts/auriga/run_halo12_uniform_constraint_ablation.sh`；绘图用主线 `plot_potential_2d.py`、
  `plot_enclosed_mass.py`、`plot_pt_adjudication.py`。
- 日志：`orx logs 49af3bb9-e1c0-42d4-a509-a447a9a6f8cc`；uniform Phi checkpoint 与约束文件随 run 工作区保留在 gpu。
