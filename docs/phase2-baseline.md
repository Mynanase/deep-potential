# Phase-2 基线裁决（2026-09-24）

用户于 2026-09-24 拍板两项基线裁决；本文为 phase-2 的控制配置定义，
落地分支 `orx/phase-2-baseline-w1024-production`（根节点 `5d108a36`）。
`docs/phase-1-summary.md` 中"主线胜者 = λ=10"的解读自本文起降级为 phase-1 裁决史。

## 裁决 1：控制线 = innerA（λ=1）

Phase-2 control configuration 是 innerA 契约：

- 分支 `orx/inner-band-fix-a-radius-balanced-prior-grid` @ `5f76faa`，
  冻结训练 run `2b32eb04`（节点 `9788e6de`）；
- w1024 三阶段、`halo12-clean-smooth.h5`（seed 0、mass weighting）、S1 分层采样、
  半径平衡（r = R·u）解耦负密度先验、`lambda_=1`；
- 固定 run command：`bash scripts/auriga/run_w1024_csmooth_s1_gridprior.sh`
  （本分支恢复 innerA 版 runner，并新增 `lambda_==1` 预检 assert）。

λ=10 主线（`b9a981f`，tag `phase1/spine-lambda10`，训练 run `e1cf8f74`）
降级为对照锚点：其负密度更好（约 0.8%），但 30–50 kpc dM 过正则；
innerA 带级 dM（2-10/10-30/30-50/50-70 = 1.25/5.95/8.76/8.14%，run `313e0cc7`）
与谱诊断 f_hi<10/<5 = 10.7%/2.8%（run `e4d4db5d`）整体更均衡；
且 osc-pair 权重 7.612e-06 正是在 λ=1 语境校准，选 innerA 使 round-1 契约无需平移。

**子节点规则**：phase-2 子节点创建时一律显式
`--run-command 'bash scripts/auriga/run_w1024_csmooth_s1_gridprior.sh'`。
根节点 `5d108a36` 创建时记录的 command 是 clean 线历史驱动器
`run_w1024_phase2.sh`（其唯一 run `6b6e13aa` 已冻结于 phase-1），
run command 创建后不可编辑——**不要直接运行根节点**，以本节规则为准。

## 裁决 2：96³ 为生产 truth；~10 kpc 定量前沿；<5 kpc 诊断域

- 生产 truth 固定为 `data/auriga/halo12_particle_truth_grids.h5`
  （96³、1.5625 kpc 胞元、恒星系坐标、总物质 cell-average 直方图，
  谱系 `dpjax.particle-truth-grids.v1`）。**停止追求更细网格**，
  不再构建 128³/192³ 生产 truth。
- 依据（run `b6420956`，节点 `c48076d7`，2026-09-24）：
  - 严格分辨率收敛 gate（64→96→128→192，逐 log-λ bin 中位
    |P_N2−P_N1|/P_N2 ≤ 10%）全尺度未过——细化并不买到收敛；
  - shot-noise A/B 分摊：96³ 在 λ=10 kpc 噪声占比 9.6%（信号主导），
    5 kpc 37.3%、4 kpc 54.0%、2.5 kpc 61.6%；
  - 更细反而更噪：192³ 在 10 kpc 噪声占比 27.3%、5 kpc 66.3%。
- 操作口径：
  - **定量域**：λ ≳ 10 kpc 的谱量（f_hi<10 kpc 列）、径向带 dM、
    30–70 kpc 负密度带——可作模型间结论与验收判据；
  - **诊断域**：λ < 5 kpc 只作模型间相对比较与训练监控，不作绝对结论
    （truth 本体噪声主导；A/B 谱一致性 2.8–9.4% 说明可复现，但不改变噪声占比）；
  - Nyquist 保护 3.125 kpc（96³ 胞元），2.5 kpc 列沿用 cell-limited 标注。

## 对 round-1（振荡抑制线）的影响

- osc-pair：权重 7.612e-06 依 innerA d=4 kpc 校准，直接有效，无需重校准；
- λ cosine-anneal：重新定位为"从强抑制起步的课程选项"（λ10 降为参照 regime）；
- spectral-norm ceiling：与 λ 无关，契约不变；
- 三者验收统一：粒子真值裁决套件 + 密度谱诊断，按上述定量/诊断域口径读数。
