# Auriga 静态引力势恢复：研究与实验路线

## 结论

主线应保持 Green et al. (2023) 的两阶段静态 Deep Potential：

1. 学习平滑的六维 tracer phase-space distribution；
2. 冻结 DF/score，用静态 CBE 约束学习三维标量势；
3. 在 Auriga 模拟真值上验证势、加速度和失效区域。

Kalda & Green (2025) 最值得继承的不是 pattern speed，而是 ensemble、数据选择、
平滑 selection、导数后取 median、seed/重采样不确定性和非稳态 residual 诊断。
研究上的新增量不应只是“在 Auriga 上跑一遍”，而应是：

> 建立 truth-calibrated safe-use atlas：输出恢复的静态势/力场，同时给出仅依赖
> 可观测或内部诊断量的可信区域与 veto 区域。

## 当前证据

### 2023 静态方法

- 用可微 normalizing flow 学习 6D DF，再用 CBE 恢复静态势。
- 原实现使用 3 个 FFJORD block；势网络为 4 × 512 tanh。
- DF 和势分别保留 25% validation；DF 拟合后抽样约一百万个 phase-space 点训练势。
- 方法要求完整 6D 数据和在所选参考系中的统计稳态。
- residual 不只是训练量，也可以作为非稳态或模型失配的空间诊断。

### 2025 更新

- 16 个独立 flow 和 16 个势网络用于 seed/model convergence 不确定性。
- 在每个模型内部先求导，再对导数取 median，避免对分段模型中值直接求导。
- 明确区分模型 seed、观测误差重采样、bootstrap shot noise 和参考系不确定性。
- selection 必须平滑；不连续边界会制造虚假的 DF 梯度和质量密度。
- 空间高频波动可能来自架构、导数噪声、selection mismatch 或真实非稳态，不能
  未经 mock truth 检验就解释为物理结构。
- 论文明确建议把 Deep Potential 用到 Auriga/FIRE-2 等高分辨率 zoom simulation。

## 2024–2026 方法与工具更新

按当前实验收益排序：

1. **Truth-calibrated reliability gate**：把模拟真值只用于校准和测试，将 CBE
   residual、ensemble spread、split consistency、support/edge 指标和 score
   conditioning 映射为静态力场可信概率。这是当前最直接、最有科学价值的方向。
2. **Direct score estimation**：GalaxyScore 用 score matching 直接学习
   `∇ log f`，目标是减少 normalizing-flow 求导的计算开销和噪声。它值得作为
   第二阶段消融，但目前证据主要来自受控球对称 mock，不应立即替换已知基线。
3. **Conditional Deep Potential**：直接学习 `p(v|x)`，可对纯空间 selection
   保持不变，避免显式拟合复杂 selection function。适合后续 radial/dust-like
   mask 实验；clean snapshot 基线不需要先引入它。
4. **Structured/residual potential**：GalactoPINNS 类型的“解析势 + neural
   residual”可以在已知 acceleration 的模拟上作为 oracle 或势表示基线，并提供
   校准不确定性；但它使用 acceleration supervision，不能替代从 6D snapshot
   反演势的主任务。
5. **Spline-flow backend**：FlowJAX 当前提供 rational-quadratic spline、
   autoregressive/coupling/conditional flow，可以在 `dpjax.flows.api` 后作为
   独立实验后端，规避 FFJORD 的 ODE 稳定性和外区重尾问题。它仍明确提示 major
   release 可能 breaking，因此应固定版本并用 backend contract tests 隔离，
   不建议立刻重写整个仓库。Distrax 更轻量、更底层，可作为 bijector 组件备选。
6. **Diffrax FFJORD 路线仍有效**：官方 CNF 示例同时支持 exact divergence 和
   Hutchinson 近似。六维问题可以保留 exact trace 作为可解释基线，但应把 spline
   或 direct-score 的科学误差/成本对比列入消融。

## 数据真值边界

公开 AuriGaia 的三维 gravitational potential/force grids 只覆盖 Au6、Au16、
Au21、Au23、Au24、Au27，不含 Halo12。公开 Auriga 原始 snapshot 的共同字段
包括 `Coordinates`、`Velocities`、`ParticleIDs` 和逐粒子 `Potential`，但没有
标准的逐粒子 `Acceleration`。

因此第一项不可回避的决定是：

- 如果本地 Halo12 文件带有可靠 acceleration/force，直接采用并记录单位；
- 如果只有 `Potential`，可以先检查势的相对形状和加法常数对齐误差，但主 force
  metric 仍缺失；
- 若要尽快完成 force-calibrated baseline，先在有官方 grid 的 Au6/16/21/23/24/27
  中选一个，再把流程迁回 Halo12；
- 若坚持 Halo12，应从原 snapshot 全部质量组分重算或插值得到统一 force grid，
  并用独立方法验证其数值误差。

## 可执行实验阶段

### Phase 0：固定数据与真值契约

状态：核心实现已加入当前分支。

- 标准 schema：`dpjax.auriga.mock.v1`。
- 必需：`eta[N,6] = [x,y,z,vx,vy,vz]`。
- 可选真值：`particle_id`、`mass`、`potential`、`acceleration`、`source_index`。
- 优先按唯一 ParticleID 对齐；ID 缺失时使用显式六维容差的 KD-tree，并拒绝缺失、
  重复和多重匹配。
- 单位、中心位置、系统速度、selection 和随机种子写入 HDF5 attrs。
- 势误差只拟合不可观测的加法常数；禁止为改善结果拟合乘法尺度。

### Phase 1：最小静态 baseline

1. 固定一个 snapshot、一个 halo、一个无 selection 的 tracer sample。
2. 固定 galaxy center、bulk velocity、坐标方向、长度/速度/势/力单位。
3. 使用 `transform: none` 的 Halo12 v22 类 DF；不要绕过 v21 power transform
   的 Φ/CBE 门禁。
4. 至少训练 4 个独立 DF seed，先验证 NLL、score 分位数、sample profile 和外区
   support；通过后再扩大到 16 seed。
5. 使用 4 × 512 tanh 静态势网络和 `l2_reg=0.001` 基线；每个 DF seed 对应独立
   势训练。
6. 报告势 normalized RMSE、三维 acceleration vector relative L2、relative
   error 分位数、方向余弦和径向分箱误差。

首轮预注册门槛：

| 指标 | 首轮目标 |
|---|---:|
| Potential normalized RMSE | `< 0.10` |
| Acceleration median relative error | `< 0.10` |
| Acceleration p90 relative error | `< 0.25` |
| Median cosine similarity | `> 0.98` |
| Seed-to-seed coverage | 实际误差落入预测区间的比例需另行校准 |

这些值是工程 gate，不是最终科学结论；必须在等价静态 mock 上校准后冻结。

### Phase 2：可靠区域 / 静态性 gate

将空间划分为有最低 tracer 数量的 cell。每个 cell 计算不使用模拟真值的特征：

- CBE residual 的 median、p90、p99 和符号结构；
- DF/势 ensemble spread；
- 随机 half-split、角向 sector split 或 tracer-population split 的一致性；
- flow support / edge distance；
- score norm、score curvature 或 conditioning；
- selection realization variance；
- mass-positivity violation 比例。

模拟真值只生成标签：

`e_a,c = median(||a_hat - a_true|| / ||a_true||)`。

训练一个简单、可校准的 gate 预测 `P(e_a,c < 0.1 | diagnostics)`。空间 sector、
radial shell、selection family 和 seed 必须成组 hold out，禁止同一邻域泄漏到
训练和测试两侧。输出：

- accepted-volume fraction；
- accepted cells 的 force-error coverage；
- rejected cells 的主要失败类型；
- 失效区与 substructure/非稳态结构的事后关系。

`subhalo_id` 等 simulation-only 标签只能用于样本构建或事后解释，不能作为部署
时的 gate 输入。

### Phase 3：按因果顺序做消融

1. **数据量**：`N = 1e5, 3e5, 1e6`。
2. **空间范围**：内区、外晕、多个局部 patch。
3. **substructure**：保留、掩膜、只保留平滑主晕。
4. **selection**：clean、径向 completeness、角向/dust-like mask。
5. **DF backend**：FFJORD exact trace vs rational-quadratic spline。
6. **导数**：flow autodiff score vs direct score matching。
7. **势表示**：free MLP vs analytic prior + neural residual。
8. **不确定性**：seed ensemble、bootstrap、selection realization、split
   consistency，检查 interval coverage 而不只报告 error bar。

每次只改变一个主要因素，并始终在相同 truth metric 与空间 holdout 上比较。

## 现在不应做的事

- 不把 pattern speed 设为主目标。
- 不把 CBE training loss 当作势恢复成功的充分证据。
- 不绕过非线性 coordinate transform 的物理链式法则门禁。
- 不在 baseline 尚未通过 truth gate 前做 joint fine-tuning。
- 不先追求 Gaia selection realism，再补模拟真值。
- 不因 FlowJAX 或 score matching 更新而一次性替换已知 FFJORD 基线。

## 下一项实现

完成 Phase 0 后，下一项代码工作应是：

1. 对实际 Halo12 HDF5 运行 `experiments.prepare_auriga`；
2. 确认 `Potential`、`Acceleration`、单位和 row alignment；
3. 若 acceleration 缺失，决定“重建 Halo12 force grid”还是“先切换公开有 grid
   的 Auriga halo”；
4. 完成 v22 DF 后运行 `phi_halo12_static_v1.yaml`；
5. 用 `experiments.eval_auriga_truth` 生成第一份 truth metrics；
6. 只有拿到这份报告后，才决定优先修 FFJORD、换 spline flow，还是直接 score。

## 主要来源

- Green et al. (2023), [arXiv:2205.02244](https://arxiv.org/abs/2205.02244)
- Kalda & Green (2025), [arXiv:2507.03742](https://arxiv.org/abs/2507.03742)
- Putney et al. (2024), [arXiv:2412.14236](https://arxiv.org/abs/2412.14236)
- Kalda & Green (2025), conditional Deep Potential,
  [arXiv:2512.02115](https://arxiv.org/abs/2512.02115)
- ClearPotential, [arXiv:2512.09989](https://arxiv.org/abs/2512.09989)
- GalactoPINNS, [arXiv:2606.18386](https://arxiv.org/abs/2606.18386)
- [FlowJAX](https://github.com/danielward27/flowjax)
- [Distrax](https://github.com/google-deepmind/distrax)
- [Diffrax CNF example](https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/)
- [Auriga public data](https://wwwmpa.mpa-garching.mpg.de/auriga/data.html)
- [AuriGaia force/potential grids](https://wwwmpa.mpa-garching.mpg.de/auriga/gaiamock.html)
