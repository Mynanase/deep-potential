# Phase-1 阶段总结（dpjax × Auriga Halo12，orx 时代）

**区间**：2026-09-17（orx 导入，baseline `codex/upstream-sync-2026-09-12`）→ 2026-09-22
**规模**：74 个实验节点 / 78 个本地分支 / 110+ 次 run（旧项目 `07d8ee01` 永久可查）
**产出基线**：`codex/phase2-baseline-2026-09-22`（本文档所在分支）
**一句话**：在 Auriga Halo12 上把 deep-potential（DF + 稳态 CBE + 神经势场）做成可复现流水线，
经 6 轮已裁决实验确立「冻结清洗数据 + S1 分层采样 + grid-prior 正则 + innerA 半径平衡网格 + λ=10」主线，
并以粒子真值裁决套件建立了逐轮验收方法学；第 7 轮（振荡抑制三变体）修复就绪、留待 Phase-2。

---

## 1. 六轮已裁决决策

| 轮 | 决策点 | 选项 | 胜者 | 证据 run | 胜者分支 |
|---|---|---|---|---|---|
| 0 | 基线与容量 | smoke → A100 w128 → w1024 | w1024 三段式 | 多次 done run | `orx/w1024-phase-2-*` |
| 1 | 训练数据 | 原始 / clump 移除 / velocity-gate / debris cascade | **clean+smooth 并集冻结** | — | `orx/freeze-training-data-clean-smooth-union-default` |
| 1b | 路线闭环 | — | 幂等冻结数据路径 + 复现检查 | — | `orx/route-closure-integration-and-verification` |
| 2 | 采样 | S1 分层 vs s11 独立重排 | **S1**（s11 仅作复核，结论一致） | — | `orx/s1-stratified-sampling-on-frozen-clean-smooth-w1` |
| 3 | 负密度正则 | 无 / volume-grid / **grid-decoupled prior** | grid-decoupled | `09faad30`（57m, done） | `orx/grid-decoupled-negative-density-prior-on-s1-w102` |
| 4 | inner band 修复 | A 半径平衡网格 vs B 内区加权 | **A** | `313e0cc7`（裁决）/ `2b32eb04` `7016399a`（训练） | `orx/inner-band-fix-a-radius-balanced-prior-grid` |
| 5 | λ 强度 | 0.1 / 1 / 10 | **λ=10**（λ0.1 负密度塌缩至 ~12%） | `369e3eae`（六模型裁决）/ `e1cf8f74` `f6c902f6`（训练） | `orx/lambda-sweep-10-on-radius-balanced-grid-prior` |

**关键数字**（出处：`plot_pt_adjudication.py` docstring，run `369e3eae`）：
- λ=10 负密度占比 ~0.8%；λ=0.1 塌缩至 ~12%；innerA 各带 dM 1.25 / 5.95 / 8.76 / 8.14（%）。
- 数据：Halo12 全部恒星 1,652,969 个，r < 75 kpc；训练域 1 ≤ r ≤ 70 kpc。
- 冻结数据链：`halo12_all_mass_clean_outer_clump_smooth.h5` → `prepare_data.py` → `halo12-clean-smooth.h5`（seed 0，mass weighting，禁覆写）。

## 2. 第 7 轮（进行中 → 移交 Phase-2 round-1）

| 变体 | 失败 run | 原因 | 修复 commit | 状态 |
|---|---|---|---|---|
| osc-pair 平滑惩罚（权重 7.612e-06 谱诊断校准） | `651f025e` | `_unpack_phi_batch` 7 元解包 | `7de9703` | 已修复待重跑 |
| spectral-norm ceiling | `e18f6d6d`（cancelled） | 非 Linear 叶子 + q_pair | `002dcdd` + 2 行补丁 | 已修复待重跑 |
| λ cosine anneal 10→1 | `65ab84cb` | `lambda_` traced bool | `51fd27e` | 已修复待重跑 |

> ⚠️ osc-pair 权重在 λ=1 语境校准；落到 λ=10 新基线后需重新校准或重新验证。

## 3. 方法学资产

- **粒子真值裁决套件**：`build_total_matter_truth` → 网格产品 → `plot_pt_adjudication`（上下格式
  enclosed-mass 评估图），每轮实验后以同一真值验收 —— Phase-2 沿用。
- **长期保留脚本**：`scripts/auriga/keep/`（径向族 ×4、真值对比两主力、真值构建 ×4、figstyle），见其 README。
- **修复账 / 图表账**：`docs/phase2-premerge-survey.md`（§A 修复 15 条、§B 图表 31 脚本、§D 裁决）。
- **汇报材料**：`docs/talk-2026-09-18-halo12-dpjax.md`（组会讲稿，覆盖至 Phase-1 DF 审计时代）+ `slides/`。

## 4. 回溯索引

| 需要什么 | 怎么查 |
|---|---|
| 任意 run 的日志/结果 | `orx runs 07d8ee01…`（旧项目，已归档名）→ `orx logs <runId>` |
| 某轮的代码状态 | 轮次表"胜者分支"列；见下方 tags |
| 图在哪 | survey §B 各表"最新渲染 run"列 → 服务器 `/home/qiutao/.orx/runs/<runId>/` |
| 主线代码怎么长的 | §1 轮次表自上而下即祖先链（已验证） |

**回溯 tags**（打在胜者尖端）：
`phase1/spine-lambda10`（=λ10 主线尖端）、`phase1/truth-line`（cap-w512）、
`phase1/phi-line`（phi-profiles）、`phase1/osc-spectra`、`phase1/base-figures`（af32ee9）。
