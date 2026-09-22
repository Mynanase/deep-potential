# AGENTS.md — dpjax-phase2 仓库操作手册(供 AI coding agent 使用)

**deep-potential(dp-jax)Phase-2 基线**:用 JAX(Equinox + Flow Matching + 稳态 CBE)
从 Auriga Halo12 示踪粒子的相空间分布函数推断引力势。
主线已裁决:**冻结 clean+smooth 数据(n=1,619,615)+ S1 分层采样 + grid-decoupled
负密度先验 + innerA 半径平衡网格 + λ=10**。改动代码前先读完本文件。

## 0. 先读文档(按此顺序)

| 文档 | 内容 |
|---|---|
| `docs/phase-1-summary.md` | 六轮裁决表、关键数字、round-7 移交清单 |
| `docs/phase2-premerge-survey.md` | 修复账(§A)/图表账(§B)/keep 收录裁决(§D) |
| `docs/branches.md` | 分支谱系与维护约定(不删分支;"可删"标记需人工确认) |
| `scripts/auriga/README.md` | 上游迁移说明、单位体系、smoke→全量训练步骤 |
| `scripts/auriga/keep/README.md` | 长期保留脚本的来源与用法 |

## 1. 环境与测试

- 本仓库**没有自带 venv**。可用解释器:
  - `/Users/qttao/Documents/Research/flow-diff/myanase-deep-potential/.venv-halo/bin/python`(本地 CPU 验证,依赖齐)
  - 服务器:`/home/qiutao/miniforge3/envs/dp-jax/bin/python`(GPU runner 内写死)
  - ⚠️ 本地 conda `dp-jax` 环境缺 `e3nn_jax`,**不能**用来跑本仓库
- 测试(CPU,仓库根目录):

  ```sh
  env JAX_PLATFORMS=cpu <python> -m pytest tests/ -q
  # 预期:46 passed, 3 skipped(3 个 skip 因 truth HDF5 不在本地,正常)
  ```

- 测试只验证**机器**(解析势上的损失代数/求积/单位换算),不碰训练 checkpoint;
  新增损失/采样/审计逻辑必须配同风格测试(解析例子 + 精确断言)。

## 2. 硬性约定(违反即返工)

### 数据
- 训练数据已冻结:`halo12-clean-smooth.h5`(seed 0、mass weighting、lineage attrs 齐全)。
  `prepare_data.py` 输出已存在即报错——**绝不覆写冻结数据**;新数据用新文件名。
- 新实验 runner 必须像 `run_w1024_csmooth.sh` 一样幂等:先验 lineage 再复用。
- `data/` 已被 .gitignore 整体忽略;例外(smoke 数据、粒子真值网格 h5)已强制入库。

### 代码分层
- **计算/绘图严格分层**:计算脚本持久化数组/JSON;绘图脚本只读持久化数组,
  **绝不重新加载模型**。
- 坐标轴范围用手调 `set_ylim` 常数并在注释标注数据范围(2026-09-15 约定)。

### 科学红线(来自 auriga/README §1/§6)
- 单位:`q = x/L`、`p = v/V`,`L = 10 kpc`、`V = 100 km/s`;`Phi_phys = V²φ`。
- **不要添加 `--potential-ignore-nobs`**(会丢空间密度梯度,破坏完整稳态方程)。
- mass weighting 与 number weighting 是两个不同目标,**不得混用**。
- 负密度惩罚 `lambda_` 只抑制负 Laplacian,不保证正密度平滑;保留负密度和局部
  波动,**不截零掩盖问题**。
- `eta.attrs` 的 `r_in/r_out` 是 Phi 训练域(1≤r≤70 kpc),不是截断 DF。
- 星粒子质量径向直方图**不是**总密度真值;真值用粒子真值产品族(keep/ #5)。
- 模拟真势/加速度不进入训练输出。

### 仓库卫生
- 产物落 `runs/`(已忽略),代码入库、产物不入库;`*.png/pdf/json` 默认忽略
  (例外清单见 .gitignore)。
- `.note/` 是私有研究笔记,绝不入库、绝不外传内容。
- 提交信息风格(沿用主线):`feat(phi)`/`fix(runner)`/`eval(lambda)`/`fig(...)`/
  `docs:`/`bench:`/`test(phi)`/`chore:`,正文写动机与证据 run。
- 分支:实验分支从基线分出;**不删除任何分支**;裁决胜者在 docs 里记表。

## 3. 训练/评估入口

- 三阶段入口都是 `scripts/fit_all.py`(`--flow-training` / `--flow-sampling` /
  `--potential-training`),读 run 目录内 `options.json`;每阶段独立,重复执行
  **不是续训**——重训用新 run 目录。
- Runner 脚本(`scripts/auriga/run_*.sh`)模板:`set -eo pipefail`、
  PREFLIGHT 检查(python/数据/依赖/options 断言)、先跑相关单测再进训练、
  GPU 默认不抢(CUDA_VISIBLE_DEVICES 可外部覆盖)。
- 每轮实验后的标准验收 = 粒子真值裁决套件:
  `validate_enclosed_mass.py`(计算)→ `plot_pt_adjudication.py` / 
  `plot_shell_error_pairs.py` / `plot_pt_generations.py`(绘图)。
- 微基准:`scripts/bench_*.py`(dispatch overhead / orbit conv / spec3d score)。

## 4. 历史回溯

- 远程:`origin` = 上游 fork;`phase1-archive` = 本地旧仓库 `/Research/dpjax`
  (全部 Phase-1 冻结分支与 run 记账)。
- tags:`phase1/spine-lambda10|truth-line|phi-line|osc-spectra|base-figures`
  打在胜者尖端;决策→分支→run 的映射见 `docs/phase-1-summary.md`。
- run 日志:`orx logs <runId>`(旧项目 `07d8ee01` 永久可查);渲染图在服务器
  `/home/qiutao/.orx/runs/<runId>/`。
- round-7(振荡抑制三变体)cherry-pick 素材已就绪、未并入基线:
  osc-pair `7de9703`+`aa7b695`+`3d07384`、spectral-norm `002dcdd`、
  cosine-anneal `51fd27e`;⚠️ osc-pair 权重在 λ=1 语境校准,落到 λ=10 需重校准。

## 5. 修改代码前的最小检查

1. 改 `scripts/potential.py` 损失/采样 → 跑 `tests/test_grid_prior_loss.py`,
   并考虑 4/5/6/7 元 batch 解包契约(`_unpack_phi_batch`)。
2. 改 `validate_enclosed_mass.py` / `audit_df_constraints.py` → 跑对应两个测试文件。
3. 改 runner → 本地先用 `options-smoke.json` 路径过一遍 dry 结构。
4. 提交前:`git status` 应只有预期改动;不要顺手提交 `runs/`、`.note/`、数据。
