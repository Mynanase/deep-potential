# AGENTS.md — dpjax / Auriga Halo12 deep-potential(Phase-2 基线)

适用于在本仓库工作的所有编码与运行代理(Codex、Claude Code 等)。本文件承自
`codex/upstream-sync-2026-09-12` 分支的 AGENTS.md(0a23194,2026-09-17),按
Phase-2 基线现状修订;科学背景、单位推导与完整操作手册见
`scripts/auriga/README.md`。本文件只固定必须遵守的约定,冲突时以用户的当次指令为准。

## 背景与文档索引

主线已裁决(Phase-1 六轮,2026-09-17→09-22):**冻结 clean+smooth 数据(n=1,619,615)
+ S1 分层采样 + grid-decoupled 负密度先验 + innerA 半径平衡网格 + λ=10**。
决策/证据/分支映射:`docs/phase-1-summary.md`;修复账与图表账:
`docs/phase2-premerge-survey.md`;分支谱系:`docs/branches.md`;长期保留脚本:
`scripts/auriga/keep/`(README 标注来源,暂不接主流程)。

## 计算环境

- 实验计算在 ssh 主机 `gpu`(8×A100-40GB,驱动 570 / CUDA 12.8)上进行,单卡
  `CUDA_VISIBLE_DEVICES=0`(session 约定一 run 一卡,runner 里可外部覆盖)。
- 服务器 Python 一律使用现成 conda 环境
  `/home/qiutao/miniforge3/envs/dp-jax/bin/python`。不要新建 venv,不要向该环境
  安装或升级包;依赖有缺口时先报告再行动。
- 本地 CPU 验证用 `/Users/qttao/Documents/Research/flow-diff/myanase-deep-potential/.venv-halo/bin/python`;
  ⚠️ 本地 conda `dp-jax` 环境缺 `e3nn_jax`,不能跑本仓库。
- `scripts/auriga/requirements.txt` 是精确复现用的版本快照,仅在明确决定重建环境时使用。
- JAX 进程统一加 `JAX_PLATFORMS=cuda`(本地测试用 `JAX_PLATFORMS=cpu`)与
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

## 运行契约

- 冒烟基线固定入口:仓库根目录 `bash scripts/auriga/run_smoke_baseline.sh`
  (`--flow-training` → `--flow-sampling` → `--potential-training` 三个独立进程,
  再 `plot_potential.py` + `smoke_summary.py`)。
- 正式训练走 `scripts/auriga/run_w1024_*.sh` 家族:模板为 `set -eo pipefail` +
  PREFLIGHT(解释器/数据/依赖/options 断言)+ 先跑相关单测再进训练;幂等——
  目标数据存在时先验 lineage 再复用。
- 三个训练阶段必须逐个检查退出码后再继续;本仓库不承诺中途续训,重新训练一律
  使用新的 run 目录,不要混入旧 checkpoint 或旧梯度文件。
- `fit_all.py` 从 `<run-dir>/options.json` 读参数:运行前必须先把 options 复制进
  run 目录(冒烟用 `scripts/auriga/options-smoke.json`,全量起点 `options.json`)。
- **计算/绘图严格分层**:计算脚本持久化数组/JSON,绘图脚本只读持久化数组、
  绝不重新加载模型;坐标轴范围用手调 `set_ylim` 常数并在注释标注数据范围
  (2026-09-15 约定)。
- 每轮实验后的标准验收 = 粒子真值裁决套件:`validate_enclosed_mass.py`(计算)→
  `plot_pt_adjudication.py` / `plot_shell_error_pairs.py` / `plot_pt_generations.py`(绘图)。
- 禁止给 Halo12 运行加 `--potential-ignore-nobs`(丢掉空间密度梯度,破坏完整
  稳态方程)和 `--basic-potential-benchmarking-gaia-units`(单位体系不符)。
- 实验一律通过项目的实验编排启动,不要手工在服务器起长任务;运行快照只包含
  已提交内容,改动必须先 commit。

## 随机种子

- 四层种子固定:数据洗乱 `--seed 0`(`prepare_data.py`,写入 attrs `shuffle_seed`);
  DF flow `df.seed=0`;score 采样 `flow_sampling.seed=1`;势网络 `Phi.seed=2`。
- 不要给 `fit_all.py` 传 `--seed`:它会同时覆盖三个训练种子,破坏上述分层。
- 固定实验需一并记录 input HDF5、run/options.json、源码提交哈希与模型文件。

## 数据

- 训练数据已冻结:`halo12-clean-smooth.h5`(服务器 DATA_ROOT;seed 0、mass
  weighting、lineage attrs 齐全)。`prepare_data.py` 拒绝覆盖已存在输出;新数据
  用新文件名;不得把已无量纲化的 h5 再喂给转换器;输入必须带 kpc / km/s 元数据。
- 已入库数据仅两件:`data/auriga/halo12-smoke.h5`(512 粒子冒烟)与
  `data/auriga/halo12_particle_truth_grids.h5`(粒子真值网格);`data/` 其余被忽略。
- 默认质量加权 `m/mean(m)`;`--weighting number` 是不同的科学目标,两者结果
  不要混用比较。

## 物理与单位

- 无量纲化 `q=x/L, p=v/V`,L=10 kpc、V=100 km/s;保存的 `dlnf_deta` 是对输入
  `(q,p)` 的导数;`Phi_phys = V²φ`。
- 由势推得的 ρ 是总引力源密度,不是恒星质量直方图;轴剖面不是球壳平均。
- `eta.attrs` 的 `r_in/r_out` 是 Phi 训练域(1≤r≤70 kpc),不是截断 DF。
- 负密度惩罚 `lambda_` 只抑制负 Laplacian,不保证正密度平滑;保留负密度和局部
  波动,不截零掩盖问题。模拟真势/加速度不进入训练输出。

## Git 与实验树

- 实验分支前缀 `orx/*`:某节点一旦有运行给出结果即冻结,禁止改写;后续变化开
  子节点。不删除任何分支;"可删"标记需人工确认。
- 不向上游 push;`origin` 仅用于可见性。历史回溯:`phase1-archive` 远程(本地
  旧仓库 `/Research/dpjax`,全部 Phase-1 冻结分支)、tags `phase1/*` 五枚、
  run 日志 `orx logs <runId>`(旧项目 `07d8ee01` 永久可查)。
- 提交信息风格:`feat(phi)`/`fix(runner)`/`eval(lambda)`/`fig(...)`/`docs:`/
  `bench:`/`test(phi)`/`chore:`,正文写动机与证据 run。
- 提交前轻量检查:`bash -n`(shell)、`python -m py_compile`、
  `env JAX_PLATFORMS=cpu <python> -m pytest tests/ -q`(预期 46 passed, 3 skipped;
  skip 因 truth HDF5 不在本地)。测试只验证机器(解析例子上的代数/求积/单位),
  不碰 checkpoint;新增损失/采样/审计逻辑必须配同风格测试,注意
  `_unpack_phi_batch` 的 4/5/6/7 元 batch 解包契约。
- 产物落 `runs/`(已忽略),代码入库、产物不入库;`.note/` 是私有研究笔记,
  绝不入库、绝不外传内容。

## Phase-2 round-1 移交(进行中)

振荡抑制三变体修复就绪、待 cherry-pick(源在 `phase1-archive`):osc-pair
`7de9703`+`aa7b695`+`3d07384`、spectral-norm `002dcdd`、cosine-anneal `51fd27e`;
⚠️ osc-pair 权重在 λ=1 语境校准,落到 λ=10 基线需重校准。落地后更新本节。
