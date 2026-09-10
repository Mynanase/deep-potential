# 总密度非负但起伏明显：分步实验方案

制定日期：2026-09-08。本文是一份按结果推进的研究操作单，不是需要一次跑完的参数扫描。

研究现象：原先外围密度出现负值；增强负密度约束、增加外围点后，负值减少，但密度场仍有明显起伏。要判断这些起伏来自数值计算、约束覆盖、优化、DF score、势网络表示、有限数据，还是物理假设与真实结构。

先完成第 00–06 步，只分析已有模型；得到结果后选择后续分支。每次可以只做一个小步骤。没有必要在原因已经清楚时继续跑完所有实验。

## 使用方式与当前接口

本文以 Halo12 的现象为主线，用 Plummer 作为有解析真值的控制实验。具体实验目录、原惩罚权重和新增外围点的生成方式目前尚未确认，第 00 步负责锁定；文中的候选数值不是已经确认适合该实验的参数。

步骤标记：

- **现有入口**：当前代码已有对应训练、评估或读取函数。
- **离线分析**：读取已有数组，在 `notebooks/figure_debug.ipynb` 中顺序执行研究代码；数值产物和绘图分开。
- **需补充实现**：当前 CLI 没有这项能力。先实现并验证该小功能，再执行对应实验；不要把本文的概念字段直接加入 YAML。

当前代码核对结果：

| 能力 | 当前状态及使用限制 |
|---|---|
| 调整质量约束强度 | `phi.model_overrides.train.lambda_mass` |
| 调整负密度惩罚形状参数 | `phi.model_overrides.train.beta`；主因排查期间固定 |
| 调整质量约束点数 | `mass_batch_size`，取当前训练 batch 的前若干个位置；超过 batch size 无效 |
| 增加独立外围空间点 | 当前本地 `run_phi.py` 未提供独立采样器；可能存在于用户其他实验版本，需第 00 步核对 |
| 径向重加权 | `reweight.gamma`；当前同时影响 CBE 和质量惩罚，不能当作只调整外围质量约束 |
| 固定一个外部 DF | `phi.df_run`；必须固定实际 checkpoint，避免其目录继续训练后被加载为新的 latest |
| Plummer 解析 score | `phi.model_overrides.score.source: plummer_analytic` |
| Phi 网络配置 | `phi.model_overrides.potential.hidden_sizes`、`output_scale` |
| Phi 随机种子 | `phi.seed` 同时影响初始化和 batch 随机顺序，现有接口不能只改变其中一个 |
| 真正独立的 Phi 验证行 | 当前 Phi 使用整个 DF support；DF validation 行也可能参与 Phi 训练，需补充拆分才是独立验证 |
| 常规 Phi 评估 | `eval_phi` 加载 latest checkpoint；径向曲线沿 x 轴，二维切片是 z=0 平面 |
| 任意固定三维评估点、指定历史 step | 数值函数/restore 函数可复用，但当前阶段 CLI 没有直接入口，需补充评估实现 |
| 模型加载的跨平台恢复 | GPU checkpoint 能否在 CPU 恢复需实际检查；不通过时在原训练环境评估 |

代码参考：[Phi 训练](../experiments/run_phi.py)、[Phi 评估](../experiments/eval_phi.py)、[负密度惩罚](../dpjax/physics/cbe.py)、[势及导数](../dpjax/models/potential.py)、[score 来源](../experiments/workflows/score_sources.py)。

## 总体顺序与计算预算

| 阶段 | 步骤 | 本轮回答的问题 | 新训练预算 |
|---|---|---|---|
| A：固定问题 | 00–02 | 究竟比较哪两个模型，比较区域和指标是否一致？ | 0 |
| B：已有结果检查 | 03–06 | 起伏是真实网络输出、数值/显示误差，还是主要在弱支撑区？ | 0；只重评估 |
| C：小型因果实验 | 07–12 | 权重与外围采样分别改变了什么？ | 先 1 个基准、再最多补 3 个因子组合 |
| D：定位学习阶段 | 13–16 | Phi 随机性、优化和 DF score 哪一个主导？ | 先 1 次重复；有必要再做 oracle 和 DF 重训 |
| E：容量与数据 | 17–20 | 网络是否过强/不足，数据是否限制空间分辨率？ | 每次只新增 1 个设置 |
| F：物理与修正 | 21–24 | 非平衡/selection 是否重要，什么修正能保留真实结构？ | 根据前面结论分支 |

不预估具体 GPU 小时：先用第 07 步的真实速度、JIT 编译时间和显存占用估算。计时分开记录编译、训练、评估；不要用 CPU smoke 的速度推算真实多卡收益。

## 每一步都使用的记录规则

每个实验有一个新的 run YAML 和新的 `name`、`output_dir`。命名示例：`density-c00-base-s42`、`density-c01-weight-s42`，目录可用 `runs/halo12/density-c01-weight-s42/`。

每次只记录五项即可：

```text
步骤/实验编号：
本次唯一改变：
固定项及实际 checkpoint：
结果数字、图和文件路径：
判断：支持什么 / 尚不能排除什么 / 下一步做什么：
```

正式输出继续使用 `results/data`、`results/figures`、`results/debug`。离线数值建议用新的 `density_audit_sXX_*.npz/json` 文件名，调试图用同前缀存到 debug；不要覆盖常规 `phi_diagnostics.npz` 或手工修改正式 manifest。对照集坐标可以在一个选定的基准 run 下保存一次，其他实验记录同一文件和 SHA-256。

旧模型的额外评估使用新文件名或独立审计输出位置，保留原结果。完整 checkpoint 及其 config、Normalizer、DF selection 必须保持配套。继续训练时不要同时读取其 latest 做跨模型比较。

新的训练设置使用 `execution.resume: false`。改 loss、网络或 epochs 后不能在原输出目录打开 resume 混跑；`phi.init_params` 是仅加载参数、重新初始化优化器的实验，不等同完整恢复。

## A. 固定问题

### 00｜找出“修改前”和“修改后”的确切实验

**本步只做盘点，不训练。现有入口。**

1. 找到两个模型的 run YAML、`phi/config.yaml`、DF 路径、Phi checkpoint step、`metrics.csv` 和原始密度数组。
2. 抄下实际 `lambda_mass`、`beta`、`mass_batch_size`、`reweight.gamma`、网络、`output_scale`、L2、batch size、总更新步数、优化器和学习率曲线。
3. 确认改动后是从头训练，还是从原模型加载参数继续调整；如果是继续调整，记录起始 checkpoint 和优化器是否重置。
4. 查清“外围点”是什么：仅空间坐标用于非负约束；六维坐标参与 CBE；已有粒子重采样；flow 生成；还是新增独立粒子。
5. 保存每个模型对应的代码 commit 和未提交 diff。旧训练的 commit/代码找不到时，写“来源不完整”，不要把旧/新结果当严格因果实验。

`git diff` 不包含未跟踪的新代码文件。当前工作区新增的 `figure_io.py`、`evaluation_units.py` 等若参与运行，也要作为运行代码快照保存；不要把无关未跟踪研究材料一起打包。

本步可以直接填写这张表，不必先写完整报告：

| 字段 | 修改前 | 修改后 |
|---|---|---|
| run 配置/输出目录 | 待填 | 待填 |
| 代码 commit 与未提交改动 | 待填 | 待填 |
| DF 路径、step、数据/Normalizer 哈希 | 待填 | 待填 |
| Phi 路径、step、实际更新数 | 待填 | 待填 |
| lambda、beta、L2、output scale | 待填 | 待填 |
| 质量点数/来源/覆盖、gamma | 待填 | 待填 |
| 网络、学习率、batch、seed | 待填 | 待填 |
| 从头训练/加载旧参数/完整恢复 | 待填 | 待填 |
| 评估区域、网格、单位、原始数组 | 待填 | 待填 |

终端先运行以下只读命令；在实际保存模型的机器上执行：

```fish
conda activate dp-jax
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
python -m experiments.list_runs --json
```

**本步交付：** 一张旧/新参数对照表，明确哪些参数不同；外围点类型一句话。

**判断：** 若不仅权重和采样变了，先承认存在混杂因素。可继续做现象检查，但第 08–10 步必须重新建立配对实验。

### 01｜只确定可信区域、目标尺度和真值

**本步只定义比较范围。离线分析。**

1. 区分训练数据覆盖、DF support、额外惩罚点覆盖、评估网格覆盖。画在同一坐标系，保留 sigma clipping 或 clump mask 的边界。
2. 预先划出内部可信区、过渡区、外围弱支撑区、支撑以外区。依据数据覆盖确定，不依据哪一片密度更好看来确定。
3. 指定科学目标：例如恢复某尺度以上的三维密度，或只恢复壳平均密度。没有必要要求任意小尺度都准确。
4. 指定三个固定的诊断平滑尺度 `ell/2, ell, 2*ell`。Halo 的 kpc 与 Plummer 模型单位分开，不能照搬数字。
5. 如果要判断“准确”，确认真值：Plummer 用解析总密度；Halo 需要所有引力成分及一致平滑尺度的模拟密度。只有恒星 tracer 的计数不能作总密度真值。
6. 当前 Auriga validator 只保证检查势/加速度，并记录 `total_density: False`。没有总密度真值时，结论限于非负性、稳定性、尺度依赖和动力学一致性。

**本步交付：** 固定区域 mask、单位、三个尺度、真值来源。对旧/新模型使用同一个 mask；mask 变化要开启新的审计版本。

### 02｜只定义指标，不挑模型

**本步只做一个统一评分表。离线分析。**

对每个固定区域分别记录以下指标。空间网格/均匀体积点用于体积统计，tracer 点用于 tracer 加权统计，两者不要混为一栏。

| 指标 | 定义与排查用途 |
|---|---|
| 有限值比例 | 单独报 NaN/Inf；不能将其过滤后宣布结果良好 |
| 负密度比例 `Fneg` | 有限、受支持点上 `rho < -rho_tol` 的比例，同时报告 `rho < 0` 的比例 |
| 负密度幅度 | `mean(max(-rho, 0))` 和低分位数；区分很小的数值负值与严重负密度 |
| 背景变化 `B_ell` | 新旧平滑场之差的 RMS，检查整体抬高/压低 |
| 起伏绝对幅度 `A_ell` | `sqrt(mean((rho - smooth_ell(rho))**2))` |
| 起伏相对幅度 `Q_ell` | `A_ell / rho_ref`，同一区域所有实验使用固定的 `rho_ref` |
| 尖峰 | 密度和起伏的 1%、50%、99% 分位数、最大绝对值和坐标 |
| 动力学代价 | 同一组六维点上的 CBE RMS、绝对值 99% 分位数、空间分区 residual |
| 真值误差 | 有真值才报密度误差、加速度向量误差；势先对齐一个加法常数 |

`rho_ref` 优先使用该区域固定真值的 RMS；无真值时使用冻结的基准背景 RMS。接近零时改报绝对误差，不除以每个模型自己的背景。否则模型抬高密度就可能让“相对起伏”变小。

`rho_tol` 在第 04 步通过数值稳定性估计后固定，同时保留不带容差的原始符号结果。

同时冻结两套点评估文件，而不只记录一个 seed：

- **排查点集：** 一套固定的空间点/网格，以及一套固定的真实六维点；保存坐标、源行索引、mask、单位、seed 与 SHA-256。不同 DF support 时用预先定义的共同可信区域，不把各自随机抽到的点当配对比较。
- **最终确认点集：** 另一个 seed 和独立空间位置，留到第 24 步再看。若要求独立观测检验，真实六维行必须在 DF、Normalizer 拟合及 Phi 训练前预留，不能在训练过后才称为 held-out。

已有模型不能事后获得新的独立训练外观测。对它们使用固定新空间点检验函数场稳定性，并明确 residual 来自训练分布内诊断。第 07 步开始的新系列若要使用真正 held-out CBE，需要先补训练/验证行拆分与其持久化；否则继续标注为分布内诊断，不把这一点隐藏在“验证 loss”名称里。

**筛选规则：** 起伏更小但加速度明显变差、背景大幅漂移或结构被抹掉，不能判定改进。先报告各指标间的取舍。

可预先采用“起伏幅度改善至少 20%、有真值的加速度相对误差增加不超过 0.01”为第一轮候选规则；这是可修改的研究筛选标准，不是普适物理阈值，也不是统计显著性。应在看新实验结果前结合科学目标确认。最终改进还须超过重复种子和数值误差带，并在独立确认点集上复现。

## B. 只使用已有模型

### 03｜排除图片显示造成的起伏错觉

**一个离线会话；不训练。**

1. 读取旧/新的原始 `phi_diagnostics.npz`；使用 artifact loader 兼容旧 `eval/` 路径。
2. 比较网格坐标是否完全一致，注意 x 轴剖面不是球平均，z=0 切片不是三维投影。
3. 原始密度统一使用线性色标看一次；再用固定同一 `linthresh` 的 symlog 看一次。负值不截断，零点不补成小正数。
4. 旧/新图使用共同 color limits。另保存允许看清弱结构的局部图，但标清其范围。
5. 直接画一条穿过明显起伏区域的原始像素行，关闭图像插值，检查波动是否已经存在于数组。

当前绘图函数会根据值是否全正选择 log/symlog，所以两张自动生成图的视觉幅度不能直接比较。以下代码只读取和检查数组；替换两个路径后，在 notebook 顺序执行：

```python
from pathlib import Path
import numpy as np
from experiments.diagnostics.phi_artifacts import load_phi_diagnostics

old_run = Path("runs/填写修改前实验目录")
new_run = Path("runs/填写修改后实验目录")
old = load_phi_diagnostics(old_run)
new = load_phi_diagnostics(new_run)
print(old.keys())
print(new.keys())
np.testing.assert_array_equal(old["slice_x"], new["slice_x"])
np.testing.assert_array_equal(old["slice_y"], new["slice_y"])
rho_old = old["slice_density"].astype(np.float64)
rho_new = new["slice_density"].astype(np.float64)
print("finite:", np.isfinite(rho_old).mean(), np.isfinite(rho_new).mean())
assert np.isfinite(rho_old).all() and np.isfinite(rho_new).all()
print("negative:", np.mean(rho_old < 0), np.mean(rho_new < 0))
print("old percentiles:", np.percentile(rho_old, [1, 50, 99]))
print("new percentiles:", np.percentile(rho_new, [1, 50, 99]))
```

**本步交付：** 两张统一色标图、一条同位置密度曲线、上述数字。网格不同就停止逐点比较，先执行第 04 步统一重评估，不通过插值制造“完全一致”的坐标。

### 04｜排除评估网格和浮点精度的问题

**只重评估一个已有 checkpoint。现有入口 + 少量评估实现。**

分三次小操作进行：

1. **网格测试：** 选一个有明显起伏的小区域，使用等距 `h` 和 `h/2` 网格；用 `n` 和 `2*n-1` 个端点对齐网格使粗网格点是细网格子集。仅绘图采样变密不应改变共同坐标处的模型输出。当前常规入口可改 `slice_grid`，但会覆盖评估文件，旧模型审计应使用独立输出或增加固定点评估能力。
2. **批大小测试：** 在完全相同的位置、同一 checkpoint 上，用两种评估 batch size，比较共同点处密度及 Hessian 对角线。差异应远小于待解释的起伏。
3. **精度测试：** 取数百个固定点，分别用 float32 和真正的 float64 网络计算对比。要在 JAX 初始化前启用 x64，并转换参数和输入；现有 eval 中的 float32 强制转换要在这个小型诊断里避开。只设置 `JAX_ENABLE_X64=1` 不等于现有流水线已经按 float64 计算。

保存 `Phi`、三维加速度、物理坐标 Hessian 对角线和密度。分别查看三个二阶导数是否很大但相互抵消；此时外围的小 Laplacian 容易受计算误差影响。

用几档物理步长计算加速度的中心差分散度，与自动微分 Laplacian 比较。寻找步长收敛平台，不把某一个很小步长的差分当真值。自动微分与差分一致只说明在求同一个网络的导数，不证明网络的物理正确性。

**本步交付：** 同点密度差异与波动幅度的比例，是否存在数值收敛平台。

**转向：** 若改变 batch/精度就足以改变波动，先修数值诊断再做新训练；若稳定，进入第 05 步。

### 05｜只测量背景和起伏怎样变化

**离线分析，不重新训练。**

先在第 01 步冻结的二维 mask 上做一张切片，确认方法后再扩展三维。定义：

`background_ell = smooth_ell(rho)`，`delta_rho_ell = rho - background_ell`。

平滑尺度用物理单位，不用固定像素数。不得将不支持区域补零后直接滤波，不跨过 mask 洞或截断边界混合密度。先使用归一化的带 mask 平滑，再将距无效区域/边界不足 `3*ell` 的点排除；旧/新用同一剩余区域。

下面接着第 03 步的变量执行，仅适用于等距 z=0 切片。`support` 必须来自第 01 步的实际数据支撑审计；示例不根据模型密度自动生成 mask。

```python
from scipy.ndimage import gaussian_filter, distance_transform_edt

x = old["slice_x"].astype(np.float64)
y = old["slice_y"].astype(np.float64)
dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
assert dx > 0 and dy > 0
np.testing.assert_allclose(np.diff(x), dx, rtol=1e-4, atol=1e-8)
np.testing.assert_allclose(np.diff(y), dy, rtol=1e-4, atol=1e-8)
support = np.load("填写第01步保存的共同支撑mask.npy").astype(bool)
assert support.shape == rho_old.shape == rho_new.shape
ell = 2.0  # 示例：必须改成第01步确定的物理尺度；不是通用的2 kpc处方
assert ell >= 2 * max(dx, dy), "先加密评估网格或选择可分辨的尺度"
sigma = (ell / dy, ell / dx)
normalization = gaussian_filter(support.astype(float), sigma, mode="constant", cval=0, truncate=3)
old_numerator = gaussian_filter(np.where(support, rho_old, 0), sigma, mode="constant", cval=0, truncate=3)
new_numerator = gaussian_filter(np.where(support, rho_new, 0), sigma, mode="constant", cval=0, truncate=3)
background_old = np.divide(old_numerator, normalization, out=np.full_like(rho_old, np.nan), where=normalization > 0)
new_background = np.divide(new_numerator, normalization, out=np.full_like(rho_new, np.nan), where=normalization > 0)
distance = distance_transform_edt(np.pad(support, 1, constant_values=False), sampling=(dy, dx))[1:-1, 1:-1]
interior = support & (distance > 3 * ell) & (normalization > 0.999)
assert interior.any(), "共同可信区域太窄：缩小已预定尺度或扩大有数据支持的评估区域"
delta_old = rho_old - background_old
delta_new = rho_new - background_new
rho_ref = np.sqrt(np.mean(background_old[interior]**2))
assert rho_ref > 0
print("A old/new:", np.sqrt(np.mean(delta_old[interior]**2)), np.sqrt(np.mean(delta_new[interior]**2)))
print("Q old/new:", np.sqrt(np.mean(delta_old[interior]**2)) / rho_ref, np.sqrt(np.mean(delta_new[interior]**2)) / rho_ref)
print("background change:", np.sqrt(np.mean((background_new[interior] - background_old[interior])**2)))
```

保存原密度、背景、起伏三个面板，分别在预定的三个尺度上重复。不同尺度若用于直接比较，用最大尺度共同有效的区域，或明确报告 mask 差异。二维结果称为“切片起伏”，不能直接称为三维体积起伏。

**判断：**

- 负值消失，背景上升，绝对起伏没有减少：支持“符号改善但起伏保留”。
- 背景稳定，起伏减少：支持有实质稳定性改善，还需检查加速度/真值。
- 起伏变小但背景严重变动：可能以有偏大尺度场换取平滑。
- 波峰位置、波长大幅变化：需要检查优化和约束冲突。

### 06｜只确定起伏出现在哪里

**离线分析；需要时补固定点评估。**

1. 在相同坐标上叠加起伏幅度、数据空间密度、惩罚点覆盖、距支撑边界距离。
2. 分内部/过渡/外围区计算第 02 步指标。外围不能只报一个平均数；报告误差最大的点及坐标。
3. 有不等权重时报告每区 `N_eff = (sum(w))**2 / sum(w**2)`，并注明独立性假设。它只是权重集中程度，不能反映潮汐流等相关粒子的全部信息损失。
4. 将 CBE residual 按其真实六维评估点所属空间区统计，与起伏区域对照。当前 residual 点是 DF support 中抽样，不是天然独立验证。
5. 对 Halo 用至少两个方位或切面复查，不把 x 轴波动直接解释为径向球壳波动。

**阶段 B 交付：** 一句话结论——“显示/数值问题”“主要是边界/稀疏区”“内部也普遍起伏”或“证据尚不足”。

**到这里先停一次，判断是否需要新训练。** 若显示或单位已经解释现象，不进入参数扫描。

## C. 分离权重与外围采样

### 07｜建立一个可复现的小型基准

**只运行一个短训练基准。现有入口。**

1. 固定同一个 DF checkpoint、support、Normalizer、坐标变换、Phi 网络、output scale、optimizer、batch size、L2、beta 和种子。
2. 优先从头训练，复制一个新的 YAML；若成本要求从旧模型起步，则所有分支从同一个预先指定的 Phi checkpoint 仅加载参数并统一重置优化器。
3. 预定短预算，例如正式总更新步数的 10%；由于当前 Phi 没有独立 `max_steps` 配置，先根据 support 大小与实际 batch 计算每 epoch 的更新数，再确定短实验 epochs。短实验的学习率曲线各分支一致。
4. 记录实际更新步数和时长。短实验只能筛掉明显失败设置，不能代表最终收敛结果。最终确认时必须重新跑完整、预定的 schedule。
5. 需要时间轨迹时，训练前设置适当的 `checkpoint_every` 和 `checkpoints_to_keep`，保留约 4–6 个预定时刻。不能等训练后才找已被清理的早期 checkpoint。

先确认训练终端所用后端，精确设备信息仅写本地记录：

```fish
conda activate dp-jax
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

如果计划跑 GPU 而输出为 CPU，先解决环境，不开始正式计时。真实多卡还要确认训练日志中的 sharding 与设备数，而不只看环境能发现几张卡。

新 run YAML 只把下面已有字段合并进原配置，不复制不完整片段作为完整配置：

```yaml
phi:
  # model、seed、df_run 等保留原来确认的值
  model_overrides:
    train:
      lambda_mass: 1.0  # 改成第00步确认的基准值
      mass_batch_size: 4096  # 改成基准值，且不大于训练batch
execution:
  resume: false
```

常规执行在用户训练终端进行，先将变量设为新建且检查过的配置路径：

```fish
conda activate dp-jax
set EXP_CONFIG configs/runs/填写本次新实验.yaml
python -m experiments.run_phi $EXP_CONFIG
```

确认上一条成功、checkpoint 保存完成后，再分别执行：

```fish
python -m experiments.eval_phi $EXP_CONFIG
```

```fish
python -m experiments.plot_phi $EXP_CONFIG
```

固定外部 DF 时使用这三个单阶段入口即可。不要用 `all` 触发重复 DF 训练。后台运行可用已有 `experiments.launch phi`，但 launcher 返回成功只表示启动；下一阶段要等训练日志确认完成。

**本步交付：** 一个可重现的 `C00` 基准，实际预算和独立评估表。GPU/真实多卡状态以终端验证为准。

### 08｜只改变负密度权重

**新训练 1 个 `C10`。**

固定所有项和质量约束点采样方式，只将 `lambda_mass` 从旧值换成第 00 步的新值。若不知道旧/新值，先做一个小跨度的候选，例如基准的 3 倍，而不是直接跨几个数量级。

检查 `Fneg`、负幅度、`A_ell`、`B_ell`、CBE 和加速度。不同 lambda 下总 loss 的定义变了，不能用总 loss 大小比较优劣；比较同定义的分项和共同评估指标。

**判断：** 非负性改善而 `A_ell` 基本不变，说明增权本身没有解决起伏。非负性改善但 CBE/加速度变差，进入第 12 步检查约束冲突。不要因此自动再加大 lambda。

### 09｜只改变外围采样

**新训练 1 个 `C01`。实现是否存在取决于第 00 步。**

如果原修改只增加 `mass_batch_size`，先把本步明确称为“惩罚点数量实验”，它并不保证外围覆盖增加。固定 lambda，增大有效点数，记录每步落在外围的数量。

如果原修改确实引入独立外围点采样器，使用同一个版本，只改变采样；不要同时改变 lambda 或 CBE 的采样。如果当前缺少这个实现，先补下述最小功能再训练：

- 一个确定种子的空间采样过程，明确在哪个受支持区域采样；均匀体积壳采样与均匀半径采样不同。
- 独立记录空间点、seed、区域、数量、是否每步重采样。
- 外围采样使用独立 RNG，不能因为多取了一批随机数而改变原来的 CBE batch 顺序。
- 这批点只用于非负约束；不要为其随意拼接速度后计算 CBE。
- 固定每步总质量点数，先改变“数据点/外围点”的比例，以分离覆盖与数量；随后才研究数量。
- 定义明确的平均方式。采用固定混合比例的一个 mean，或显式的两项系数；不能从一个 mean 改成两个 mean 相加却声称权重没变。
- 如果使用径向权重，需要说明独立点上的权重如何定义和归一化；不要沿用不对应这些位置的 batch 权重。
- 在保留当前默认行为的前提下补充配置和保存采样信息；新字段须被训练入口实际读取，不能只有 YAML。

先做 CPU 小数组验证，再做一个 GPU 小步 smoke，检查额外 Hessian 计算的显存和编译代价。

**判断：** 惩罚点的负值比例改善，独立空间点仍有负值，说明覆盖不足或点间过拟合。两个点集都非负但起伏仍在，说明仅靠非负性不足以约束所需尺度。

### 10｜组合权重和采样，完成四格对照

**仅在第 08/09 都明确后新增 `C11`，不要重复运行已完成组合。**

| 编号 | lambda | 质量约束采样 |
|---|---|---|
| C00 | 旧 | 旧 |
| C10 | 新 | 旧 |
| C01 | 旧 | 新 |
| C11 | 新 | 新 |

用 `C10-C00` 看权重作用，`C01-C00` 看采样作用。某个指标的 `C11-C10-C01+C00` 是交互效应的描述；一个种子的四格差异不构成统计显著性。

若只有 C11 起伏大，重点查交互和优化。若四组起伏位置类似，考虑上游 score、固定的网络/边界偏差。若新设置全面恶化，就保留旧基准，先诊断而不是继续扩容。

### 11｜只给最关键的两个设置换一个训练种子

**先新增 2 次训练；必要时扩到每设置 3 个种子。**

选 C00 和最有希望/最能解释问题的一个设置，先各增加一个 `phi.seed`。同一对使用配对种子。当前该 seed 同时改变初始化和 batch 顺序，因此结论是“Phi 训练随机性”，不是纯初始化效应。

如需严格只改初始化，另加独立 shuffle seed，保存初始参数；先用相同模型配置验证 batch 序列不变，再做实验。

**判断：** 两个种子结论相反时，不进行大规模下一阶段；增加第三个种子并报告分散程度。平均图更平滑不是精度证明。

### 12｜只看训练过程中的约束冲突

**优先复用保存的 checkpoint，不重训。需补指定 step 的评估。**

1. 在预定早、中、晚几个 checkpoint，用同一位置/速度点评估，不修改训练。
2. 检查负值从何时开始减少；`A_ell` 是此前已经存在，还是随后上升；加速度/CBE 是否同步恶化。
3. 若证据支持冲突，再在少量固定 batch 上分别求 `grad(L_CBE)`、`grad(lambda*L_negative)`、`grad(L2)`；报告范数及非零梯度间余弦。采几个独立 batch，避免一批异常点决定结论。
4. 负密度已经消失后质量项梯度为零是正常现象。梯度方向冲突是优化诊断，不足以证明某一项物理上错误。

当前常规 eval 只加载 latest。实现指定 step 时复用 `restore_step`，输出新文件名并记录真实 step；不要改 checkpoint 目录名或删除更新的 checkpoint 来强迫加载旧 step。

**阶段 C 交付：** 权重、覆盖、交互、训练时间和随机性分别贡献了什么。只选择一个基准进入 D。

## D. 判断误差来自 Phi 还是 DF

### 13｜固定 DF，检查起伏图案是否随 Phi 训练改变

**复用第 11 步结果即可。**

在同一坐标上比较背景与起伏的跨种子相关性。报告平均场、逐点标准差、波峰位置是否稳定。

- 背景稳定、起伏位置不稳定：当前约束没有稳定确定这些小尺度细节。
- 背景和起伏都不稳定：先查收敛、尺度和强弱约束。
- 起伏稳定：可能是 DF 偏差、结构性偏差或真实信号，不能直接宣布真实。

先对每个网络求导，再汇总密度/加速度。点对点中位数势可能不光滑，不能先拼一个中位数势再求导。算术平均势与平均密度在线性求导下相容；逐点中位数密度不一定对应同一中位数势。

### 14｜只测试优化是否充分

**仅当第 12/13 提示不收敛时做；每次新增 1 个设置。**

先固定总更新数和其他项，只将学习率整体降低，例如到原来的 1/3。若明显改善，再另做更长预算实验。不要同时改学习率、L2 和网络。

当前 cosine schedule 依赖 epochs。改变 epochs 会同时改变 schedule，因此“更长训练”实验必须记录完整曲线，不能把结果全部归因于更新数。若要求同一个前缀 schedule 的严格预算比较，先增加显式固定 schedule/步数控制并测试，不存在的 `max_steps` 字段不能直接填入 Phi YAML。

**判断：** 网络不变即可稳定改善，优先修优化。只降低训练 residual 却恶化独立物理指标，不是成功。

### 15｜在 Plummer 上只替换 score 来源

**最有价值的分层控制；现有 oracle 入口。先新训练一个 oracle。**

1. 找到 full-baseline 的真实 DF、Phi 配置，确认 `[128,128,128]` 等实际 overrides；不要只拿默认模型 YAML 对比。
2. oracle 使用同一 DF support/Normalizer、同一 Phi 配置与初始化种子、同一训练预算。唯一变化是 `score.source`。
3. 用固定相同的解析位置/速度点评估势、加速度、密度、score 和 CBE。
4. 先用少量点确认解析 score + 解析势的 residual 接近数值零，再扩大验证点数。

当前已提供 full oracle 配置，可在完成路径和配置核对后执行：

```fish
python -m experiments.run_phi configs/runs/plummer_full_oracle.yaml
```

然后单独执行 eval/plot。若该输出已存在，不覆盖；复制为新 run 配置。外部 DF 的真值验证不要直接运行只认 `experiment_dir/df` 的旧 analysis 常量脚本。应使用下面这个顺序调用，确保 `phi_df_dir` 正确：

```python
from experiments.workflows.config import load_run_spec
from experiments.validation.plummer import evaluate_plummer_truth

spec = load_run_spec("configs/runs/plummer_full_oracle.yaml")
result = evaluate_plummer_truth(spec.phi_df_dir, spec.phi_dir, spec.result_data_dir, n_eval=4096, batch_size=256, seed=7301, r_min=0.01, r_max=5.0, n_r=129, slice_grid=65, slice_rmax=5.0, artifact_prefix="validation_plummer_s15")
```

这里的区间只是一个 full Plummer 控制示例；baseline 用完全相同参数调用，cut 对照则另设共同的有效区间和前缀。固定 `seed` 只有在采样方式与区域相同时才保证同点。

四种组合分两次小操作完成：

| score | 势 | 诊断 |
|---|---|---|
| 解析 | 解析 | 方程、单位和链式法则 |
| 学习 | 解析 | 上游 score 已经使真实势产生多大 residual |
| 解析 | 学习 | 去掉 DF 误差后，Phi 是否能恢复导数 |
| 学习 | 学习 | 完整链条；低 residual 可能在补偿 DF 偏差 |

**转向：** oracle 明显改善 → 第 16 步。oracle 也失败 → 优先第 14、17 和 23 步。Plummer 成功只验证该控制问题，不证明 Halo12 平衡假设成立。

### 16｜只改变 DF 训练随机性

**成本较高；先新增 1 个 DF。**

固定原始数据、split、预处理、结构、训练预算，仅改变 DF 训练随机性；然后固定 Phi 训练设置，分别消费两个 DF。

注意当前 DF seed 也影响训练 jitter（若启用）。若要隔离模型随机性，需要冻结 jitter realization 或增加独立 jitter seed；否则诚实记录为“DF 整体训练随机性”。

在同一组物理六维点上对比两个 DF 的 score。分别比较空间/速度部分，避免混合不同量纲；有解析真值时报告绝对 score 误差，无真值时只能报告一致性。

**判断：** Phi 起伏随 DF 改变而系统改变，支持上游传播；多个 DF 的 score/密度一致仍不能排除共同偏差。若需要 ensemble，先在诊断层逐模型求导后汇总，不把不同 score 的逐点拼接自动当成某个一致 DF 的 score。

## E. 容量、样本量与空间分辨率

### 17｜先测试更小的 Phi，再决定是否扩大

**每次只训练 1 个模型。**

选择稳定 DF（Plummer 可用 oracle），先固定层数减半宽度。观察背景、加速度、`A_ell`、训练随机性。若起伏减少且物理精度保持，原模型自由度可能超过该数据/目标下稳定可约束的水平。

若小模型明显欠拟合，再测试基准宽度的更大版本。最后才固定宽度改变深度，不一次改两项。

记录实际参数量、收敛速度和总更新数。不同架构不可能共享同一参数树；相同 seed 也不等于相同初始函数。当前 L2 是所有参数的平均平方，同一 L2 系数在不同架构下不代表完全相同的函数先验，需在解释中说明。

**容量足够的实用标准：** 在预定尺度上，更大模型不再提供超过种子/数值波动的物理精度改善，且基准满足预定误差要求。训练 loss 平台本身不足以证明容量足够。

### 18｜只检查表示能力，不经过 DF/CBE

**仅在 oracle 仍失败且已检查优化时做。需补充 mock 监督实验。**

给同一个 Phi 网络解析 Plummer 势/加速度作为监督，先从小样本开始。损失量纲和权重预先固定，在独立位置上验证梯度和密度；只拟合势值成功不能证明二阶导数正确。

- 能监督恢复导数，但 CBE oracle 失败：优先查目标条件性、CBE 点覆盖和优化。
- 监督下也失败：再检查网络/尺度/优化；一次失败仍不能单独证明表达能力不足。

该实验是 mock 诊断，不把解析密度标签加入 Halo 主流程。

### 19｜只改变数据量，保持物理区域不变

**先做 N/2，再视结果做 N/4；每次一个规模。**

对同一选择范围内的原始独立数据做嵌套子集，保留固定、从未用于任何训练的最终测试样本。每个子集都有自己的 DF checkpoint、selection 和正常生成的 Normalizer。不同 Normalizer 属于端到端数据量实验的变化，若要进一步隔离其影响，再设计单独固定归一化实验。

分开两类问题：

- **端到端数据量：** 每个子集重训 DF 和 Phi，检查最终物理误差。
- **Phi 约束点数量：** 固定 DF，仅改变从同一 support 中取出的 CBE 训练点数量；当前入口需要增加显式子集控制。

当前 selection 会检查源数据 SHA-256。不能直接把较小 HDF5 填给原 DF 的 Phi 训练，或改 selection 文件绕过检查。每个新数据版本走完整对应 DF；Phi-only 子集实现则在已校验 support 后取子集并记录索引。

等 epoch 意味着更新数不同。主比较使用充分优化预算，并额外报告同更新数结果，区分数据收益和优化收益。重复/flow 生成点不算新增独立数据。

**判断：** 等 N 增加，误差和跨种子波动持续下降，支持有限样本瓶颈；波动下降但固定偏差保持，转查模型/物理/selection。

### 20｜只测可恢复的空间尺度

**复用前面模型，不重训。**

在多个固定物理尺度上绘制 `Q_ell`、有真值时的同尺度密度误差、跨种子差异。真值也施加相同平滑核/体积平均。保留未经平滑的原始结果。

寻找在哪个尺度以上结果对种子、采样、网络容量稳定。将它报告为该数据和方法的有效恢复尺度，而不是用评估网格像素间距充当分辨率。

如果目标是总质量而不是局部密度，再比较体积积分质量和闭合面引力通量：`M(V) = integral_V rho dV = integral_boundary grad(Phi) dot n dS / (4*pi*G)`。同一网络两种积分相符只验证数值自洽，不证明质量真值正确；非球对称情形不能仅用 x 轴加速度套球对称质量公式。

## F. 物理假设与有针对性的修正

### 21｜只检查 selection 和支撑边界

**先重用已有 full/cut 或 raw/clean 数据；不急着新增训练。**

按距边界距离而非仅按 r 检查 score/CBE/密度误差。先改变评估可信区，观察误差是否集中在边缘；这只是定位，不能通过缩小区域宣布全域成功。

若做 padding 对照，固定内部目标区与评估点，仅改变外围数据处理。新增数据仍须保持物理选择语义和来源索引。硬切内部选择函数恒定时，解析局部 score 不必失效；主要检查边缘及 flow 为逼近边缘产生的过渡。

不能将完全相同物理区域的随机下采样与径向 cut 混为“样本减少”实验。

### 22｜只检查非平衡与局部约束强弱

**先做局部诊断，有证据再扩大。需补充诊断计算。**

在少量可信位置，选择受到 DF 数据支持的不同速度点，构造 `A_i = grad_v log f`、`b_i = v_i dot grad_x log f`，用 SVD/QR 解局部 `A g ≈ b` 并检查奇异值。不要显式求病态法方程的逆，也不要把不同位置的 g 假定完全相同而不做邻域尺度测试。

若弱方向与密度起伏区域重合，支持局部动力学信息不足。局部引力解不保证全局可积，主要用于对照全局 Phi 网络。这一路径参考 [An et al. 关于 DF 恢复引力的研究](https://academic.oup.com/mnras/article/506/4/5721/6325178)。

如有模拟真实加速度，用同一学习 score 代入真实引力检查 residual；有多个相邻时间快照时，再检查时间变化。学习 score + 真实引力产生非零 residual，既可能来自 score，也可能来自非平衡，不能单独确诊。

不同 tracer 子群、方位或是否包含潮汐结构的比较，必须同时考虑各自 selection 和覆盖。只有静态假设失败的证据清楚时，才考虑旋转参考系/时间项等物理扩展。

### 23｜一次只尝试一种修正

**只有前面证据指向明确原因时执行。先在 mock 上验收。**

| 主要证据 | 首选修正方向 | 必须防止的代价 |
|---|---|---|
| 非负约束点覆盖不足 | 改质量点空间采样、独立点验证 | 只在训练惩罚点上非负 |
| 加权后 CBE/引力恶化 | 重新平衡项的尺度或采样比例 | 用巨大 lambda 压倒动力学 |
| Phi 随机性或短尺度自由度过大 | 先更小网络/更稳优化，再测试平滑先验 | 抹掉真实结构、背景偏移 |
| DF score 主导 | 改 DF 训练/正则/数据边界，再考虑 ensemble | 只改善 NLL、不改善 score |
| 有限数据主导 | 增加独立数据或降低目标空间分辨率 | 把生成样本当新增观测 |
| 稳定的 selection/非平衡偏差 | 处理 selection 或修正物理模型 | 用平滑掩盖系统偏差 |

若要增加函数空间平滑约束，先定义要抑制的尺度。例如固定物理间距的邻域密度差惩罚，可先在少量点上研究；直接惩罚 `grad(rho)` 涉及势三阶空间导数，训练代价和稳定性都需先做小型测试。

这些是新的科学先验，不是无代价的数值修复。用含已知小尺度结构的 mock 同时检查“去伪影”与“保信号”。单纯让密度输出恒正或事后滤波，不能证明势—密度恢复正确。

### 24｜独立确认，决定是否结束本轮

**只比较原基准与最终候选；预先固定完整预算。**

1. 用未参与参数选择的最终确认坐标/数据与新随机种子重复关键比较。
2. 使用预定完整训练预算；记录编译、训练、评估时间和设备。至少 3 个种子用于描述稳定性；若波动仍大或效果接近误差带，增加重复而不是强行宣布显著。
3. 同时检查负密度、起伏、背景、加速度、CBE、真值及可信区域。不给“更平滑”单独判胜。
4. 如果只有较粗尺度可信，明确报告该尺度和区域，不强求全域像素级恢复。
5. 保存基准、失败实验和候选，写下原因与尚未排除的问题。常规 CPU 检查与真实 GPU/多卡验证分别记录。

**结束条件：** 已定位主要误差来源，并有独立复现实验支持；或明确证明当前数据/假设只能支持某个恢复尺度。若仍无法区分，写出缺少哪种证据，停止无目的扩大参数扫描。

## 长期执行节奏

每次只完成一个步骤中的一个动作。例如第一轮只交第 00 步的配置表，第二轮只做共同网格图，第三轮只做背景分解。每轮结束都保存五行记录，并根据结果选择下一步。

建议的最短路径是：`00 → 01 → 02 → 03 → 04 → 05 → 06`。之后再决定是否执行 `07–12`，不要现在就预先启动所有训练。

如果 oracle 已证明 DF 主导，可以先跳到 16；如果精度测试发现数值问题，停在 04；如果同尺度真值已满足科学要求而更小尺度不稳定，可以直接进入 20 和 24。

## 实施准备与验证边界

本次只制定方案，没有运行新训练、改变 loss 或新增采样器。本文引用的现有字段已核对，尚未实现的能力已标记。

后续每次补充数值或训练能力，优先修改现有阶段入口、共享诊断和 artifact loader，保持线性研究脚本风格；不为每一步建立一个新框架或复制训练循环。独立真值/监督 mock 可放在 analysis。

涉及代码改动后，按影响运行针对性测试、Ruff、CPU pytest、compileall、CLI help、`uv lock --check` 和 `git diff --check`。涉及科研图要真实渲染检查。完整 GPU 训练与真实多卡测试在用户终端执行，CPU 检查不代替 GPU 验证。

## 与论文的关系

负密度约束与密度起伏是不同问题。Kalda & Green 的结果讨论了二阶导数放大、非平衡与真实结构都可能造成密度起伏，并使用多模型研究不确定性；这支持本方案的分层诊断，但不证明本项目的起伏来源已经确定。[Deep Potential, 2025，密度与不确定性部分](https://arxiv.org/html/2507.03742v2)

本方案中的具体实验编号、预算、采样对照、阈值和执行顺序是针对本仓库提出的研究设计，并非论文已经验证的通用处方。
