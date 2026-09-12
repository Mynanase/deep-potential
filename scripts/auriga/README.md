# Auriga Halo12：按上游结构迁移与运行

这套适配基于 `gregreen/deep-potential` 的 `19d01e9`，开发分支为
`codex/upstream-sync-2026-09-12`。旧实现保留在 `halo12-outer-clump-removal`。
迁移复用旧项目对 Halo12 数据、质量权重、坐标系和空间边界的经验；训练仍使用作者的
Equinox / Flow Matching / CBE 实现。没有移植旧 `dpjax`、YAML 调度器或审计框架。

训练核心仅修改 `scripts/potential.py` 三处：将当前 JAX 已移除的
`jnp.clip(..., a_min=0)` 改为等价的 `jnp.maximum(..., 0)`。
`fit_all.py`、flow 模型、采样和 CBE 公式均沿用上游。

## 1. 结构与科学含义

```text
Halo12 物理坐标导出文件
  → auriga/prepare_data.py
  → data/auriga/halo12.h5：eta、weights、ID、单位与范围
  → fit_all.py --flow-training
      → 空间 flow n(x)
      → 条件速度 flow p(v|x)
  → fit_all.py --flow-sampling
      → data/df_gradients.h5：eta、lnf、dlnf_deta、lnp、dlnp_deta
  → fit_all.py --potential-training
      → potential.PotentialModel：Phi(x)
  → auriga/plot_potential.py：物理单位下的轴向加速度、带符号密度
```

| 文件 | 职责 |
| --- | --- |
| `prepare_data.py` | 读取已有 Halo12 导出，选择范围、随机排列、质量权重、无量纲化、写上游 HDF5 |
| `options.json` | 全量基线起始参数；由作者 `fit_all.py` 直接读取 |
| `options-smoke.json` | 极小网络和样本，验证训练及模型保存／加载 |
| `requirements.txt` | 本次 CPU 验证使用的直接依赖与关键数值依赖版本 |
| `plot_potential.py` | 复用作者自动微分，恢复物理单位并保存 PNG 和 NPZ |

基线选择的是 **Halo12 星系中的全部恒星粒子**，不等于已经筛选出的“恒星晕”成员。
若课题要求只研究 halo tracer，需要另行定义成员选择；本次不会用模拟真势计算轨道分类
再把分类隐式混进训练。默认质量权重 `m / mean(m)`，得到归一化的质量加权示踪分布；
如果目标是粒子数密度，可显式用 `--weighting number`，两个目标不要混用。

已核对的本地文件：

| 输入 | 粒子数 | 含义 |
| --- | ---: | --- |
| `data/halo_12_stars.hdf5`（旧仓库） | 1,652,969 | 已居中、旋转，r < 75 kpc，PartType4 标量坐标 |
| `data/auriga/halo12_all_mass.h5`（旧仓库） | 1,652,969 | 相同粒子，旧版 eta 格式 |
| `data/auriga/halo12_all_mass_clean_outer_clump.h5`（旧仓库） | 1,643,286 | 已移除外侧三维团块中的 9,683 个粒子 |

旧文件里的 58,979 是 r ≥ 40 kpc 的团块搜索样本数，不是整个 Halo12 数据量。
转换器支持以上三种文件；不会再次自动居中、旋转、sigma clip、加噪或重做团块清理。
它要求明确的 kpc、km/s 元数据，不直接接收未经物理单位转换的 Gadget 快照。
模拟 `Potential`、加速度和总密度真值均不进入训练输出。

### 单位、DF 与 CBE

先显式定义 `q = x/L`、`p = v/V`，默认 `L = 10 kpc`、`V = 100 km/s`。
作者内部还会用训练集均值／标准差规范化 flow，但这层变换已包含在可微模型中，
保存的 `dlnf_deta` 是对输入 `(q,p)` 的导数，不是对网络内部标准化坐标的导数。

模型拟合 `F(q,p) = n(q) p(p|q)`，并使用完整 DF 的稳态方程：

```text
R = p · ∂q ln F − ∂q phi · ∂p ln F
Phi_phys = V² phi
acceleration_phys = −(V²/L) ∂q phi
rho_phys = V²/(4π G L²) ∇q² phi
```

这里 `G` 使用 kpc、km/s、Msun；`rho_phys` 是由势推得的总引力源密度，
并非恒星质量直方图。物理相空间概率密度为 `F/(L³V³)`，恒定归一化不影响 score。
无量纲 CBE 残差乘 `V/L` 才是物理时间单位下的残差。

默认固定非旋转参考系：所有 frame 参数为零且不训练；没有额外选择函数网络。
**不要添加 `--potential-ignore-nobs`**：该开关会丢掉空间密度梯度，
单独使用条件速度密度不能替代这里的完整稳态方程。

DF 用 r < 75 kpc 全部样本，Phi 只使用 1 ≤ r ≤ 70 kpc 的生成样本。
`eta.attrs` 的 `r_in/r_out` 指的是 Phi 区域，不是在加载时截断 DF。
1 kpc 的中心留边和 5 kpc 的外侧留边是初始选择，尚未证明足以排除边界偏差；
后续应检查改变留边后推断是否稳定。原数据在 75 kpc 已硬截断，无法凭空补出外侧信息。

去团块版可作为另一个数据分支使用，但不规则缺口也改变空间选择。
球形半径属性无法描述该缺口；不得把它直接视作选择函数已知、完整平衡的样本。
因此本次默认用未清理版打通流程。没有移植旧项目的硬裁剪、平滑密度惩罚或自定义 CBE 项。
负密度惩罚 `lambda_` 只抑制负 Laplacian，不能保证正密度场平滑。

## 2. 环境

在仓库根目录操作。本地根目录是：

```sh
cd /Users/qttao/Documents/Research/flow-diff/myanase-deep-potential
python3.11 -m venv .venv-halo
.venv-halo/bin/python -m pip install -r scripts/auriga/requirements.txt
.venv-halo/bin/python -m pip check
.venv-halo/bin/python -c 'import jax; print(jax.__version__, jax.devices())'
```

本机没有 `python3.11` 命令时，可用已安装的
`/opt/homebrew/Caskroom/miniforge/base/envs/dp-jax/bin/python` 创建环境。
本次已创建 `.venv-halo`，不必重建，也没有改动原来的 `dp-jax` 环境。
上游 requirements 漏列的 pandas、tqdm、cmasher、progressbar2 已补上。

Linux NVIDIA 服务器先在其仓库根目录执行上述环境安装，然后根据驱动选择 GPU wheel。
例如 CUDA 12 路线：

```sh
nvidia-smi
.venv-halo/bin/python -m pip install 'jax[cuda12]==0.10.2'
env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda .venv-halo/bin/python -c 'import jax; print(jax.devices())'
```

CUDA 12 需 Linux 驱动 ≥ 525；CUDA 13 需 ≥ 580 且受支持的 GPU，可把安装项换为
`jax[cuda13]==0.10.2`。以 [JAX 官方安装说明](https://docs.jax.dev/en/latest/installation.html)
及服务器实际设备为准。`JAX_PLATFORMS=cuda` 让 GPU 不可用时直接报错，避免悄悄退回 CPU。
这一迁移没有多卡训练实现；先用一张 GPU，显示多块设备不等于代码已并行。

以下命令使用 `env` 和明确 Python 路径，可直接在 bash、zsh 或 fish 运行。

### 传到服务器，无需 GitHub push 或 PR

当前修改尚未提交，分支也没有推到任何远程；直接在服务器 clone 旧 fork 不会得到这些适配。
本次已把当前工作区源码打包为 `runs/halo12-transfer/code.tar.gz`，不含旧框架残留、虚拟环境、
训练数据或模型。把这个包以及 `data/auriga/halo12.h5`、`data/auriga/halo12-smoke.h5`
传到服务器的一个新目录，然后：

```sh
mkdir halo12-upstream
tar -xzf code.tar.gz -C halo12-upstream
mkdir -p halo12-upstream/data/auriga
mv halo12.h5 halo12-smoke.h5 halo12-upstream/data/auriga/
cd halo12-upstream
```

在该目录创建服务器自己的环境，再按本文运行。包是源码快照，没有 `.git`，
不会修改服务器已有的旧项目，也不会对上游产生写入。
若后来修改本地源码，重新打包后再传输；不要复制本机 macOS 虚拟环境到 Linux。

## 3. 准备数据

当前机器的全量转换命令：

```sh
.venv-halo/bin/python scripts/auriga/prepare_data.py \
  --input /Users/qttao/Documents/Research/deep-potential/data/halo_12_stars.hdf5 \
  --output data/auriga/halo12.h5
```

该文件本次已生成，**可直接复用**。输出已存在时转换器报错，重做请给新文件名。
服务器可以只传输这个已转换 HDF5，不需要复制旧项目所有模块，也不需要模拟真势。
若要从旧清理版转换，只改输入并用另一个输出名，例如 `halo12-clean.h5`；
与未清理版使用不同 run 目录。`source_index`、`particle_id`、质量在选择／打乱时保持逐行一致。

为第一次冒烟运行另建 512 粒子文件：

```sh
.venv-halo/bin/python scripts/auriga/prepare_data.py \
  --input /Users/qttao/Documents/Research/deep-potential/data/halo_12_stars.hdf5 \
  --output data/auriga/halo12-smoke.h5 --max-particles 512 --seed 0
```

本机同样已生成；在服务器上若不传原始文件，直接传这份小文件即可。
不要把已经无量纲化的 `halo12.h5` 再传给转换器。
输入排列种子决定固定的训练／验证划分，不做质量重复采样，因此不会因重复粒子泄漏到验证集。

## 4. 小样本完整流程

以下用新目录 `runs/halo12-smoke-repeat`，避免覆盖本次已完成的 `runs/halo12-smoke`。
每一步成功后再执行下一步；这也是正式运行的三个独立阶段。

```sh
mkdir -p runs/halo12-smoke-repeat
cp scripts/auriga/options-smoke.json runs/halo12-smoke-repeat/options.json

env JAX_PLATFORMS=cpu .venv-halo/bin/python scripts/fit_all.py \
  --input data/auriga/halo12-smoke.h5 --run-dir runs/halo12-smoke-repeat --flow-training

env JAX_PLATFORMS=cpu .venv-halo/bin/python scripts/fit_all.py \
  --input data/auriga/halo12-smoke.h5 --run-dir runs/halo12-smoke-repeat --flow-sampling

env JAX_PLATFORMS=cpu .venv-halo/bin/python scripts/fit_all.py \
  --input data/auriga/halo12-smoke.h5 --run-dir runs/halo12-smoke-repeat --potential-training

.venv-halo/bin/python scripts/auriga/plot_potential.py \
  --input data/auriga/halo12-smoke.h5 \
  --potential-dir runs/halo12-smoke-repeat/models/Phi \
  --output-dir runs/halo12-smoke-repeat/plots
```

在服务器做同一检查时，把 `JAX_PLATFORMS=cpu` 换成
`CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda`。若 Matplotlib 缓存目录不可写，
在 `env` 后增加 `MPLCONFIGDIR=/tmp/halo12-mpl`。

预期得到：

```text
models/df/flow/flow_pos_only-0_model.eqx     空间阶段
models/df/flow/flow-1_model.eqx              两个 flow 完整模型
models/df/flow/*_loss.json 和 *_loss.pdf    两个 flow 的训练／验证损失
data/df_gradients.h5                       128 个 eta 和对应完整／条件 score
models/Phi/potential-0_model.eqx            势模型
models/Phi/*_loss.json 和 *_loss.pdf        势训练／验证损失
plots/potential_axes.png 与 .npz           带物理单位的三轴剖面
```

这些文件证明保存、重新加载和逐阶段执行可用。小样本势图只是数值流程示例，不能解释为
Halo12 科学结果。轴剖面也不能与球壳平均总密度直接比较。

## 5. 全量服务器运行

`options.json` 是可执行的基线起点，不是已优化好的科学配方。

| 参数 | 正式起点 | smoke |
| --- | --- | --- |
| 输入粒子 | 1,652,969 | 512 |
| 两个 flow | 各 MLP width=128, depth=3 | 各 width=16, depth=2 |
| Flow Matching | 随机配对，各 256 epochs，batch=4096 | 各 2 epochs，batch=64 |
| 生成 score 样本 | 262,144 | 128 |
| sample / grad batch | 1024 / 64 | 128 / 16 |
| Phi MLP | width=128, depth=3 | width=16, depth=2 |
| Phi 训练 | 256 epochs，batch=1024 | 2 epochs，batch=32 |
| 验证比例 | 25% | 25% |

这使用作者已有的普通 Flow Matching，不预计算 OT pairing 文件。
两者共享同一模型／损失接口；先减少数据配对缓存与额外超参数，再单独比较 OT 是否有收益。
`sb_constant`、`kinetic_reg`、`jacobian_reg` 初始均为 0；不会照搬旧 FFJORD 的正则系数。

先在服务器跑上一节 GPU smoke，再用正式宽度、短 epoch 在新 run 目录估算成本。
可以复制正式 options，把两个 flow 的 `n_epochs` 和 Phi 的 `n_epochs_noselfn` 改为 2，
把 `flow_sampling.n_samples` 改为 4096；保持正式宽度／batch，其余沿用。
从日志中训练 epoch、采样 batch 的实测速度决定正式资源预算，不能把笔记本小模型时间外推。

正式运行命令（服务器仓库根目录；确认 `data/auriga/halo12.h5` 已传入）：

```sh
mkdir -p runs/halo12-baseline
cp scripts/auriga/options.json runs/halo12-baseline/options.json

env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-halo/bin/python -u scripts/fit_all.py \
  --input data/auriga/halo12.h5 --run-dir runs/halo12-baseline --flow-training \
  > runs/halo12-baseline/flow.log 2>&1

env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-halo/bin/python -u scripts/fit_all.py \
  --input data/auriga/halo12.h5 --run-dir runs/halo12-baseline --flow-sampling \
  > runs/halo12-baseline/sampling.log 2>&1

env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-halo/bin/python -u scripts/fit_all.py \
  --input data/auriga/halo12.h5 --run-dir runs/halo12-baseline --potential-training \
  > runs/halo12-baseline/potential.log 2>&1

.venv-halo/bin/python scripts/auriga/plot_potential.py \
  --input data/auriga/halo12.h5 --potential-dir runs/halo12-baseline/models/Phi \
  --output-dir runs/halo12-baseline/plots
```

可另用作者的 `--basic-flow-benchmarking --basic-flow-benchmarking-n-samples 4096`
检查 flow 的投影／速度分布，其坐标是训练无量纲单位；密度图看起来相似并不能证明 score 准确。
不要对无量纲 Halo 输入加 `--basic-potential-benchmarking-gaia-units`，那不是本适配的单位体系。

三个训练命令都应检查退出状态和日志后再继续。服务器长任务请在已有 tmux／作业调度器中运行。
梯度计算内存不足时优先减 `flow_sampling.grad_batch_size`；采样不足则检查生成分布与
Phi 球形区域的交集，不要只靠增加抽样次数。Flow / Phi 的 batch 不得大于各自训练集大小。

作者入口每次 `--flow-training` / `--potential-training` 都会重新构造模型。
**本迁移不承诺中途精确续训**；已有 checkpoint 可供独立采样、画图，但重复训练命令
不是恢复 optimizer 状态。重新训练用新目录，避免混入旧 checkpoint 或旧梯度文件。
固定实验至少保留 input HDF5、run/options.json、源码版本与模型文件；无需另建数据库。

## 6. 如何从“可运行”走到可信科学结果

1. 先固定未清理、质量加权、完整 DF 的上述基线，确认 held-out flow 分布及 score 行为。
2. 在同一组相空间点检查 score、CBE 残差及对求解容差的敏感性，不能只看 Flow Matching loss。
3. 检查势梯度与 Laplacian；保留负密度和局部波动，不通过截零把问题隐藏掉。
4. 单独比较边界留边、团块清理或 halo 成员选择；每次改变都会改变示踪分布，避免一起改。
5. 最后才使用模拟加速度或总物质球壳密度独立验证。星粒子 `Masses` 的径向直方图不是总密度真值。

本次完成的是结构迁移和可执行性验证。稳态假设、非平衡亚结构、全恒星与恒星晕目标差异，
仍可能主导科学误差；重新基于上游开发本身不能消除这些问题。

## 7. 本次实际验证结果（2026-09-12 至 13）

环境：Apple Silicon CPU、Python 3.11.15、JAX/JAXlib 0.10.2、Equinox 0.13.8、
Diffrax 0.7.2、FlowJAX 19.1.1、Optax 0.2.8；详细版本在本目录 requirements。

| 检查 | 结果与范围 |
| --- | --- |
| 全量输入转换 | 1,652,969 粒子写入 `data/auriga/halo12.h5`，磁盘约 70 MiB |
| 原始导出 vs 旧 all_mass | 同种子 512 粒子转换后 eta、weights、ID、source_index 逐项完全一致 |
| 旧清理版 | 512 粒子抽查，ID／非连续 source_index、质量权重和物理坐标还原一致 |
| smoke | 512 输入、两个 16 宽度 flow 各 2 epochs、128 梯度、16 宽度 Phi 2 epochs；三个 CLI 阶段均成功 |
| 正式网络短跑 | 8,192 输入，保持正式 128 宽度、depth=3、Flow batch=4096、Phi batch=1024，训练 2 epochs，生成 2,048 梯度，全部成功 |
| 模型与损失 | 两组模型从磁盘重新加载，所有保存数组及 loss 有限，frame 参数保持固定为零 |
| score 差分 | 每组 4 个点、6 个坐标方向，步长 0.002；最大绝对差分别 2.47e-4、2.99e-4 |
| 条件 vs 完整 DF | 相同点的速度 score 一致；位置 score 不应直接互换 |
| 解析谐振子 | 梯度符号、Laplacian、L/V 换算与平衡 CBE 残差检查通过 |
| 物理单位绘图 | 已加载 smoke 势，生成带符号密度与加速度 PNG／NPZ，并检查图像 |

正式网络短跑仅把正式 options 的两个 flow epoch、Phi epoch 改为 2，
把梯度样本数改为 2048，其余相同。其配置保存在 `runs/halo12-pilot/options.json`。
数值记录为 `runs/halo12-smoke/numerical_checks.json`，日志在两个 run 目录；
快照已纳入版本控制：`scripts/auriga/numerical_checks_20260913.json`。

作者日志内计时（含该阶段编译与保存，但不含进程启动）：smoke flow 2.78 s、采样及
score 5.39 s、Phi 1.74 s；正式宽度短跑分别 3.26 s、7.64 s、1.91 s。
短跑每个 epoch 只有少数更新，不可据此声称全量 256 epochs 已完成或已有可靠 GPU ETA。

特别注意：短跑生成样本的最大半径分别约 30.17、34.87 kpc，**没有验证 35–70 kpc 外晕覆盖**。
绘图脚本可画到指定范围，但超出实际样本支持的部分仍是外推。
有限差分检查验证的是计算导数的实现，不是这些导数对真实 Halo12 DF 的科学准确性。
尚未运行全量 GPU 训练、收敛评估或模拟真值比较。
