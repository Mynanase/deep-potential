# Deep Potential 操作与维护指南

本文是当前 JAX/Flax 实现的日常操作手册与长期维护约定。命令默认从仓库根目录
执行，适用于 `dpjax` 当前检查点格式和坐标变换 schema v2。

算法原理见 `PROJECT_STRUCTURE.md`，更完整的实验参数说明见 `README_JAX.md`。
历史 TensorFlow/Sonnet 代码只保存在 `archive/legacy_tensorflow/`，不属于本文
覆盖的运行环境。

## 1. 当前支持边界

当前正式支持的流程是：

1. 生成或载入 `(N, 6)` 相空间数据。
2. 训练 DF：FFJORD 为主，RealNVP 只用于兼容旧检查点。
3. 评估 DF 的密度、采样和 score。
4. 冻结 DF，训练势能网络 Φ。
5. 评估 CBE 残差、势能、加速度和密度。
6. 按需进行 DF + Φ 联合微调。

相空间列顺序固定为：

```text
[x, y, z, vx, vy, vz]
```

非线性坐标变换 `asinh`、`log` 和 `power` 当前只支持 DF 训练与 DF 评估。
在完整物理梯度链式法则实现前，Φ/CBE 训练、评估和势场绘图会拒绝使用带有
非线性变换的 DF 检查点。

因此：

- Plummer 两阶段流程可直接使用当前无非线性变换的配置。
- Halo12 v23 使用质量加权 NLL 和 `transform: none`，是当前 DF 与后续
  Φ/CBE 实验的首选起点。
- Halo12 v21 使用 `power` 变换，目前只能进行 DF 实验，不能直接进入 Φ/CBE。
- 当前 Halo12 DF 阶段的服务器命令和验收线见
  `docs/auriga_halo12_df_stage.md`。
- 无 `schema_version=2` 的历史 `coord_transform.npz` 会被警告并按 no-op 处理；
  需要验证真实坐标变换效果时必须重新训练。

## 2. 目录职责

| 目录 | 职责 | 维护规则 |
|---|---|---|
| `dpjax/` | 数据、模型、物理、绘图和检查点核心库 | 只放可复用、可测试的实现 |
| `experiments/` | CLI 参数解析与实验流程编排 | 保持轻量，不复制核心算法 |
| `configs/` | 可复现实验的 YAML 配置 | 已运行配置保持不可变，新实验新建文件 |
| `jobs/` | 无调度器 Bash 服务器脚本 | 不硬编码服务器路径、GPU 编号或用户名 |
| `scripts/` | 专用核对和离线诊断 | 不新增训练主入口 |
| `tests/` | 快速单元与回归测试 | 每次修复都应补对应测试 |
| `notebooks/` | 交互式结果分析 | 不承载唯一实现，不保存大体积输出 |
| `archive/` | 只读历史代码 | 活跃代码不得从这里导入 |
| `data/` | 本地数据 | 不提交 Git，另行备份和记录校验值 |
| `runs/` | 配置快照、日志和检查点 | 不提交 Git，重要实验需外部归档 |

判断代码应放在哪里时，使用以下规则：

- 能被两个入口复用，或可以独立测试：放入 `dpjax/`。
- 只负责把配置、文件路径和核心函数串起来：放入 `experiments/`。
- 只用于一次数据核对且没有复用价值：放入 `scripts/`。
- 不再受支持但仍需保留历史：放入 `archive/`。

## 3. 环境初始化与更新

### 3.1 首次创建

```bash
conda env create -f environment.yml
conda activate dp-jax
env UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" \
  uv sync --extra dev --extra tracking --locked
```

显式设置 `UV_PROJECT_ENVIRONMENT` 是为了把锁定依赖安装到当前 Conda 环境，
避免 `uv` 另外创建 `.venv`。此写法可直接用于 fish 和 bash。

检查环境：

```bash
python --version
python -m pip check
uv lock --check
```

### 3.2 拉取代码后的常规更新

当 `environment.yml`、`pyproject.toml` 或 `uv.lock` 发生变化时：

```bash
conda activate dp-jax
conda env update -n dp-jax -f environment.yml --prune
env UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" \
  uv sync --extra dev --extra tracking --locked
python -m pip check
python -m pytest
```

不要只在本机执行临时 `pip install` 后提交代码。新增或删除运行依赖时，必须同步
修改 `pyproject.toml` 并更新 `uv.lock`。

### 3.3 CPU/GPU 检查

```bash
python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY
```

共享 GPU 上出现初始化 OOM 时：

```bash
set -x XLA_PYTHON_CLIENT_PREALLOCATE false
```

仅做 CPU smoke 时：

```bash
env JAX_PLATFORM_NAME=cpu python -m experiments.smoke_dpjax
```

Linux/CUDA 服务器上的 `jax` 和 `jaxlib` 必须与服务器 CUDA 版本匹配。升级它们
后至少重新执行 CPU smoke、单 GPU smoke 和一次小批量训练。

## 4. 数据操作

### 4.1 生成 Plummer 数据

不划分测试集：

```bash
mkdir -p data
python -m experiments.gendata_plummer \
  --total-n 131072 \
  --seed 42 \
  --train-out data/plummer_n131072.h5
```

生成固定训练/测试拆分：

```bash
python -m experiments.gendata_plummer \
  --total-n 131072 \
  --test-frac 0.1 \
  --seed 42 \
  --train-out data/plummer_train.h5 \
  --test-out data/plummer_test.h5
```

### 4.2 检查任意 HDF5 数据

```bash
python -m experiments.inspect_data \
  --data data/plummer_n131072.h5
```

导入新数据前至少确认：

- `eta` 存在且 shape 为 `(N, 6)`。
- 六列顺序和单位明确。
- 没有 NaN 或 Inf。
- 位置和速度尺度合理。
- 数据过滤、抽样和随机种子可以复现。

业务代码统一使用 `dpjax.data.load_eta_h5` 和 `save_eta_h5`。不要在各实验入口中
重复编写 HDF5 校验逻辑。

### 4.3 准备 Auriga mock 与模拟真值

先把 Auriga/Gadget 星粒子文件转换成项目统一 schema。推荐在转换时写清单位、
中心位置和系统速度；以下中心数值只是命令格式示例，必须替换成 snapshot 的真实
参考系参数：

```bash
python -m experiments.prepare_auriga \
  --input /path/to/halo_12_stars.hdf5 \
  --group PartType4 \
  --output data/auriga/halo12_static_mock.h5 \
  --position-center 0 0 0 \
  --velocity-center 0 0 0 \
  --r-max 75 \
  --length-unit kpc \
  --velocity-unit km/s \
  --potential-unit '(km/s)^2' \
  --acceleration-unit '(km/s)^2/kpc'
```

若已有历史 `eta` 训练文件，需要把原始 snapshot 中的 `Potential`、`Masses`
等真值重排到同一行序：

```bash
python -m experiments.prepare_auriga \
  --input /path/to/halo_12_stars.hdf5 \
  --align-to data/halo_12_train.h5 \
  --atol 1e-4 1e-4 1e-4 1e-2 1e-2 1e-2 \
  --output data/auriga/halo12_train_aligned.h5 \
  --length-unit kpc \
  --velocity-unit km/s
```

如果两边都有 `particle_id`/`ParticleIDs`，程序自动按 ID 对齐并忽略 `--atol`。
没有 ID 时必须显式给出位置和速度各自的容差；任何缺失、重复或多重匹配都会报错，
不会静默取第一个粒子。输出的 `source_index` 用于回溯原始 snapshot 行号。

导入后运行：

```bash
python -m experiments.inspect_data \
  --data data/auriga/halo12_static_mock.h5
```

除通用 shape/NaN 检查外，还应使用 HDF5 工具确认：

- `schema` 为 `dpjax.auriga.mock.v1`；
- `length_unit`、`velocity_unit`、`potential_unit` 和
  `acceleration_unit` 与训练约定一致；
- `potential`/`acceleration` 至少存在一个；
- `source_index` 和 `particle_id` 能追溯到原始 snapshot。

## 5. 标准实验流程

建议运行目录使用：

```text
runs/<system>/<stage>_<backend>_<tag>/
```

例如：

```text
runs/plummer/df_ffjord_seed0/
runs/plummer/phi_ffjord_seed0/
runs/halo_12/df_ffjord_v22/
```

已完成的运行目录不应被新实验覆盖。调参时创建新目录，只有确实恢复同一次中断
任务时才使用 `--resume`。

### 5.1 训练 DF

```bash
python -m experiments.train_df \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_n131072.h5 \
  --run-dir runs/plummer/df_ffjord_seed0 \
  --logger csv
```

临时 smoke 可以使用 JSON override，而正式实验应把最终参数保存为独立 YAML：

```bash
python -m experiments.train_df \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_n131072.h5 \
  --run-dir runs/smoke/df_ffjord \
  --override '{"train": {"epochs": 1, "max_batches_per_epoch": 2}}'
```

恢复中断训练时，必须使用同一份配置、数据和运行目录：

```bash
python -m experiments.train_df \
  --config configs/df_plummer_ffjord.yaml \
  --data data/plummer_n131072.h5 \
  --run-dir runs/plummer/df_ffjord_seed0 \
  --resume
```

不要用 `--resume` 顺便修改模型结构、优化器或数据预处理；这种情况应建立新实验。

### 5.2 评估 DF

```bash
python -m experiments.eval_df \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --out-dir runs/plummer/df_ffjord_seed0/eval \
  --n-samples 32768 \
  --plummer-diag
```

进入 Φ 阶段前检查：

- 验证 NLL 没有持续发散。
- `score_p99` 和 `score_max_abs` 没有失控增长。
- 采样边际分布与数据一致。
- Plummer 实验中的 score 斜率接近 1，残差无明显系统偏差。

DF 的密度拟合正常不代表其梯度一定适合 CBE；score 诊断是两阶段之间的强制
验收项。

### 5.3 训练 Φ

```bash
python -m experiments.train_phi \
  --config configs/phi_plummer.yaml \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --run-dir runs/plummer/phi_ffjord_seed0 \
  --logger csv
```

恢复同一次 Φ 训练：

```bash
python -m experiments.train_phi \
  --config configs/phi_plummer.yaml \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --run-dir runs/plummer/phi_ffjord_seed0 \
  --resume
```

基于旧 Φ 权重、但使用新优化器或新损失启动新实验：

```bash
python -m experiments.train_phi \
  --config configs/phi_plummer.yaml \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --run-dir runs/plummer/phi_finetune \
  --init-params runs/plummer/phi_ffjord_seed0
```

`--resume` 与 `--init-params` 不能同时使用。

Halo12 静态势的首个可复现实验使用：

```bash
python -m experiments.train_phi \
  --config configs/phi_halo12_static_v1.yaml \
  --data data/auriga/halo12_static_mock.h5 \
  --df-run-dir runs/halo_12/df_ffjord_v22 \
  --run-dir runs/halo_12/phi_static_v1 \
  --logger csv
```

`df_ffjord_v21` 使用非线性 `power` 变换，当前仍会被 Φ/CBE 门禁拒绝。不要绕过
门禁；应先验证 v22 无变换基线，或完整实现物理 score、势梯度和 Laplacian 的
变换链式法则后再重训。

### 5.4 评估 Φ 和 CBE

```bash
python -m experiments.eval_phi \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --phi-run-dir runs/plummer/phi_ffjord_seed0 \
  --out-dir runs/plummer/phi_ffjord_seed0/eval \
  --system plummer
```

单独生成训练曲线和势场切片：

```bash
python -m experiments.plot_training \
  --run-dir runs/plummer/phi_ffjord_seed0

python -m experiments.plot_phi_slice \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --phi-run-dir runs/plummer/phi_ffjord_seed0
```

Auriga 实验必须再对模拟真值做量化评估：

```bash
python -m experiments.eval_auriga_truth \
  --data data/auriga/halo12_static_mock.h5 \
  --df-run-dir runs/halo_12/df_ffjord_v22 \
  --phi-run-dir runs/halo_12/phi_static_v1 \
  --out-dir runs/halo_12/phi_static_v1/eval/auriga_truth \
  --n-eval 65536 \
  --radial-bins 12
```

输出包括：

- `auriga_truth_metrics.json`：势的加法常数对齐后误差、三维加速度相对误差、
  方向余弦以及径向分箱误差；
- `auriga_truth_predictions.npz`：逐粒子的预测、真值、位置、标准化文件行号和
  原始 `source_index`。

若 snapshot 真值单位需要换算，可使用 `--truth-potential-scale` 和
`--truth-acceleration-scale`。换算因子必须记录到实验说明；禁止为了获得更好
误差而事后拟合乘法尺度。势能只允许拟合物理上不可观测的加法常数。

建议第一阶段验收门槛（之后根据等价静态 mock 校准）：

- potential normalized RMSE `< 0.10`；
- acceleration median relative error `< 0.10`；
- acceleration p90 relative error `< 0.25`；
- median cosine similarity `> 0.98`；
- 误差随半径、子结构掩膜和 seed 不出现未解释的系统漂移。

主要验收内容：

- CBE residual 的均值接近 0，标准差和高分位数随训练下降。
- Plummer 势能和径向加速度趋势与解析解一致。
- `rho = ∇²Φ/(4π)` 没有大面积负值或数值爆炸。
- 不只看训练 loss，还要检查空间切片和径向曲线。

### 5.5 联合微调

只有 DF 和 Φ 分别通过评估后再进行联合微调：

```bash
python -m experiments.finetune_joint \
  --config configs/joint_plummer.yaml \
  --data data/plummer_n131072.h5 \
  --df-run-dir runs/plummer/df_ffjord_seed0 \
  --phi-run-dir runs/plummer/phi_ffjord_seed0 \
  --run-dir runs/plummer/joint_seed0
```

输出的可独立加载模型位于：

```text
runs/plummer/joint_seed0/df/
runs/plummer/joint_seed0/phi/
```

联合训练涉及更高阶梯度，优先使用较小学习率和 `mode: alt`。如果结果恶化，应
回到 DF score 和 Φ residual 的单独诊断，不要只调整损失权重。

## 6. 日志、检查点与实验归档

典型 DF 运行目录包含：

```text
config.yaml
normalizer.npz
coord_transform.npz  # 仅使用非线性变换时存在
ckpt/
metrics.csv
logger_metrics.csv
tb_logs/              # 使用 TensorBoard 时存在
```

其中：

- `config.yaml` 是训练实际使用的合并后配置。
- `normalizer.npz` 和可选的 `coord_transform.npz` 是模型输入契约的一部分。
- `ckpt/` 保存模型、优化器状态和 step。
- `metrics.csv` 是训练器的详细指标。
- `logger_metrics.csv` 是统一 logger 的外部追踪副本。

日志后端：

```bash
--logger csv
--logger wandb
--logger tensorboard
--logger wandb+tb
```

无网络环境下使用 W&B：

```bash
env WANDB_MODE=offline python -m experiments.train_df ...
```

重要实验完成后，应在外部存储归档以下内容：

- 完整运行目录。
- 输入数据的路径、版本和 SHA-256。
- 当前 Git commit。
- `uv.lock`。
- GPU 型号、JAX/JAXLIB 版本和随机种子。
- 一份最终评估结果和关键图表。

`data/`、`runs/`、图片和检查点默认不进入 Git，因此不能把 Git 仓库当作实验
备份。

## 7. 普通 Linux 服务器操作

Halo12 使用普通 Bash 脚本，不依赖集群调度器。单次训练示例：

```bash
env \
  CONFIG=configs/df_halo12_ffjord_v23_mass.yaml \
  DATA_PATH=data/auriga/halo12_all_mass.h5 \
  RUN_DIR=runs/halo_12/df_ffjord_v23_mass/seed_42 \
  GPU_DEVICES=0,1 \
  bash jobs/train_halo12_df.sh
```

可选环境变量：

- `DEEP_POTENTIAL_ROOT`：仓库绝对路径；默认根据脚本位置自动定位。
- `DATA_PATH`：输入 HDF5；默认 `data/halo_12_train.h5`。
- `CONDA_ENV_NAME`：Conda 环境名；默认 `dp-jax`。
- `GPU_DEVICES`：可见 GPU，例如 `0` 或 `0,1`；默认使用全部可见设备。
- `LOGGER`：日志后端；默认 `csv`。
- `WANDB_PROJECT`：W&B 项目名。
- `SEEDS`：ensemble seed 列表，默认 `42,43,44,45`。
- `RESUME=1`：从已有 checkpoint 恢复训练。

后台运行完整 prepare → train → eval 流程：

```bash
mkdir -p logs
nohup env \
  INPUT_PATH=/path/to/halo_12_stars.hdf5 \
  GPU_DEVICES=0,1 \
  LOGGER=csv \
  bash jobs/run_halo12_df_pipeline.sh \
  > logs/halo12-pipeline.log 2>&1 &
```

使用 fish 时可用 `echo $last_pid` 查看刚启动的后台进程，并用
`tail -f logs/halo12-pipeline.log` 跟踪日志。

维护服务器脚本时：

- GPU 通过 `GPU_DEVICES` 传入，不在脚本中固定设备编号。
- 不写个人 home、Conda 安装路径或服务器专用仓库路径。
- 实验差异通过 `CONFIG`、`RUN_DIR` 和环境变量传入。
- ensemble 默认顺序运行，避免多个 FFJORD 进程争用同一 GPU 显存。

多 GPU 训练会根据 `jax.local_device_count()` 启用分片。全局 batch size 最好能
被设备数整除；否则训练器会向下调整并打印实际 batch size。

## 8. 故障排查

### 8.1 环境解析到了 `.venv`

确认同步命令包含：

```bash
env UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked
```

并确认 `conda activate dp-jax` 已执行。

### 8.2 JAX 没有看到 GPU

依次检查：

```bash
nvidia-smi
python -c "import jax; print(jax.default_backend(), jax.devices())"
python -c "import jaxlib; print(jaxlib.__version__)"
```

通常原因是 `jaxlib` 与 CUDA/驱动不匹配，而不是训练代码。

### 8.3 GPU OOM

按以下顺序处理：

1. 设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
2. 减小 batch size。
3. 降低 FFJORD 隐层宽度、block 数或 ODE 最大步数。
4. 先在 CPU 或单 GPU 上运行 smoke。
5. 检查是否同时存在其他 JAX 进程。

### 8.4 FFJORD 出现 NaN、ODE 步数耗尽或 score 爆炸

检查：

- 输入数据是否含 NaN/Inf 或异常尺度。
- `dt0`、`rtol`、`atol` 和 `max_steps`。
- 学习率与 warmup。
- `kin_reg`、`jac_reg` 和梯度裁剪。
- `score_p99`、`score_max_abs` 是否在更早阶段已经恶化。

不要单纯提高 `max_steps` 掩盖发散。先用小数据、小 epoch 和诊断指标定位是数据、
优化器还是 ODE 动力学问题。

### 8.5 非线性变换被 Φ/CBE 拒绝

这是预期的安全检查。当前选择：

- 使用 `transform: none` 重新训练 DF；或
- 只对该 DF 做密度与 score 评估。

不要删除安全检查或手动伪造 `coord_transform.npz`。正确修复方式是为 CBE
实现坐标变换的一阶、二阶链式法则，并增加解析回归测试。

### 8.6 旧检查点无法恢复

先确认运行目录包含匹配的：

```text
config.yaml
normalizer.npz
ckpt/
```

如果模型结构、优化器树或预处理 schema 已变化，优先保留旧环境用于复现，并在
新运行目录执行显式迁移。不要直接覆盖原检查点。

## 9. 每次修改后的质量门禁

### 9.1 所有改动

```bash
conda activate dp-jax
python -m pytest
env JAX_PLATFORM_NAME=cpu python -m experiments.smoke_dpjax
python -m pip check
uv lock --check
git diff --check
```

### 9.2 数据与预处理改动

额外要求：

- HDF5 读写 round-trip。
- forward/inverse transform round-trip。
- 输入数组不被意外原地修改。
- 固定随机种子下采样结果可复现。
- 旧 schema 的兼容行为有测试。

### 9.3 DF 或 ODE 改动

额外要求：

- FFJORD 和 RealNVP API smoke。
- `log_prob`、`score` 和 `sample` shape/finite 检查。
- 小数据一轮训练、保存、重载和采样闭环。
- 对比修改前后的 NLL、score 分位数和 ODE 失败率。

### 9.4 Φ/CBE 改动

额外要求：

- Plummer 解析势、加速度和密度回归。
- 标准化到物理单位的梯度换算测试。
- 一阶和二阶导数 finite 检查。
- CBE residual 的尺度与符号检查。
- 单 GPU 与目标多 GPU 配置各做一次小规模运行。

### 9.5 CLI、配置或服务器脚本改动

额外要求：

```bash
python -m experiments.<module> --help
bash -n jobs/*.sh
```

并确认 README 或本指南中的命令仍然有效。

## 10. 如何扩展而不重新引入冗余

### 10.1 新增数据集

1. 在 `dpjax/datasets/` 添加纯函数采样或转换实现。
2. 通过显式 `numpy.random.Generator` 注入随机性。
3. 在 `experiments/` 添加只负责参数解析和保存的薄 CLI。
4. 复用 `load_eta_h5` / `save_eta_h5`。
5. 添加形状、范围、可复现性和统计性质测试。
6. 增加最小配置和操作示例。

不要把数据生成核心放回 Notebook 或不可导入脚本。

### 10.2 新增 DF 后端

1. 在 `dpjax/flows/` 实现模型。
2. 通过 `dpjax.flows.api` registry 接入统一接口。
3. 支持 `build/init/log_prob/score/sample` 契约。
4. 明确定义配置 schema 和检查点加载方式。
5. 扩展 smoke 与端到端保存/重载测试。
6. 只在统一 API 中分派，避免下游脚本判断具体模型类型。

### 10.3 新增坐标变换

1. 在 `CoordinateTransform` 中同时实现 forward 和 inverse。
2. 对所有选择维度添加 round-trip 测试。
3. 确认 NumPy 高级索引没有导致静默 copy。
4. 变更持久化语义时提升 schema version。
5. 在进入 Φ/CBE 前实现物理梯度和 Hessian 的完整链式法则。
6. 用解析分布验证 score、CBE 和密度，而不只验证数组可逆。

### 10.4 新增实验入口

优先为已有核心函数添加参数，不要复制完整训练脚本。公共参数放入
`experiments/_cli.py`；可复用实现放入 `dpjax/`；CLI 自身应能通过
`python -m experiments.<name> --help` 运行。

### 10.5 修改检查点格式

- 新格式必须有版本号。
- 保留显式迁移函数或清晰的拒绝信息。
- 不把错误的历史语义解释成新语义。
- 添加旧版加载和新版 round-trip 测试。
- 在文档中标明哪些实验需要重新训练。

## 11. 依赖、配置和文档维护

### 11.1 依赖

运行依赖写入 `[project.dependencies]`；Notebook、追踪和开发工具分别写入对应
optional dependency。修改后：

```bash
uv lock
env UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" \
  uv sync --extra dev --extra tracking --locked
python -m pip check
python -m pytest
```

避免在 `environment.yml` 和 `pyproject.toml` 中维护两套详细版本表。
`environment.yml` 只负责 Python、pip 和 uv，Python 包版本由 `uv.lock` 固定。

### 11.2 配置

- 已产生结果的 YAML 不再原地修改。
- 新实验从最接近的配置复制并使用新名称。
- 配置注释只解释当前参数，不保存冗长实验史。
- 实验史放到独立训练日志。
- 高频临时参数可用 `--override`，最终复现实验必须固化为 YAML。

### 11.3 文档

- 操作命令变化：更新本文和相关 README。
- 算法或目录职责变化：更新 `PROJECT_STRUCTURE.md`。
- 历史实验结论：更新独立训练日志，不塞入主操作指南。
- 删除入口时，全仓搜索文件名、旧模块路径和 `sys.path` hack。
- Notebook 只保留结果展示，核心逻辑必须能够从 Python 模块导入。

## 12. 后续维护优先级

建议按以下顺序继续：

### P0：建立持续集成

在 CI 中固定执行：

- Python 3.10 和 3.11 的单元测试。
- CPU `experiments.smoke_dpjax`。
- `uv lock --check` 和 `pip check`。
- YAML、Notebook JSON 和服务器 Bash 语法检查。

这能防止当前人工质量门禁随着后续提交失效。

### P0：实现非线性变换的物理链式法则

当前这是 Halo power-transform DF 进入 Φ/CBE 的主要阻塞。实现时必须同时处理：

- `∂log f/∂eta_phys` 的一阶链式法则。
- Φ 输入坐标与标准化/变换之间的映射。
- Laplacian 所需的二阶导数项。
- Plummer 或其他解析分布上的数值回归。

完成前保持现有显式拒绝，不允许静默近似。

### P1：统一训练指标

当前训练器写 `metrics.csv`，统一 logger 还会写 `logger_metrics.csv`。后续可定义
单一指标事件源，由 CSV、W&B 和 TensorBoard 共同消费，从而避免双份 CSV 字段
逐渐不一致。

### P1：配置强类型校验

为 data、flow、potential 和 train 配置增加 dataclass 或等价 schema 校验，
在启动前报告缺失键、错误类型和非法组合，减少运行数小时后才发现配置问题。

### P1：记录完整实验清单

训练启动时自动保存：

- Git commit 和 dirty 状态。
- 数据文件 SHA-256。
- Python/JAX/JAXLIB 版本。
- 设备列表。
- 随机种子和最终合并配置。

这比依赖目录命名更可靠。

### P2：退役 RealNVP

先盘点仍需读取的 RealNVP 检查点，并提供导出或迁移方案。只有在 smoke、文档和
历史实验都不再依赖后，才能从 registry 和依赖中删除。

### P2：拆分大型实验说明

`README_JAX.md` 应逐步收敛为快速入口，长篇调参记录放入版本化实验日志；本文
保持操作契约，`PROJECT_STRUCTURE.md` 保持架构说明，避免三份文档重复维护。

## 13. 周期性维护清单

每月：

- 运行完整测试和 CPU smoke。
- 检查依赖安全与兼容更新。
- 抽查主要 CLI 帮助和文档命令。
- 清理无引用的临时脚本和空目录。

每个重要实验完成后：

- 保存配置、commit、数据校验值和环境版本。
- 生成 DF/Φ 标准评估报告。
- 将重要运行目录复制到受备份的外部存储。
- 记录成功条件和失败条件，不只记录最佳指标。

每季度或里程碑：

- 在目标 GPU 上执行完整小规模两阶段流程。
- 检查检查点向后兼容性。
- 盘点弃用功能和归档候选。
- 检查文档是否引用已删除的文件或命令。
- 评估是否需要更新依赖下限、Python 版本或 CUDA 基线。

遵循这套边界后，仓库应长期保持“核心逻辑可导入、实验入口轻量、配置可复现、
历史代码隔离、物理变换显式安全”的结构。
