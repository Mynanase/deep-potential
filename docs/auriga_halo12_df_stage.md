# Auriga Halo12 DF 阶段实施指南

## 本阶段科学目标

用 Halo12 恒星粒子训练静态六维分布函数（DF），首先验证：

1. 生成样本的球对称质量密度剖面与输入恒星质量分布一致；
2. 独立随机种子训练得到的物理坐标 score
   `∇_(x,v) log f` 在主要数据区域内可重复；
3. DF 使用的预处理可安全进入后续静态 CBE/Phi 阶段。

Halo12 没有解析的六维 score 真值。因此，当前的 score 验收是“独立模型一致性
+ held-out Stein consistency”，不能表述为绝对梯度真值误差。真正的下游物理验收
仍需用 Auriga 力场真值检查恢复出的加速度。

## 为什么使用质量权重

`halo_12_stars.hdf5` 有 1,652,969 个恒星粒子，粒子质量并不完全相同。若直接把每个
粒子作为等权样本，flow 拟合的是粒子数密度而不是质量密度。规范化预处理会写入
`tracer_weight = Masses / mean(Masses)`，训练损失、normalizer 和验证损失均使用该
权重。

## 服务器运行顺序

### 1. 生成全恒星规范化数据

```bash
env \
  INPUT_PATH=/path/to/halo_12_stars.hdf5 \
  OUTPUT_PATH=data/auriga/halo12_all_mass.h5 \
  bash jobs/prepare_halo12_df.sh
```

若全样本训练不可行，可把 `COMPONENT` 设为 `cold`、`warm`、`hot` 或 `counter`。
多个成分的组合应直接调用 `experiments.prepare_auriga`，重复使用
`--component` 参数。

准备作业会重新计算运动学标签。原始提取脚本曾使用 `v²` 代替 binding energy；
当前实现已改为 `Potential + 0.5 v²`，因此不要直接依赖旧文件中的
`kinematic_label`。

### 2. 先提交一个种子作为数值门禁

```bash
env \
  DATA_PATH=data/auriga/halo12_all_mass.h5 \
  SEEDS=42 \
  GPU_DEVICES=0,1 \
  bash jobs/train_halo12_df_ensemble.sh
```

确认 NLL 有下降趋势、没有触发 score early-stop、checkpoint 和 `metrics.csv`
正常后，再训练剩余三个 seed：

```bash
env \
  DATA_PATH=data/auriga/halo12_all_mass.h5 \
  SEEDS=43,44,45 \
  GPU_DEVICES=0,1 \
  bash jobs/train_halo12_df_ensemble.sh
```

### 3. 评估局部分布与 6D score

单个 seed 完成后即可先运行 data/model 分布诊断；四个 seed 全部成功后，同一入口
会额外计算 score ensemble：

```bash
env \
  DATA_PATH=data/auriga/halo12_all_mass.h5 \
  SEEDS=42,43,44,45 \
  GPU_DEVICES=0 \
  bash jobs/eval_halo12_df_ensemble.sh
```

需要离开 SSH 后继续运行时，可以使用：

```bash
mkdir -p logs
nohup env \
  INPUT_PATH=/path/to/halo_12_stars.hdf5 \
  GPU_DEVICES=0,1 \
  LOGGER=csv \
  bash jobs/run_halo12_df_pipeline.sh \
  > logs/halo12-pipeline.log 2>&1 &
```

完整流程默认顺序训练四个 seed，任一步失败都会停止，不会继续产生误导性的评估结果。

主要输出：

- `auriga_df_metrics.json`：恒星 tracer 质量密度、条件速度边缘分布和 score 指标；
- `auriga_df_diagnostics.npz`：保留独立 model/seed 维度的绘图数据；
- `density_profile.png`：球对称恒星 tracer 密度；
- `df_spatial_rz_by_phi.png`：不同方位角区间的 data/model `R-z` 密度和
  `log10(model/data)`；
- `velocity_marginals_by_r.png`、`velocity_marginals_by_theta.png`、
  `velocity_marginals_by_phi.png`：逐空间 bin 的
  `v_r/v_theta/v_phi` data/model 直方图；
- `score_consistency.png`：至少两个 seed 时生成。

上述密度是 DF 学到的归一化恒星 tracer 质量密度，不是
`nabla^2 Phi/(4 pi G)` 给出的总引力质量密度。

## 预注册的阶段性验收线

第一轮使用以下工程门槛，不把它们误当成最终科学结论：

- 密度剖面：`log10_rmse_dex <= 0.10`，且
  `p90_fractional_error <= 0.30`；
- score 有限值比例：`finite_point_fraction >= 0.999`；
- 独立模型 score：`pairwise_cosine_median >= 0.95`，
  `pairwise_cosine_p10 >= 0.80`；
- 六维相对 MAD：每一维 `< 0.20`，中位数 `< 0.10`；
- 训练过程中没有 score p99/max early-stop，四个种子均产生最终 checkpoint。

Stein 指标用于横向比较配置，不设置脱离尺度的绝对阈值。若密度通过而 score
不通过，优先增加 ensemble、训练时长或 regularization，而不是进入 Phi 训练。

## 成分回退顺序

若全 stars 数值不稳定，按以下顺序定位：

1. `hot`（压力支撑、最接近静态 halo tracer）；
2. `warm + hot`；
3. `cold` 单独建模；
4. 最后再尝试四成分混合 DF。

不要把不同动力学成分随意删除后仍称为“全恒星 DF”。每次运行必须在数据文件属性、
配置快照和运行目录名中保留成分定义。
