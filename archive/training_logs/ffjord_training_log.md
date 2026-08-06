# FFJORD DF Training Log — Halo12

本文档记录了 Halo12 数据集上 FFJORD 分布函数（DF）模型的训练迭代过程，
包括每次参数调整的动机、变更内容和结果。

数据：Auriga halo12 模拟粒子，6D 相空间 (x, y, z, vx, vy, vz)。

---

## 实验总览

| Run | Transform | Network | Reg | LR | Epochs | NLL | val_NLL | p99 | 结果 |
|-----|-----------|---------|-----|-----|--------|-----|---------|-----|------|
| v10 | asinh(0-2)+zscore (BUG: no-op) | 6L×128 | 1e-3 | 1e-3 | 30 | diverged | — | — | ❌ NLL→-1600 发散 |
| v11 | asinh(0-2)+zscore (BUG: no-op) | 6L×128 | 1e-3→5e-4 | 5e-4(cos) | 30 | stagnated | — | — | ❌ 正则过高，无学习信号 |
| v12 | asinh(0-2)+zscore (BUG: no-op) | 6L×128 | 5e-4→1e-4 | 5e-4→1e-4(cos) | 30 | stagnated | — | — | ❌ 同 v11 |
| v13 | asinh(0-2)+zscore (BUG: no-op) | 6L×128 | 1e-3 | 1e-3→5e-3(step) | early-stop | — | — | — | ❌ lr 过高，early-stop |
| v14 | asinh(0-2)+zscore, clip+σ=4.5 (BUG: no-op) | 6L×128 | 1e-3 | 1e-3(cos) | 30 | diverged | — | — | ❌ score 爆炸 step ~75-100 |
| v15 | asinh(0-2)+zscore, clip+σ=4.5, jitter=0.01 (BUG: no-op) | 6L×128 | 1e-3 | 1e-3(cos) | 30 | 5.74 | 5.71 | 26.3 | ⚠️ 但 transform 是 no-op |
| v16 | asinh(0-2)+zscore, clip+σ=4.5, jitter=0.01 (BUG: no-op) | 6L×128 | 5e-4 | 5e-4(cos) | 30 | ~6 | — | — | ⚠️ 同上 |
| v17 | asinh(0-2)+zscore, clip+σ=4.5, jitter=0.01 (BUG: no-op) | 6L×128 | 1e-3 | 1e-3(const) | ~1200 | collapsed | — | — | ❌ step ~1200 崩塌 |
| v18 | asinh(0-2)+zscore, clip+σ=4.5, jitter=0.01 (BUG: no-op) | 6L×128 | 1e-3 | 1e-3(cos decay) | ~1200 | collapsed | — | — | ❌ 同 v17 同时崩塌 |
| v19 | power(α=0.3, ALL)+zscore, clip+σ=4.5, jitter=0.01 | 6L×128 | 1e-3 | 1e-3(cos) | 30 | 4.16 | 4.10 | 20.8 | ⚠️ 稳定但 kurtosis 严重偏差 |
| v20 | power(α=0.3, ALL)+zscore, clip+σ=4.5, jitter=0.01 | 6L×128 | 1e-5 | 1e-3(cos) | 30 | 4.18 | 4.12 | 20.5 | ⚠️ 与 v19 几乎相同 |
| **v21** | **power(α=0.5, ALL)+zscore, clip+σ=4.5, jitter=0.01** | **12L×128** | **1e-5** | **3e-4(cos)** | **20** | **5.83** | **5.84** | **15.2** | **✅ 最佳，ρ(r)±10% r<50kpc** |

---

## 关键发现

### 1. CoordinateTransform Bug（v10-v18 全部受影响）

`dpjax/data.py` 中 `CoordinateTransform.transform()` 使用 numpy 高级索引：
```python
x = data[:, self.dims]   # self.dims 是 ndarray → 高级索引 → 创建副本！
x[:] = np.arcsinh(...)   # 修改副本，原数组不变
```
**结果**：v10-v18 所有坐标变换都是 no-op，模型实际在 raw z-scored 数据上训练。
z 维原始 kurtosis=20，这解释了为什么所有 run 都不收敛或崩塌。

**修复**：
```python
idx = self.dims
data[:, idx] = np.arcsinh(data[:, idx] / scale)  # 直接索引赋值，原地修改
```

### 2. 变换选择：power(α) vs asinh

| 变换 | 效果 | 问题 |
|------|------|------|
| asinh(velocity dims) | vx/vy 近高斯 kurt≈2.3, 无需变换 | z 维 kurt 仍≈20 |
| power(α=0.3, ALL) | kurtosis 全部降至 <2.6 | 过度压缩，出现 platykurtic（负 kurtosis），难以建模 |
| **power(α=0.5, ALL)** | **z kurt=3.65（近高斯）, 其他维度适中** | **压缩与保持结构的良好平衡** |

### 3. 正则化影响很小

v19 (reg=1e-3) vs v20 (reg=1e-5)：NLL、val_NLL、p99 几乎相同。
说明 FFJORD 的瓶颈不在正则化，而在网络容量或数据变换。

### 4. 网络容量提升有帮助但有限

v20 (6层) → v21 (12层)：
- 原始空间 kurtosis 大幅改善：z: 92→30, x: 33→11.52
- 但变换空间质量反而变差：std 偏移从 1.2-4x 恶化到 1.8-10.8x
- 原始空间的改善来自 α=0.5 更温和的逆变换压缩偏差，而非 flow 本身的改善

### 5. FFJORD 变换空间质量核心问题

即使 v21 是最佳结果，变换后空间中模型 std 仍达数据的 1.8-10.8 倍。
这意味着 FFJORD 没有真正收敛到数据的联合分布，只是在单变量边际上较好地恢复了原始空间统计量。

**物理后果**：
- ρ(r) 内区 (r<50 kpc) ±10%
- σ_v(r) 内区 (r<15 kpc) <5%
- r>50 kpc 严重恶化：模型将粒子放到 r≈145 kpc（数据 r_max=75 kpc）
- 重尾行为说明 flow 没有正确建模分布尾部

---

## 详细记录

### v10-v13：asinh 变换初期探索（transform bug，均为 no-op）

**背景**：首次在 halo12 数据上训练 FFJORD，使用 asinh 预处理空间维度。

- **v10**：基线配置。NLL 直接发散至 -1600。原因：高 kurtosis 数据导致 ODE 求解器不稳定。
- **v11**：降低正则化 (1e-3→5e-4)。NLL 停滞，说明正则过高压制了 score 学习。
- **v12**：继续降低正则化 (5e-4→1e-4)。同样停滞。
- **v13**：尝试学习率 warmup。lr 过高导致 early-stop。

### v14-v15：加入 sigma-clip 和 jitter

**动机**：v10 的发散可能与极端异常值有关。

- **v14**：clip_sigma=4.5 + jitter_std=0.01。所有 run 在 step 75-100 发散（score 爆炸）
- **v15**：同 v14 配置。最终 NLL=5.74。但 transform 是 no-op，模型实际只学 z-scored 数据。

### v16-v18：稳定性与学习率方案

- **v16**：降低 lr 到 5e-4。与 v15 相当。
- **v17**：恢复 lr=1e-3，constant。在 step ~1200 崩塌。
- **v18**：lr=1e-3, cosine decay。同样在 step ~1200 崩塌。
- **结论**：崩塌与 lr 方案无关，是 FFJORD ODE 数值不稳定（velocity field 过陡）。不同 ODE solver 无济于事。

### v19：修复 transform bug + power(α=0.3)

**变更**：
- 修复 `CoordinateTransform.transform()` 的 numpy 高级索引 bug
- 变换从 asinh(dims 0-2) 改为 power(α=0.3, ALL 6 dims)
- power(0.3) 将所有 kurtosis 降至 <2.6

**结果**：NLL=4.16, val_NLL=4.10, p99=20.8。30 epochs 稳定训练。
**问题**：原始空间评估显示 kurtosis 严重偏差（z: 92 vs data 22, x: 33 vs data 12）。
KS test 全部失败 (p<0.01)。

### v20：降低正则化

**变更**：reg 从 1e-3 降至 1e-5（匹配 Green 2023 论文推荐）。

**结果**：NLL=4.18, val_NLL=4.12。与 v19 几乎相同。
**结论**：正则化不是瓶颈。

### v21：α=0.5 + 12 层网络 ⭐

**变更**：
- power α 从 0.3 升至 0.5
  - α=0.3 使所有维度 platykurtic（负 kurtosis），难以建模
  - α=0.5 保持 z kurtosis=3.65（近高斯），更好
- 网络从 6 层增至 12 层（匹配 Green 2023 论文架构）
- 学习率 3e-4, cosine decay, 20 epochs

**结果**：NLL=5.83, val_NLL=5.84, p99=15.2, val_p99=14.5

**原始空间评估**：
| 维度 | 数据 kurtosis | 模型 kurtosis | KS p-value |
|------|--------------|--------------|------------|
| x    | 11.67        | 11.52        | <0.01      |
| y    | 4.17         | 10.56        | <0.01      |
| z    | 22.40        | 30.04        | 0.035 ✓    |
| vx   | 2.36         | 3.23         | <0.01      |
| vy   | 2.28         | 3.19         | <0.01      |
| vz   | 4.80         | 4.92         | 0.13 ✓     |

**变换空间评估**：模型 std 1.8-10.8x 数据 std（比 v20 的 1.2-4x 更差）。

**物理评估**：
- ρ(r)：r=5-40 kpc 范围内 ±10%，r=40-50 kpc 约 ±20%
- σ_v(r)：r<15 kpc <5%，外区恶化
- β(r)：趋势正确但噪声大
- r_max：模型 ~145 kpc vs 数据 75 kpc（外区重尾问题）

---

## 当前诊断与下一步方向

### 核心问题

FFJORD 在 6D 相空间分布的建模上存在根本局限：
1. 变换空间中模型 std 偏移 1.8-10.8x，说明没有正确学到多变量依赖结构
2. 外区 (r>50 kpc) 重尾，模型概率质量放置不正确
3. 增加网络层（6→12）反而恶化变换空间质量

### 可选方向

- **A) 继续调参 FFJORD**：更多层、不同激活函数、不同 ODE solver。边际收益预期很小。
- **B) 切换到 Neural Spline Flow (NSF)**：Green 2023 推荐 NSF 作为 drop-in 替代。NSF 对重尾分布建模更好，不会有 ODE 数值不稳定。代码库需实现 NSF。
- **C) 更激进的数据变换**：log 或 Box-Cox 变换进一步压缩重尾，或 PCA whitening 重构维度间依赖。
- **D) 尾部约束**：在 loss 中加入尾部正则项，或对 flow 输出做 clip/bound。

---

## 数据管线（当前）

```
raw (N, 6) → power(α=0.5, dims=[0,1,2,3,4,5]) → sigma_clip(4.5σ)
           → z-score(ALL 6 dims) → jitter(σ=0.01)
```

- power(α=0.5)：`sign(x) * |x|^α`，在 clip 之前应用
- sigma_clip(4.5σ)：移除 ~1.73% 样本（pre-asinh），post-power 约 0.08%
- jitter(σ=0.01)：防止 score function 奇点

## 配置文件命名

- 配置：`configs/df_halo12_ffjord_v{N}.yaml`
- 运行目录：`runs/halo_12/df_ffjord_v{N}`
- 评估脚本：`scripts/eval_v19.py`（原始空间）、`scripts/eval_transformed.py`（变换空间）、`scripts/eval_physics.py`（物理诊断）