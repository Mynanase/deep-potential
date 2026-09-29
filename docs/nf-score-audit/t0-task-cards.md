# T0 任务卡：A / B / C（派发时逐张复制；未列出的训练与后续阶段不在授权范围）

通用：执行说明 `docs/nf-score-audit-plan.md` 第 1–4 节 ＋ 对应任务卡 ＋ 必要附录；交接格式见其第 6 节；上游输入以 `t0-manifest.json` 为准；交付写入 `runs/nf-score-audit/<stage>/` 并同步登记 manifest。

## 卡 T1 / Agent A：数值链与共享 score 缓存

- **目标**：证明（或证伪）control 模型在目标区域的 score 数值链可靠，产出 B/C 共用的固定点缓存。
- **输入**：t0-manifest.json（checkpoints＋sha256、服务器路径、S1 契约）；服务器权威解释器 `/localdisk/kosmos/my-deep-potential/.venv-halo/bin/python`、JAX flags（cuda、禁预分配）。
- **允许执行**：本地 CPU 解析测试与完整 pytest；只读访问服务器 control 快照；吞吐试点（不超过 2048 点、单卡、不超过 30 min）与全量缓存（不超过 2 h 单卡）须在本卡被接受时由用户逐项授权，且必须通过编排节点＋已提交 runner 执行（先把节点 45b0446d 命令改指向 runner 再启动，不得直接启动该节点）。
- **文件归属**：`scripts/auriga/audit_df_constraints.py`（适配）、新 `scripts/auriga/nf_score_cache.py`、`tests/test_df_constraint_audit.py`（扩展）；禁改 fit_all.py / potential.py 的生产行为。
- **必做转换**：α=−∇qφ（命名＋符号测试）；不搬 2026-09-15 历史 run 的阈值与 checkpoint 编号；FD 需相邻步长稳定区间；ODE 容差与精度敏感性；control 为单 flow 对，ensemble/混合 score 仅作记录性检查。
- **交付**：数值检查报告 ＋ 固定点缓存（manifest 绑模型/代码/点序/精度 hash）＋ 第 6 节交接。
- **停止条件**：hash 不符；数值链失败（缓存可交排错用，但标明不可供 B/C 科学结论）；预算超限。

## 卡 T2 / Agent B：Stein 弱 score 检验（校准先行）

- **目标**：在解析 mock 上校准弱检验的零假设/功效/区间，再（若 T1 缓存合格）对真实 heldout 给出探索性估计。
- **输入**：t0-manifest（总体、分层、heldout 资格＝探索、种子候选）；T1 合格缓存（真实检验的前置）。
- **允许执行**：仅本地 CPU（无服务器、无 GPU）；不训练。
- **文件归属**：新 `scripts/auriga/nf_weak_score_tests.py` ＋ 解析测试；不改共享审计代码（属 A）。
- **必做**：h=e_j ψ(z) 固定函数库冻结（六分量/外区平滑窗/预定波长）；窗口导数入散度；质量权重；独立重复样本的假阳性率不确定度；注入 δlogF=εψ_k 的功效；联合区间处理多函数；cluster/block 重采样敏感性。
- **交付**：`weak_score_tests` 数组/指标 ＋ 校准报告；Auriga 结果一律标 exploratory。
- **停止条件**：校准失真先修检验；T1 缓存不合格时不得报告真实样本结果。

## 卡 T3 / Agent C：Plummer oracle 与局部力反演

- **目标**：O0（真势＋解析 score）与 O1（解析 score＋局部解）合格；加权 SVD、δα 投影与可吸收/正交分解在解析例上成立。
- **输入**：t0-manifest（无 mock NF；**逐点真势存在**（`true_potential_assets`）、**无逐点加速度块**；**mock 参数已在 T0 冻结**：参考实现＝`0c7e23f` Plummer 体系，a=6.8 kpc、N_mock=1,619,615、检测区三带 ±25% 密度匹配，见 `mock_design`）；共享速度点定义（与 T1 卡协调）。
- **允许执行**：仅本地 CPU；不训练；Auriga 逐点真势的只读核验（经批准的只读 ssh 或服务器导出的少量数组）。
- **文件归属**：`scripts/plummer/plummer_sphere.py`（修复 draw_from_sphere 的 rng 违规并加测试）、新 `scripts/plummer/plummer_oracle.py`（解析 score/真加速度/质量＋oracle 输入接口）＋ 解析测试；局部力求解放 `scripts/auriga/local_force_inversion.py`（α 约定）。
- **必做**：从 `0c7e23f` 移植生成器与解析 score 源并**按冻结参数重新生成 mock 样本**（纯 CPU、带完整 lineage attrs；legacy `plummer_n*.h5` 禁用）；验证解析分带份额 vs 冻结值在 ±25% 容差内（二项噪声＋采样器离散误差单列）；**不使用 r-cut／选择变体**：完整 Plummer 数据，oracle score 恒为完整解析 score，无选择修正；采样器离散精度与 RNG 修复；g=∇φ 到 α=−∇φ 转换测试；秩亏不给稳定力认证；mean(E)=0 但投影偏差的反例；能量依赖反例；单位与符号；**A5 前置小项**＝逐点真势单位标定（对单极模型/已知 M_total，消 `acceleration_unit=unknown`）＋主轴系局部梯度估计的误差定量（外区稀疏重点检查）。
- **交付**：O0/O1 测试证据 ＋ 局部四组合框架（NF 接口留给 T1 缓存）＋ 支持图数据；A5 可行性结论（条件可行/阻断）。
- **停止条件**：oracle 公式/单位失败即阻断 T4 前置；α* 梯度估计误差未定量前不得用于 A5 或任何真力对照；不扩大为 Auriga 真力声明。
