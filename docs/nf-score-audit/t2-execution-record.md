# T2 执行记录：弱 Stein score 检验校准（Agent B 交付）

冻结日期 2026-09-30。任务卡：t0-task-cards.md 卡 T2；上游 t0-manifest.json（总体/分层/heldout=探索资格/种子候选）＋ t1-cache-registry.json（真实检验前置，certified）。完整校准报告与数组在项目 artifacts：nf-score-audit-t2/（t2-weak-score-calibration-report.md、data/metrics.json、data/weak_score_tests_calibration.npz、figures/t2-calibration.*）。

## 任务 / 状态

**完成（校准机械通过全部 7 门；真实 heldout 探索性估计已出，全程 EXPLORATORY）**。三层状态：代码＝新增 scripts/auriga/nf_weak_score_tests.py ＋ run_nf_weak_score_tests.sh ＋ nf_weak_score_stage.txt ＋ requirements-weak-score.txt ＋ tests/test_nf_weak_score_tests.py（解析测试 15 项；本地全套 81 passed / 3 skipped，worktree 自建 .venv，未触碰共享审计代码）；数值＝run 91a17beb @ 3b06e59（R=240，172.7 s，rc=0）；科学＝校准合格只证明"检验机械在 mock 上无失真"，不构成对 control 模型的任何判断。

## 代码

- base 7c7d256（父节点 tip）→ 本卡 commits：f7e161e（检验库+校准+runner+测试）、7f536d2（local 快照内自建 pinned venv：numpy/scipy/h5py，无 jax 依赖）、83416c3（macOS 无 GNU timeout → 进程内 SIGALRM 预算）、3b06e59（名义 χ² 联合门失败后的经验联合校准修复）。
- 节点 1f76800e 命令重定向为 bash scripts/auriga/run_nf_weak_score_tests.sh（与 T1 相同的带时间戳备份 orx.db 单行更新，orx exp status 复核通过）。stage 文件 nf_weak_score_stage.txt 当前 'calibrate'；'full' 需另行提交。
- 停止条件编码在 runner/脚本内：校准门任一失败 → rc=3 立即退出（c8b9df6a 触发过一次）；真实阶段 sha256/点序 hash 不符 → 拒绝；缓存缺失 → 干净跳过并打印 REAL-SAMPLE STAGE SKIPPED。
- 运行史：232eb65c 失败（timeout 命令缺失，实现性即死即修）；c8b9df6a 失败（χ₂₁ 名义联合参考反保守：FPR 0.117–0.129 vs 0.05——真实校准发现，修复＝经验/自助联合参考）；91a17beb 通过（calibrate）；1ffb762e（首次 full）maxT p 因统计量定义错误无效——事后审计发现 |t−boot_mean|/sd 恒≈0，修复 c6d3457（观察统计量＝|t|/sd＋回归测试）；6778ac8c 通过（full，最终）。

## 冻结契约（摘要）

- 检验库 h=e_j ψ：S/V/W 三族 63 列（空间径向 / 速度均值 / 速度弥散），窗 [3,7] q=30–70 kpc C∞ bump（导数入散度），波长 {5,10,20} kpc，幅度归一 E[u²]=1（mock 径向律四分差冻结）。
- 权重：Hájek 三明治 Σw²(g−t)²/(Σw)²；质量权重与对数正态权重臂均通过零假设门。
- mock：Plummer a_q=0.68 精确逆 CDF ＋ σ²=(1+r²/a²)^(−1/2) 高斯速度，heldout 尺寸带计数 [15832,5806,2012]；种子 mock=5、boot=4。

## 发现（校准，R=240）

- 零假设：单变量 FPR 0.052（Wilson [0.029,0.085]）、95% 覆盖 0.948、max|mean z| 0.14–0.18；joint max|z| 经验 FPR 0.067–0.092。
- **名义 χ₂₁ 族联合参考反保守**（FPR 0.117–0.129，均值仍 ≈21）：联合推断必须用经验（mock 分裂半样本）或自助（真实样本）临界值；名义值保留为诊断。
- f32 量级 score 噪声（T1 外区中位 2.7e-3 与 p99 3e-2 两臂）不抬高 FPR（三明治方差同步吸收）。
- 功效（ε_ref＝匹配位移 2SE：S 0.078/V1 0.035/V2 0.136）：S m=0 匹配列 0.11/0.49/0.93 @0.25/1/2×ε_ref，联合 1.00 @2×；m≥1 余弦注入主要落在 sin 列（∇cos∝−sin），联合在 1×ε_ref 全检出而同指标列 0.04–0.07——联合统计量必要性的直接证据。非梯度压力场 0.88/1.00（只作压力标注）。
- 重采样：角向 48 簇宽度 ×2.24、覆盖 0.94（真实阶段主参考）；径向 8 块 ×13.2（极端保守，仅敏感性记录）。

## 真实样本阶段（EXPLORATORY，已完成）

缓存取回已获用户批准（一次性只读 scp 至 artifacts nf-score-audit-t2/t1-cache-local/，sha256 83a20cb4…/99f25bfd… 与注册表精确匹配，点序 hash 三方一致；兼作 T1 产物异地备份）。stage full run 6778ac8c（c6d3457）：窗口 n=19,975 / n_eff=19,011。**结果对相关性假设高度敏感**：iid 参考下三族 max-Q p≈0、maxT p=0.000、20/63 列排除 0（预注册规则触发 flagged=True）；角向簇参考（主读出）下仅 V 族联合 p=0.0245 存活、10 列排除 0；径向块全不显著（该方案标定宽度 ×13–28，仅敏感性）。角向簇稳健列的结构：S{x,y,z,sin λ=1.0}（10 kpc 径向波长，三分量同号，连径向块都存活）、V 族负向（V{y,const} |t|≈6.6×ε_ref）、W{x,z,const} 正向（W{x,const}=全局 max|z| 5.17）。效应尺度 1–7× 单统计量 2σ 参考幅度，f32 噪声臂已排除数值解释。完整表格与边界见 artifacts nf-score-audit-t2/t2-real-heldout-exploratory.md。

## 限制与移交

- mock 经验临界值向 Auriga 的迁移是近似（径向律不同）——真实阶段以样本内自助临界值为主。
- heldout＝探索资格不变；存储 f32 score 外区链误差（中位 2.7e-3/p99 0.03 floored-rel）给出可检出效应下限的语境。
- 有限投影通过 ≠ 逐点 score 正确（方案 §T2 停止条件原文）；本记录只认证校准。
- 下一步（协调）：T2 交付完毕。真实读出的主结论＝速度侧（V 均值型＋W 弥散常数窗）与空间侧 10 kpc sin 分量的簇稳健偏移，成团使 iid 参考不可用作主读出；交 T5 裁决时与 T3 力反演、外区稀疏证据合并判断。

