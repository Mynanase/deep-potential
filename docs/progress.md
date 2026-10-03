# 近期研究状态（2026-10-04）

当前问题：用完整 Plummer mock 比较 P00（解析 score）与 P11（项目 conditional NF score）的 Phi 结果。
代码已回主线；两臂还没有有效训练结果，不能据此解释 Auriga 的负密度或高频误差。

## 已有证据

| 工作 | 最终 run / source commit | 可以保留的结论与限制 |
| --- | --- | --- |
| T1 数值链与固定点缓存 | `f7f2c973-5a1b-4330-ba72-286526b9e3d2` / `8948905` | 8 项数值检查通过，缓存合格；只说明导数计算可靠，不认证 NF 的物理正确性。 |
| T2 Stein 检验 | `6778ac8c-cd19-470c-9746-339740895417` / `c6d3457` | 校准通过；Auriga 读出仍是探索性。cluster/block maxT p=0.2235/0.1645，不能单独认定 score 偏差。 |
| T3 Plummer oracle / 局部力 | `fade7ea7-d38c-4796-8450-8daa0139393e` / `00f7049` | 完整 mock 与解析 machinery 合格。Auriga 真势局部梯度相对误差中位数约 25%，不能用作可靠真力标签。 |
| T4a Plummer conditional NF | `14e2e76a-d059-42df-8f7d-28694fff1701` / `1d5b18f` | 项目 conditional NF 的数值/缓存检查通过。score 相对误差 median≈3.1%、p99≈20.4%；这还不是下游 Phi 结论。 |
| T4 P00/P11 | `f1270fa6-81cd-4ed6-9f09-1b20afad02e6` / `da67573` | 启动前分层约束配额不足，失败；没有完成配对科学比较。 |

T4a 的训练 checkpoint 来自 `dac804e7-9cac-490f-9771-5dfa998624da`；该 run 的下游实现失败，
最终资格 run 只复用了已完成训练 checkpoint 来修复下游计算。保留这一来源区别，不把资格 run 当重新训练。

## T4 尚未解决的问题

实际完整 mock 在 q=[0.1,0.2,1,2,3,4.5,6,7] 各档的可用数量为：
`[31501, 879219, 458855, 127494, 63501, 23230, 8094]`。
最新已提交配额末两档要求 37350/11094，均超过可用量。
旧 worktree 的未提交补丁改为 23230/25214，最后一档仍不足；补丁随工作区归档保存，未作为有效修复合入主线。

下一步只需先确定可行的约束分配及其权重，在真实输入上验证，再开展同条件 P00/P11。
本次整理没有改变配额、重新训练，或扩展到混合臂与架构实验。

## Halo12 对照与前一轮

- Phase-2 control 为 innerA、lambda=1，96³ production truth；lambda=10 是历史对照。
- truth 分辨率研究没有建立全尺度收敛；约 10 kpc 以上波长用于定量谱比较，<5 kpc 仅诊断。
- round-1 的 osc-pair、spectral-norm ceiling、lambda anneal 没有改善定量域振荡指标；不作为主线默认设置。
- 相关报告保留在旧项目 artifacts：`particle-truth-spectrum-resolution/`、`osc-suppression-round1/`、`nf-score-audit-t2/`、`plummer-oracle-t3/`。

## 代码与证据的位置

最近 T1–T4 实现来自 `orx/t4-minimal-p00-versus-p11-paired-phi`；truth 分辨率工具来自
`orx/particle-truth-spectrum-resolution-convergence-6`。原实验分支和 source commit 保留。
完整旧计划、任务卡、manifest 和执行记录在 [history](history/)，不是新会话的任务要求。
运行日志用 `orx logs <run-id>` 查询；本地 CPU run、服务器 checkpoint、项目 artifacts 保持原路径。

重开会话时带上当前问题、相关代码和所需输入即可。完成一次判断后更新本页的状态，
历史细节留在 run 日志和 Git，无需复制旧计划或继续使用 A/B/C 派发角色。
