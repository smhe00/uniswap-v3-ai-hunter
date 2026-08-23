# Architect Review — R0-T001 Iteration 3

## Decision

`PASS`

R0-T001（旧系统结论来源与可复现性审计）通过。

## Reviewed Snapshot

- remote_head: `f3660e9ece9b97296dbab6fc493ecdbee3187bbd`
- task_id: `R0-T001`
- iteration reviewed: `3`
- consumed harness handoff: `R0-T001-HARNESS-20260823-003`

## Independent Review Findings

1. Iteration 3 实际 diff 仅修正 Claim `R0-T001-C7` 的样本外状态及其派生结果、测试和交接文件，未修改旧策略代码、模型或原始数据。
2. `R0-T001-C7` 的 `oos_status` 已从 `OVERLAP` 改为 `UNKNOWN`，与 `v3_hunter_monte_carlo.py` 的脚本级判断一致。
3. 新增一致性测试明确断言 Claim C7 与 Leakage Matrix 均为 `UNKNOWN`。
4. Harness 报告记录 `15 passed in 0.13s`；本次 diff 与该测试新增数量一致。
5. Iteration 1/2 中已确认的范围违规、模型缺失降级测试和过度样本外判定均已修正。

其中：

- **OOS = Out-of-Sample，样本外验证**。
- `UNKNOWN` 表示当前证据不足以证明严格样本外，也不足以证明一定发生训练/验证重叠。

## R0-T001 Final Audit Conclusions

以下旧数字不得继续作为已验证绩效使用：

- `$29,270` 最终净值；
- `+40.3%` 总 ROI；
- `+45.3%` 相对 Alpha；
- `91.7%` Monte Carlo 胜率；
- `+40.44%`、`+47.65%`、`+32.88%`、`+24.15%`；
- “原子级 / Raw Log 已真实逐笔验证”的表述。

其中：

- **ROI = Return on Investment，投资回报率**；
- **Alpha = Excess Return，超额收益**；
- **Monte Carlo Simulation = 蒙特卡罗模拟**；
- **Raw Log = 原始链上事件日志**。

可以作为“待验证候选”保留的旧资产包括：

- `RANGE_PCT = ±8.13%`；
- `XGB_RISK_THRESHOLD = 0.57`；
- 4 天再平衡冷却期；
- 19 个旧模型特征；
- LP / ETH / USDC 三态切换架构。

但这些都不等于已经通过未来数据验证。

## Why R0-T002

旧项目在 2026-03-13 已形成并提交其所谓“最终版本”，而本地数据已经延伸到 2026-08-21。因而 2026-03-14 之后的数据是在旧代码、旧参数和旧模型冻结之后才发生的，可以构造比旧回测更强的时间外验证。

下一任务固定旧参数，不重新优化，直接检验冻结系统在 2026-03-14 至 2026-08-21 后冻结数据上的表现，并与 Always LP / ETH / USDC 等简单基准比较。

## Next State

- R0-T001: `PASS`
- next task: `R0-T002`
- next owner: `harness`
- next state: `HARNESS_READY`
