# Architect Review — R0-T002 Iteration 3 (REVISED Iteration 4 Scope)

## Decision

`CHANGES_REQUIRED`

本文件**取代上一版 Iteration 4 指令**。用户已明确：`退出 LP 后 4 天内不重新进入 LP` 是有意加入的 anti-churn（抑制短期震荡与频繁交易）策略规则，不是 bug。Harness 不得删除该规则，也不得根据 `lp_smart_agent.py` 当前表面状态机自行改成 SAFE->ACTIVE 立即重入。

Iteration 4 的目标不是继续追求所有细节完美，而是只修会实质改变策略结论或回测可信度的问题。小型口径/命名/审计瑕疵允许记录为 Technical Debt（技术债）后进入下一阶段。

## Reviewed Snapshot

- reviewed remote_head: `77c038f827867d7dde72391f6bed9e68eb8f0fda`
- task_id: `R0-T002`
- iteration reviewed: `3`
- consumed harness handoff: `R0-T002-HARNESS-20260823-004`

## Review Priority Policy

以后本任务问题分三级：

### P0 / Blocker
必须当前修复。包括：
- future leakage / 时间因果错误；
- 资本单位、LP 资产部署、NAV/PnL 主公式错误；
- Frozen 参数被重新优化或改变；
- 数据源/模型输入与声明严重不符；
- 会改变收益符号、主要状态占比或核心比较结论的实现错误；
- 正式 OOS 过程中大量模型输入异常却被静默当成有效信号。

### P1 / Material
可能影响几个百分点、交易次数或风险指标。优先验证，但只在证据显示会改变核心结论时阻塞。

### P2 / Technical Debt
字段命名、报告表达、辅助测试完备度、很小的 ledger 精度等。如果确认不影响当前策略结论，记录后续处理，不阻塞 R0-T002。

---

# Iteration 4 Fixed Strategy Semantics

## S1 — 保留退出后 4 天 cooldown（用户明确策略意图）

Frozen Legacy 继续使用当前 anti-churn 规则：

```text
ACTIVE -> SAFE
    退出 LP
    按 bull / bear / neither 路由到 ETH / USDC / KEEP
    记录退出时点作为 cooldown 起点

SAFE -> ACTIVE
    只有距离最近退出/重建 >= 4 days 才允许重新进入 LP
    否则保持 SAFE，并记录 COOLDOWN_SKIP

ACTIVE -> ACTIVE
    持续 ACTIVE 且距离上次重建 >= 4 days 时可 periodic rebalance
```

这条规则是当前冻结策略的一部分。**不得在 Iteration 4 删除、缩短或优化 4 天参数。**

Harness Report 必须把这条规则明确称为：

`4-day anti-churn exit/re-entry cooldown（4 天退出/再进入防震荡冷却）`

而不是误写为 production bug。

---

# Iteration 4 Required Work

## P0-1 — 模型输入异常不得 fail-open

当前实现若 `predict_proba()` 或特征读取异常后设 `risk_prob=0.0`，会把异常解释为最低风险。这是正式 OOS 的可信度问题。

### 固定要求

1. OOS 开始前验证模型所需 features 全部存在且顺序与 `models_15m.pkl['features']` 一致；
2. 正式决策行输入不得含 `NaN / inf / -inf`；
3. `predict_proba()` exception 不得降级为 0.0；正式全量回测必须 `raise RuntimeError`；
4. warmup 前置 NaN 可以在进入正式 OOS 前清理，但必须记录实际首个有效 OOS decision timestamp；
5. 输出：

```json
"model_input_audit": {
  "required_feature_count": <actual>,
  "missing_feature_count": 0,
  "non_finite_decision_rows": 0,
  "predict_errors": 0,
  "first_valid_oos_decision": "..."
}
```

### 必须测试
- missing feature -> RuntimeError
- predict_proba raises -> RuntimeError
- non-finite feature -> RuntimeError 或在正式 OOS 输入前明确剔除
- 完整 OOS anomaly counts 全为 0

如果异常只发生在 warmup 且已被正确过滤，可 PASS，不要为了极小 warmup 问题继续扩大任务。

---

## P0-2 — 保持并复核现有时间因果与 OHLC 修复

Iteration 3 的以下修复已基本正确，不重新设计，只要求回归测试继续通过：

- Binance 1m 使用完整 OHLC；
- `open=first / high=max / low=min / close=last`；
- 1m open time 映射成 close available time；
- 00:15 / 04:00 精确边界测试；
- pandas_ta 指标使用正确 NATR 百分比尺度。

如果这些测试全部保持通过，**不要继续为没有原生 15m/4h 文件而阻塞**。报告标记 `native_bar_parity = NOT_AVAILABLE` 即可。

---

## P0-3 — LP 主经济结果必须保持可对账

以下 invariants 必须继续满足：

```text
final NAV = position value + idle wallet value + final uncollected fee value
absolute reconciliation error < 0.02 USDC
```

中心建仓 snapshot：

```text
idle_ratio < 1%
```

fee-on / fee-off 必须保持相同 add/remove/rebalance 事件路径；若事件次数不同，则必须解释，否则该 counterfactual 无效。

Always LP 当前 `-5.83%` 可作为候选结论，但如果 Iteration 4 的 P0 修复没有触及 LP 基准实现，理论上不应显著变化。若变化 > 0.5 percentage point，必须定位原因。

---

## P1-1 — Fee ledger：只做最低必要修正，不阻塞策略迭代

Iteration 3 的 token 级 fee 统计方向可接受。当前主要问题是 `cum_fee`/`collected_fee` 命名可能把 accrued 与 collected 混淆。

本轮只要求：

```text
fee_accrued_eth
fee_accrued_usdc
fee_uncollected_final_eth
fee_uncollected_final_usdc
```

如果可以低风险地得到：

```text
fee_collected = fee_accrued - fee_uncollected_final
```

则一起输出；如果 Demeter 数据结构使这个拆分不够可靠，**不要为此重写 fee engine**，只需：

- 保留 accrued token 数量；
- 保留 final uncollected；
- 把“collected”旧字段标为 deprecated / 不再用于结论；
- 记录 Technical Debt。

核心策略判断优先使用：

```text
fee_on_nav - fee_off_nav
```

作为手续费对 NAV 的路径贡献 sanity check。

---

## P1-2 — Legacy-Cost 只需保持诚实命名

保留旧：

```text
latency_bias = 5 bps
exit deduction = 0.0002
```

必须明确称为：

`Legacy heuristic cost assumption（旧启发式成本假设）`

不是实际 Gas、真实滑点或真实历史成交成本。

如果很容易输出 `gross_nav - legacy_cost_nav`，继续输出；不要求为了拆解每一笔 cost 再增加复杂 ledger。

---

# Iteration 4 Explicitly NOT Required

以下上一版要求本轮撤销或降级，Harness 不要执行：

1. **不要删除退出后 4 天再进入 cooldown。**
2. 不要求 SAFE->ACTIVE 15 分钟后立即 re-enter。
3. 不要求新增 `post_freeze_oos_transitions.csv`；如果已开始但未提交，可停止。
4. 不要求把所有 fee accrued / collected 做到会计级完美，只需不误导主结论。
5. 不要求为了小字段命名或极端边界继续增加大量测试。
6. 不修改 legacy 策略文件、模型文件、README、协议或原始数据。
7. 不增加新策略、不调参、不改变 4 天 cooldown。

---

# Iteration 4 Minimum Tests

现有有效测试继续通过，并新增/确认以下核心测试：

1. `test_missing_model_feature_fails_fast`
2. `test_predict_exception_fails_fast`
3. `test_nonfinite_model_input_fails_fast`
4. `test_exit_reentry_cooldown_is_four_days`
   - EXIT at t0
   - t0 + 3d23h45m 即使 active 也不得 re-enter
   - t0 + 4d00h00m active 时允许 re-enter
5. 原 F7 OHLC aggregation test
6. 原 F8 exact 15m / 4h boundary tests
7. 原 LP deploy idle<1% invariant
8. 原 NAV reconciliation test
9. fee-on / fee-off action-path equality check

测试数量不是 Gate 指标；关键是这些行为测试真正覆盖主逻辑。

---

# Iteration 4 Mandatory Rerun Output

完整重跑同一 OOS：

```text
2026-03-14 .. 2026-08-21 UTC
```

禁止重新训练、重新调参或改变起止日期。

必须输出并与 Iteration 3 比较：

```text
Frozen Legacy Gross Return
Frozen Legacy Legacy-Cost Return
Always LP Gross Return
Always ETH Return
Frozen vs ETH excess
Maximum Drawdown
Sharpe Ratio
LP time ratio
ACTIVE->SAFE count
SAFE->ACTIVE count
COOLDOWN_SKIP count
PERIODIC_REBALANCE count
fee_on - fee_off NAV impact
model input anomaly counts
NAV reconciliation error
```

Harness 最终只回答四个策略问题：

1. 在保留 4-day anti-churn cooldown 后，Frozen Legacy 的 OOS 收益是多少？
2. 是否战胜 Always LP？
3. 是否战胜 Always ETH？
4. 该旧策略是否值得作为后续新 LP/ETH/USDC routing 研究的 benchmark，而不是继续深挖旧模型？

---

# PASS Standard for Iteration 4

满足以下条件即可推荐 PASS，不要继续追小问题：

- 时间因果测试保持通过；
- LP capital/NAV 主对账正确；
- 模型输入正式 OOS 无静默 fail-open；
- 4-day anti-churn cooldown 被保留且有行为测试；
- 同一冻结参数、同一 OOS 完整重跑；
- 主要结果和限制说明清楚；
- 剩余问题只属于 P1/P2，且不会合理地改变“旧策略 vs LP/ETH”的方向性判断。

所有剩余 P2 项写入 Harness Report 的 `Technical Debt` 小节，不阻塞下一阶段。

## Allowed Files

仍仅允许：

- `research/r0_t002_post_freeze_oos.py`
- `tests/test_r0_t002_post_freeze_oos.py`
- `results/r0_t002/post_freeze_oos.json`
- `results/r0_t002/post_freeze_oos.md`
- `results/r0_t002/post_freeze_oos_equity.csv`
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止新增其它文件。