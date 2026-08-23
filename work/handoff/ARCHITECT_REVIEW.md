# Architect Review — R0-T002 Iteration 3

## Decision

`CHANGES_REQUIRED`

Iteration 3 的质量相较 Iteration 2 明显提升：OHLC 聚合、bar available-time、LP 资本部署快照、token 级手续费、fee-off counterfactual、reconciliation 以及 deterministic periodic-rebalance 测试都已加入，且 Harness 报告为 `35 passed`。这些改进方向正确，可以保留。

但本轮仍不能 PASS。原因是我独立检查冻结生产脚本 `lp_smart_agent.py` 后确认：当前 Frozen Legacy 回测的**状态机仍未忠实复现生产语义**，并且模型异常处理、手续费字段命名/对账和 Legacy-Cost 口径仍有可导致结论偏差的问题。

当前报告中的 `Frozen Legacy +16.34%`、相对 ETH `-3.29%`、LP time ratio `3.9%` 仍视为 provisional（暂定），不得作为 R0-T002 最终结论。

## Reviewed Snapshot

- remote_head: `77c038f827867d7dde72391f6bed9e68eb8f0fda`
- task_id: `R0-T002`
- iteration reviewed: `3`
- consumed harness handoff: `R0-T002-HARNESS-20260823-004`

## Accepted Improvements

以下内容本轮已达到可继续使用的质量：

1. Binance 1m 读取完整 OHLC，不再从 close 伪造 high/low；
2. 1m open-time 映射到 `+1min` available-time 后再聚合，15m/4h 边界测试方向正确；
3. 研究特征改用 `pandas_ta` 同一路径，NATR 百分比尺度已修正；
4. LP 中心建仓按 V3 区间资产比例部署，建仓快照中 idle ratio <1%；
5. Always LP deterministic 4-day periodic rebalance 测试不再是空断言；
6. fee-on / fee-off 路径 action count 一致，可用于手续费边际贡献 sanity check；
7. 最终 NAV 与 position + idle wallet + final uncollected fee 的终值对账一致；
8. 未修改 legacy 策略文件、模型文件、README 或原始数据。

---

# Iteration 4 Execution Specification

本轮 DeepSeek Harness 继续视为**严格受控执行器**。不得自行改变设计、不得添加新策略、不得重新优化任何参数。只允许实现以下 F14-F18，并完整重跑。

## F14 — Frozen Legacy 状态机与 `lp_smart_agent.py` 不一致（CRITICAL，必须修）

生产脚本真实逻辑为：

```python
if old_mode == 'ACTIVE' and new_mode == 'SAFE':
    # 退出，但不更新 last_rebalance
    ...
elif new_mode == 'ACTIVE':
    if old_mode != 'ACTIVE':
        # 立即 RE-ENTER
        state['last_rebalance'] = now_ts
    elif days_since_reb >= REBALANCE_DELAY_DAYS:
        # 只有持续 ACTIVE 时才做 4-Day periodic rebalance
        state['last_rebalance'] = now_ts
```

因此冻结语义必须严格解释为：

```text
A. ACTIVE -> SAFE
   - 立即退出 LP
   - 按 bull / bear / neither 路由到 ETH / USDC / KEEP RATIO
   - 不更新 last_rebalance

B. SAFE -> ACTIVE
   - 立即重新进入 LP
   - 不受“退出后等待 4 天”限制
   - re-entry 成功后 last_rebalance = now

C. ACTIVE -> ACTIVE
   - 若已有 LP 且 now - last_rebalance >= 4 days：periodic rebalance
   - 否则 HOLD

D. SAFE -> SAFE
   - HOLD 当前 SAFE 资产
   - bull/bear 后续变化不得在 SAFE 状态内反复换仓，因为生产脚本没有该分支
```

### 当前错误

Iteration 3 在 `ACTIVE -> SAFE` 时执行：

```python
self.last_rebalance = now
```

并在 SAFE -> ACTIVE 时检查 4 天 cooldown。这个逻辑不属于生产 `lp_smart_agent.py`，会人为延长 SAFE 时间。报告中 `COOLDOWN_SKIP=8842`、LP time ratio 仅 `3.9%` 很可能主要来自这个偏差。

### F14 必须修改

1. 删除 `ACTIVE -> SAFE` 对 `last_rebalance` 的更新；
2. SAFE -> ACTIVE 不允许检查 4-day cooldown，必须立即 re-enter；
3. `COOLDOWN_SKIP` 对 Frozen Legacy 应删除或固定为 0；不得继续表示 SAFE re-entry cooldown；
4. 4-day 规则只用于 ACTIVE -> ACTIVE periodic rebalance；
5. Always LP 的 4-day periodic 规则保持当前独立基准定义，不受此修改影响。

### F14 必须新增 truth-table 单元测试

用 deterministic stub，不依赖真实市场：

```text
Case 1: ACTIVE at t0 -> SAFE at t1
assert last_rebalance unchanged
assert state == ETH/USDC/MIXED according to macro

Case 2: SAFE at t1 -> ACTIVE at t1+15m
assert immediate LP re-entry
assert SAFE_TO_ACTIVE += 1
assert last_rebalance == t1+15m

Case 3: ACTIVE from t0 to t0+3d23h45m
assert PERIODIC_REBALANCE == 0

Case 4: ACTIVE at t0+4d
assert PERIODIC_REBALANCE == 1

Case 5: SAFE remains SAFE while macro bull->bear changes
assert no second SAFE routing swap is triggered
```

必须在报告中输出修复前后：

```text
LP time ratio
ACTIVE_TO_SAFE
SAFE_TO_ACTIVE
COOLDOWN_SKIP
PERIODIC_REBALANCE
Frozen Legacy total return
```

---

## F15 — 模型/特征异常当前会静默变成“低风险”（CRITICAL，必须修）

当前代码：

```python
try:
    X = ...
    risk_prob = model.predict_proba(...)
except Exception:
    risk_prob = 0.0
```

`risk_prob=0.0` 等价于“模型认为风险最低”，这是 fail-open（失败时放行）行为，会把数据错误、特征错列、模型反序列化异常或 NaN 问题静默转换成更积极的 LP 信号。

### Iteration 4 固定要求

生产式回测采用 fail-fast：

1. 在进入 OOS 回测前，显式验证 `features` 完全存在；
2. 特征列顺序必须严格按 `models_15m.pkl['features']`；
3. 每次决策前的输入向量不得含 `NaN / inf / -inf`；
4. 若 OOS 正式窗口内任意决策发生：
   - missing feature
   - non-finite feature
   - `predict_proba` exception
   则整次正式回测必须 `raise RuntimeError` 并停止，禁止自动回退到 0.0；
5. warmup 不足导致的前置 NaN 应在进入 OOS 前通过 `dropna(subset=features + required regime columns)` 清理，而不是在策略内补 0。

### 必须新增测试

```text
- 删除一个模型特征 -> 必须 RuntimeError
- 模型 predict_proba 主动 raise -> 必须 RuntimeError
- 某特征 NaN -> 必须 RuntimeError 或该 timestamp 在进入回测前被明确剔除
- 正常 OOS 全窗口 anomaly_count == 0
```

结果 JSON 必须增加：

```json
"model_input_audit": {
  "required_feature_count": 19,
  "missing_feature_count": 0,
  "non_finite_decision_rows": 0,
  "predict_errors": 0
}
```

---

## F16 — Fee Ledger 字段语义错误，必须拆分 accrued / collected / uncollected

Iteration 3 的：

```python
cum_fee_eth = positive_diff(base_uncollected).sum()
cum_fee_usdc = positive_diff(quote_uncollected).sum()
```

这个量更接近**累计 accrued fee token（累计产生过的手续费 token）**，其中包含最终仍未领取的手续费。

但 `compute_lp_reconciliation()` 又把它命名为：

```text
collected_fee_eth
collected_fee_usdc
```

这会把“累计产生”误写成“累计已领取”。

### Iteration 4 固定字段

必须改为：

```text
fee_accrued_eth
fee_accrued_usdc
fee_uncollected_final_eth
fee_uncollected_final_usdc
fee_collected_eth = fee_accrued_eth - fee_uncollected_final_eth
fee_collected_usdc = fee_accrued_usdc - fee_uncollected_final_usdc
```

允许浮点误差，但必须满足：

```text
fee_collected >= 0
fee_accrued = fee_collected + fee_uncollected_final
```

分别对 ETH 与 USDC 成立。

禁止再把 accrued 总量直接标成 collected。

### Valuation 口径必须分开

报告至少同时给：

```text
fee_accrued_tokens: ETH / USDC
fee_accrued_value_at_final_price
fee_counterfactual_nav_impact = fee_on_nav - fee_off_nav
```

并明确说明：

`fee_accrued_value_at_final_price` 与 `fee_counterfactual_nav_impact` 不要求相等，因为手续费可能被再投入，路径与复利不同。

### 必须新增测试

构造 collect/reset 场景，严格断言：

```text
accrued = collected + final_uncollected
```

而不只是 `>=0`。

---

## F17 — Frozen Legacy 必须输出逐次状态转换审计表

仅看汇总次数不足以验证状态机。

Iteration 4 必须在结果中新增小型：

```text
results/r0_t002/post_freeze_oos_transitions.csv
```

这是本轮唯一新增允许文件。

每一行只记录**发生状态转换或 periodic rebalance 的时点**，不得上传分钟原始数据。字段固定：

```text
timestamp
old_state
new_state
is_active
macro_rsi
macro_bucket          # BULL / BEAR / NEITHER
risk_prob
rsi_15m
natr_15m
action                # EXIT_ETH / EXIT_USDC / EXIT_KEEP / REENTER_LP / PERIODIC_REBALANCE
last_rebalance_before
last_rebalance_after
pool_price
nav_before
nav_after
```

### 必须做三类自动检查

1. 每个 `SAFE -> ACTIVE` 行必须是 `REENTER_LP`；
2. 每个 `ACTIVE -> SAFE` 行的 `last_rebalance_before == last_rebalance_after`；
3. `PERIODIC_REBALANCE` 只能发生在 `ACTIVE -> ACTIVE`。

Harness Report 必须列出前 5 行、后 5 行摘要，但不能人工挑选结果来修改策略。

---

## F18 — Legacy-Cost 只能称为“旧启发式成本”，并做成本 ledger

当前 `Legacy-Cost` 使用旧 `latency_bias=5bps` 和 `exit deduction=0.0002`，但它不是实际 Gas、真实滑点或链上成交重放。

Iteration 4 不要求发明新的现实成本模型，但必须把旧假设记账清楚。

结果增加：

```text
legacy_cost_ledger:
  n_exit_swaps
  n_reentry_swaps
  n_periodic_rebalances
  total_exit_deduction_usdc
  total_latency_bias_effect_usdc
  total_legacy_cost_effect = gross_nav - legacy_cost_nav
```

若当前实现无法准确把 `latency_bias_effect_usdc` 拆分出来，允许字段为 `NOT_SEPARATELY_MEASURABLE`，但必须保留：

```text
gross_nav - legacy_cost_nav
```

并明确禁止把它描述为真实 Gas Cost。

---

# Iteration 4 Required Tests

除现有有效测试继续通过外，至少新增以下测试；不得用源码字符串检查替代行为测试：

1. `test_active_to_safe_does_not_touch_last_rebalance`
2. `test_safe_to_active_reenters_immediately`
3. `test_safe_to_safe_does_not_reroute`
4. `test_periodic_only_active_to_active`
5. `test_missing_model_feature_fails_fast`
6. `test_predict_exception_fails_fast`
7. `test_nonfinite_model_input_fails_fast`
8. `test_fee_accrued_equals_collected_plus_uncollected_eth`
9. `test_fee_accrued_equals_collected_plus_uncollected_usdc`
10. `test_transition_audit_invariants`
11. 原 F7/F8 exact OHLC/bar-boundary tests
12. 原 F9 deploy invariant
13. 原 F11 deterministic periodic test
14. 原 fee-on/off reconciliation test

正式 OOS 全量运行必须报告：

```text
all_tests_passed
model anomaly_count == 0
transition invariant violations == 0
NAV reconciliation error < 0.02 USDC
```

---

# Iteration 4 Mandatory Output Comparison

必须把 Iteration 3 与 Iteration 4 并排输出：

```text
Frozen Legacy Gross Return
Frozen Legacy Legacy-Cost Return
Always LP Gross Return
Always ETH Return
Frozen vs ETH excess
LP time ratio
ACTIVE->SAFE count
SAFE->ACTIVE count
COOLDOWN_SKIP count
PERIODIC_REBALANCE count
fee accrued ETH
fee accrued USDC
fee counterfactual NAV impact
```

重点回答：

1. 去掉错误的 SAFE re-entry 4-day cooldown 后，Frozen Legacy +16.34% 是否仍成立？
2. LP time ratio 是否从 3.9% 显著变化？
3. Frozen 相对 ETH 的 -3.29% 是否改变方向？
4. 新状态机下 drawdown / Sharpe 是否改善或恶化？
5. Always LP -5.83% 是否保持不变（理论上 F14 不应影响 Always LP；若变化必须解释）。

---

# Allowed Files — Iteration 4

仅允许修改：

- `research/r0_t002_post_freeze_oos.py`
- `tests/test_r0_t002_post_freeze_oos.py`
- `results/r0_t002/post_freeze_oos.json`
- `results/r0_t002/post_freeze_oos.md`
- `results/r0_t002/post_freeze_oos_equity.csv`
- `results/r0_t002/post_freeze_oos_transitions.csv`（本轮新增，只有 transition/event 行）
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止修改：

- `lp_smart_agent.py`
- 任何 legacy 策略文件
- README
- 模型文件
- 协议
- 本地原始数据
- 其他辅助模块

如果现有两个 Python 文件无法完成，报告 `BLOCKED`，不得自行扩大 scope。

---

# Mandatory Execution Order

严格按以下顺序：

1. 先写/修 F14 状态机 truth-table 测试；
2. 修 Frozen Legacy 状态机直到 F14 tests PASS；
3. 写 F15 fail-fast tests；
4. 修 model input audit；
5. 写 F16 fee identity tests；
6. 修 fee ledger 字段；
7. 写 F17 transition audit 与 invariant tests；
8. 补 F18 cost ledger；
9. 运行全部 unit tests；
10. 只有全部测试通过，才运行完整 2026-03-14..2026-08-21 OOS；
11. 生成 JSON / Markdown / equity / transitions；
12. 再跑一次全部 tests，保证结果 schema 与测试一致；
13. 更新 HARNESS_REPORT；
14. `state=REVIEW_READY, owner=architect, authorized_next=[]` 后提交。

禁止在正式回测结果出来后修改阈值、range、状态逻辑或成本参数。

## Next State

- decision: `CHANGES_REQUIRED`
- task_id: `R0-T002`
- next iteration: `4`
- owner: `harness`
- authorized_next: `[R0-T002]`
