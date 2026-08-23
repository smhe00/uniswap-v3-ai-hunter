# Architect Review — R0-T002 Iteration 2

## Decision

`CHANGES_REQUIRED`

Iteration 2 修复了上一轮的主要方向性问题，但本轮仍不能 PASS。原因不是文档瑕疵，而是至少 5 个会直接影响收益、状态切换或指标可信度的实现/测试问题。

本轮以后，DeepSeek Harness 视为受控执行器。不得自行改设计，不得凭经验“等价替代”；只按本 Review 的明确步骤修复并重跑。

## Reviewed Snapshot

- remote_head: `38d05a4ba55e1b6701dd5ccc9da57d4de5ac56e8`
- task_id: `R0-T002`
- iteration reviewed: `2`
- consumed harness handoff: `R0-T002-HARNESS-20260823-003`

## Accepted Improvements

以下改动方向正确，可以保留：

1. 主结果改用 Binance 数据，Pool-derived 结果降为对照；
2. NATR 使用 pandas_ta 百分比尺度；
3. LP 资本部署不再使用上一轮错误的 `USDC_value / ETH_price` 作为 USDC swap 数量；
4. Frozen / Always LP 共用同类资本部署逻辑；
5. 增加 Gross / Legacy-Cost 分栏；
6. 增加结果文件、每日净值和事件统计；
7. 未修改 legacy 策略、模型、README 或原始数据。

但当前 `+23.91%`、Always LP `-5.83%`、累计 fee `2976.94 USDC` 仍为 provisional（暂定结果），不得作为最终策略有效性证据。

---

## F7 — Binance 1m -> 15m 输入并未复现生产 OHLC（必须修）

当前 `load_binance_ethusdt_1m()` 只读取：

```python
cdf = cdf[[0, 4]]
cdf.columns = ["ts", "close"]
```

随后 `compute_signals_from_price()` 用 1 分钟 close 的 `max/min` 生成 15 分钟 high/low：

```python
price.resample(...).agg(["last", "max", "min"])
```

这不是生产 `lp_smart_agent.py` 使用的 Binance 15m OHLC。生产指标 ADX / NATR 依赖真实 high / low；“1m close 的最大/最小”会系统性漏掉分钟内部 wick，高低点偏窄，从而改变：

- NATR = Normalized Average True Range，归一化平均真实波幅；
- ADX = Average Directional Index，平均趋向指数；
- DMP / DMN；
- XGBoost 模型输入。

### Iteration 3 唯一允许的做法

优先级固定：

1. **优先直接读取 Binance 官方历史 15m 和 4h kline 文件**，如果本地目录存在；
2. 若本地没有 15m/4h 文件，只能从 1m OHLC 聚合，必须读取完整：
   - open
   - high
   - low
   - close
3. 1m -> 15m 聚合公式必须固定：

```text
open  = first(open)
high  = max(high)
low   = min(low)
close = last(close)
```

4. 4h 同理；
5. 禁止再从单一 close 序列构造 high/low。

### 必须新增测试

构造 1 分钟数据，其中某一分钟 `high=3000, close=2100`。聚合后的 15m high 必须是 3000，不得是 2100。

如果有本地原生 Binance 15m/4h 数据，必须随机抽至少 100 根 bar，将“1m 聚合结果”与“原生 15m/4h”逐列比对并报告：open/high/low/close 最大绝对差与不一致根数。

---

## F8 — bar 时间戳仍有 1 分钟未来泄漏风险（必须修）

Binance 历史 1m kline 的时间戳字段是 **open time**。例如 `00:15:00` 这一行代表 `[00:15:00, 00:15:59.999...]` 这一分钟。

当前：

```python
resample("15min", label="right", closed="right")
```

会把 `00:15:00` 这一行包含进标记为 `00:15` 的 bar。于是 `00:15` 决策可能使用到 `00:15:59` 才最终确定的 close/high/low。

### Iteration 3 固定时间语义

必须明确采用“bar close available time”。若输入索引是 1m open time，推荐二选一：

**方案 A（优先）：**

先把每个 1m kline 映射为其 close availability time，再聚合。

**方案 B：**

使用左闭右开窗口：

```text
15m bar = [00:00, 00:15)
available_at = 00:15
```

使 `00:15` 行本身不进入上一根 15m bar。

### 必须新增精确测试

构造：

```text
00:00..00:14 close = 100
00:15 close = 1000
```

断言：

- available_at `00:15` 的上一根 15m bar close 必须仍是 `100`；
- `1000` 只能进入下一根 bar；
- 4h 同理构造 `04:00` 边界测试。

现有“尖峰前后 RSI 不相等”测试不够严格，不能替代该边界测试。

---

## F9 — LP capital deployment invariant 测试公式错误（必须修）

当前 `run_backtest()`：

```python
deployed_value = (base_in_pos + eth_bal) * last_p + (quote_in_pos + usdc_bal)
idle_value = eth_bal * last_p + usdc_bal
```

这里 `deployed_value` 实际等于“仓位 + 钱包”，不是“仓位部署价值”。

测试又计算：

```python
total = idle_value + deployed_value
idle_ratio = idle_value / total
```

导致钱包闲置价值在分母中被重复计算一次，因此 idle ratio 会被人为压低。

### Iteration 3 固定定义

必须改成：

```text
position_value = base_in_position * ETH_price + quote_in_position
idle_wallet_value = wallet_ETH * ETH_price + wallet_USDC
total_nav_components = position_value + idle_wallet_value + uncollected_fee_value
idle_ratio = idle_wallet_value / total_nav_components
```

字段名称也必须改成 `position_value`，禁止继续把 total holdings 叫 `deployed_value`。

### 必须新增 invariant

在**刚完成居中 LP 建仓的那个时点**记录，而不是只看回测最后时点：

- `idle_ratio < 1%`；
- `position_value > 98% * pre_deploy_nav`；
- 允许剩余为 tick rounding / amount rounding；
- 该 invariant 必须同时对 Frozen Legacy 与 Always LP 使用同一 helper。

报告中 `9353.14 deployed / 13.76 idle` 目前定义不可信，必须重新计算。

---

## F10 — cumulative fee 算法会混入 ETH 价格变化，不是严格手续费累计（必须修）

当前：

```python
fee_series = base_uncol * eth_price + quote_uncol
acc_fees = fee_series.diff().clip(lower=0).sum()
```

问题：`base_uncol` 是 ETH 数量，乘当前 ETH price 后，fee_series 会因为 **ETH 价格上涨** 而上升，即使这一分钟没有新增任何 fee。对所有正增量求和会把手续费 token 的 mark-to-market（价格变动）误计为手续费收入。

此外，remove / collect 导致 uncollected 清零后，负跳变被 clip 掉；虽然这能避免扣掉历史 fee，但不能解决 ETH 价格重估混入的问题。

因此当前报告的 `Always LP cumulative fee = 2,976.94 USDC` 不能直接接受。

### Iteration 3 固定实现要求

必须按 token 数量累计，不允许先换成 USDC 再对正增量求和。

至少维护：

```text
cum_fee_eth
cum_fee_usdc
```

每一步从 `base_uncollected` / `quote_uncollected` 的 token 数量变化中识别新增 fee；发生 collect/remove reset 时不能丢失历史累计。

最终报告同时输出：

```text
cum_fee_eth
cum_fee_usdc
cum_fee_value_at_earning_time_or_explicit_valuation_method
final_uncollected_eth
final_uncollected_usdc
```

如果 Demeter action log / position fee 字段可以直接给出 collect 数量，优先用可审计 action log：

```text
累计 fee = 所有已 collect token fee + 最终 uncollected token fee
```

### 必须新增两个测试

1. **零成交 / 无新增 fee，但 ETH 价格上涨**：累计 fee 必须保持 0；
2. **先累计 fee -> collect/reset -> 再累计**：累计 fee 必须单调增加且等于两段 fee token 数量之和。

禁止用 `fee_value.diff().clip(lower=0)` 作为累计手续费。

---

## F11 — Periodic Rebalance 测试是空断言（必须修）

当前测试：

```python
assert res["events"]["PERIODIC_REBALANCE"] >= 0
```

这个断言永远成立，不能验证任何东西。

### Iteration 3 必须实现 deterministic unit test

不要依赖真实 11 天市场数据碰运气。

用最小 stub / fake market 明确驱动时间：

```text
t0             : 建仓
t0 + 3d23h45m  : 不得周期再平衡
t0 + 4d00h00m  : 应触发 1 次 periodic rebalance
t0 + 4d00h15m  : 不得再次触发
```

必须断言：

```text
PERIODIC_REBALANCE == 1
last_rebalance == t0 + 4d
```

Frozen Legacy 和 Always LP 都要覆盖；若 Frozen 因 ACTIVE/SAFE 状态需要 stub signal，则固定为持续 ACTIVE。

---

## F12 — 必须做 LP PnL reconciliation，验证 Always LP -5.83% 是否合理

Iteration 2 修复资本部署后，Always LP 从约 0% 变为 `-5.83%`，同时声称累计 fee 约 `+29.77%` 初始资本。这个组合并非不可能，但在接受前必须进行独立经济核对。

Iteration 3 必须输出一个 `LP Reconciliation` 表，至少包括：

```text
start_nav
end_nav
wallet_eth_value
wallet_usdc
position_principal_value
uncollected_fee_eth
uncollected_fee_usdc
cumulative_collected_fee_eth
cumulative_collected_fee_usdc
number_of_rebalances
number_of_out_of_range_rebuilds
number_of_periodic_rebalances
```

并增加一个 **fee-disabled counterfactual**（仅研究检查，不属于策略优化）：

- 相同价格路径；
- 相同 range；
- 相同 rebalance 时点；
- 手续费收入置 0；

输出：

```text
Always LP with fee
Always LP without fee
difference
```

两者差值应与累计 fee 的价值口径大体一致；若差异明显，必须说明原因。

该 counterfactual 只用于数值 sanity check（合理性检查），不得用于调参。

---

## F13 — Production parity 必须升级为“输入 + 特征”双层验证

当前 parity test 只是同一研究脚本 resample 后再次调用 pandas_ta，因此只能证明库调用一致，不能证明与生产输入一致。

Iteration 3 必须分两层：

### Layer 1 — OHLC parity

确认研究回测生成的 15m/4h OHLC 与 Binance 原生历史 bar 一致；如果没有原生 bar，则明确写 `NOT_AVAILABLE`，但必须至少验证 1m OHLC 聚合公式。

### Layer 2 — Feature parity

在同一 OHLC DataFrame 上，对生产式计算与研究式计算逐列比较：

- RSI_14
- ADX_14
- ADXR_14_2
- DMP_14
- DMN_14
- NATR_14
- bb_width
- lag1 / lag2 / lag4

必须报告最大绝对差，目标 `< 1e-9`（若 pandas 版本造成浮点差异，可放宽到 `<1e-7`，但要写明）。

---

## Iteration 3 Allowed Files

仅允许修改：

- `research/r0_t002_post_freeze_oos.py`
- `tests/test_r0_t002_post_freeze_oos.py`
- `results/r0_t002/post_freeze_oos.json`
- `results/r0_t002/post_freeze_oos.md`
- `results/r0_t002/post_freeze_oos_equity.csv`
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止修改其他文件。禁止新建辅助模块。若确实无法在上述文件完成，报告 `BLOCKED`，不要自行扩 scope。

---

## Iteration 3 Mandatory Execution Order

Harness 必须按以下顺序执行，不得跳步：

1. 修 F7：完整 OHLC 输入；
2. 修 F8：精确 bar available-time；
3. 跑 OHLC + feature parity；
4. 修 F9：position / idle / NAV component 定义；
5. 修 F10：token-level cumulative fee；
6. 修 F11：deterministic periodic rebalance tests；
7. 先跑单元测试；
8. 单元测试全部 PASS 后，才允许跑完整 2026-03-14..2026-08-21 OOS；
9. 完整回测后做 F12 LP reconciliation；
10. 最后生成 JSON / Markdown / equity CSV 和 Harness Report。

禁止在测试失败时先跑完整回测。

---

## Mandatory Tests for Iteration 3

至少必须新增或强化：

1. 1m OHLC -> 15m wick preservation；
2. 15m exact boundary causality；
3. 4h exact boundary causality；
4. OHLC parity（若原生 15m/4h 可用）；
5. feature parity 全 19 个冻结特征；
6. post-deploy idle ratio 真正 `<1%`；
7. Frozen periodic rebalance deterministic test；
8. Always LP periodic rebalance deterministic test；
9. fee price-move-no-trade test；
10. fee collect-reset-continue test；
11. LP reconciliation identity / sanity check；
12. 原任务和 iteration 2 的有效测试继续通过。

任何“`>= 0`”“字段存在”“源码包含某字符串”之类空断言，不计入验收。

---

## Harness Report Mandatory Answers

Iteration 3 报告必须逐项回答，禁止省略：

1. Binance 输入使用原生 15m/4h，还是 1m OHLC 聚合？
2. 如果是聚合，OHLC 与原生历史 bar 的 parity 结果是什么？
3. 15m / 4h available-time 的精确定义是什么？
4. 修复 F8 后是否仍存在任何 bar boundary look-ahead？
5. 正确 idle_ratio 是多少？
6. 正确累计 fee token 数量分别是多少 ETH / USDC？
7. `2976.94 USDC` 旧 fee 数字是否成立？
8. Always LP `-5.83%` 是否仍成立？
9. Frozen Legacy `+23.91%` 是否仍成立？
10. Frozen Legacy 相对 Always ETH 的 `+3.01%` 是否仍成立？
11. fee-disabled counterfactual 与 fee-on 的差值是多少？
12. 哪个修正对结果影响最大？

---

## Next State

- decision: `CHANGES_REQUIRED`
- task_id: `R0-T002`
- next iteration: `3`
- owner: `harness`
- authorized_next: `[R0-T002]`
