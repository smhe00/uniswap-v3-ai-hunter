# Architect Review — R0-T002 Iteration 1

## Decision

`CHANGES_REQUIRED`

R0-T002 当前不能 PASS。Harness 已完成一套可运行的后冻结样本外回测框架，但本轮存在多项会实质改变收益结论的问题，因此报告中的 `Frozen Legacy +21.38%`、`Always LP -0.04%` 以及相对优势都只能视为 **provisional（暂定结果）**，不得作为策略有效性证据。

## Reviewed Snapshot

- remote_head: `b72c11c18ff6d5ef8bdfdad431699ad00008e4cb`
- task_id: `R0-T002`
- iteration reviewed: `1`
- consumed harness handoff: `R0-T002-HARNESS-20260823-001`

## Findings

### F1 — 未优先复现生产 Binance 信号，且未按任务要求分栏（必须修）

CURRENT_TASK §5.2 明确要求：生产 `lp_smart_agent.py` 使用 Binance `ETHUSDT` 15 分钟 / 4 小时行情，应优先复现生产信号；只有历史结构无法精确复现时才允许增加 Pool-derived signal（池价派生信号）对照，而且必须与 Production-like Binance signal（生产近似 Binance 信号）分栏报告。

本轮报告明确写明：

> `BINANCE_KDATA ... 未使用——池价路径足够`

同时 Known Limitations 又承认“未单独分栏”。这不满足任务要求。Iteration 2 必须：

1. 使用本地 Binance 历史数据生成主结果；
2. Pool-derived 结果只允许作为辅助对照；
3. 两套结果在 JSON / Markdown 中分栏，禁止混写；
4. 如果 Binance 历史数据确实无法构造，则报告 `BLOCKED` 并说明缺失字段/周期，不得自行把池价结果当生产结果。

### F2 — 4 小时信号存在明确未来数据泄漏（必须修）

当前实现：

```python
s4 = price.resample("4h").agg(["last", "max", "min"]).dropna()
...
feat = feat.join(macro.reindex(feat.index, method="ffill"))
```

Pandas 默认把 00:00–03:59 这一根 4 小时 bar 标记为 00:00。随后 `ffill` 会让 00:14、00:29 等决策时点看到只有到 03:59 才能完整确定的 close/high/low，属于 look-ahead（未来数据泄漏）。

Iteration 2 必须保证：

- 15 分钟信号只在该 15 分钟 bar 完整收盘后可见；
- 4 小时信号只在该 4 小时 bar 完整收盘后可见；
- 推荐使用 `label='right', closed='right'` 或等价显式 shift，使信号时间戳代表“可用时间”，而不是 bar 起始时间；
- 新增合成数据测试：让 4 小时 bar 最后一小时出现极端价格变化，bar 收盘前的决策不得看到该变化。

在修复前，本轮不能称为 strict OOS（严格样本外、严格因果）。

### F3 — 技术指标数学与生产 `pandas_ta` 不一致，尤其 NATR 尺度错误（必须修）

生产 `lp_smart_agent.py` 使用 `pandas_ta` 直接计算 RSI / ADX / NATR / Bollinger Bands；当前回测重新手写了 `_rsi/_adx/_natr`。

其中当前 `_natr` 返回：

```python
ATR / close
```

而生产 `pandas_ta.natr` 的标准输出是百分比尺度（通常为 `100 * ATR / close`）。旧 GA 阈值 `1.587...` 与 `VOL_GUARD_NATR = 2.0` 都是按生产尺度冻结的。当前实现把 NATR 缩小约 100 倍再送入 GA / XGBoost，会改变 active/safe 判定和模型输入分布。

Iteration 2 要求：

1. 生产主结果必须直接使用与 `lp_smart_agent.py` 同口径的 `pandas_ta` 指标；
2. feature 名称、列顺序、lag 语义必须与冻结模型一致；
3. 新增 parity test：在同一历史片段上，研究脚本生成的 RSI / ADX / NATR / bb_width 与生产式 `pandas_ta` 结果逐列对比；
4. NATR 必须显式验证尺度与旧阈值一致。

### F4 — LP 初始资本分配存在单位错误，导致 Always LP 基准几乎全部资金闲置（必须修）

当前 `_ensure_eth_for_lp()` 在“USDC 买 ETH”分支中：

```python
need_usdc = (target_eth_val - eth_val) / float(ps.price)
swap_amt = min(need_usdc, float(usdc_asset.balance))
self.broker.swap_by_from(self.usdc, self.eth, ...)
```

`swap_by_from(self.usdc, ...)` 的输入数量单位是 USDC，但 `target_eth_val - eth_val` 已经是 USDC 价值，再除 ETH 价格后变成了 ETH 数量级。举例：目标把 5,000 USDC 换成 ETH、ETH=2,500 USDC 时，代码实际只换约 2 USDC。

这会使绝大多数 10,000 USDC 留在钱包，只有极小部分进入 LP。它与本轮异常结果高度一致：

- Always LP 在 ETH 同期约 +20% 时仍接近 0%；
- LP Fee 只有约 0.03 USDC；
- LP 基准表现几乎等同现金。

因此当前 `Always LP -0.04%` 不能作为有效基准。

Iteration 2 不应只机械把 `/ price` 删除，而应做正确的 Uniswap V3 range 资产配比：

1. 根据当前价格和 `±8.13%` range 计算可部署全部资本时所需 ETH / USDC 比例；
2. 建仓后记录 `deployed_value` 与 `idle_wallet_value`；
3. 在正常居中建仓时，除舍入误差外不得长期留下大额闲置资金；
4. 新增 invariant test：10,000 USDC 初始资金在中心价格建仓后，闲置价值占比应小于 1%（如 Demeter 约束导致更高，必须解释并给出数值）；
5. Frozen Legacy 与 Always LP 必须使用同一套 LP 资本部署函数。

### F5 — Frozen Legacy 未复现生产脚本的 4 天周期再平衡语义（必须修）

生产 `lp_smart_agent.py` 在持续 ACTIVE 状态时有：

```text
elif days_since_reb >= REBALANCE_DELAY_DAYS:
    action = "PERIODIC REBALANCE (4-Day Rule)"
```

而当前 `FrozenLegacyStrategy` 在 ACTIVE 且已有 position 时只继续持有，不会每 4 天重建 / recenter。这样冻结策略并未真正复现生产语义。

Iteration 2 必须：

- ACTIVE 连续超过 4 天时执行与冻结语义一致的周期再平衡；
- 记录 periodic rebalance 次数；
- 新增测试证明第 4 天以前不触发，第 4 天后触发一次，并更新 `last_rebalance`。

### F6 — 累计 LP Fee 指标不满足任务要求（必须修）

任务要求输出“累计 LP Fee，流动性手续费收入”。当前 `acc_fees` 实际只读取最终时点的 `base_uncollected / quote_uncollected`，而 Harness 也承认：再平衡时 fee 已转入余额，所以该值会漏掉已实现手续费。

Iteration 2 必须输出真正累计手续费：

- 已实现 fee（remove / collect 时进入余额的 fee）；
- 加上最终未领取 fee；
- 或使用 Demeter 可审计 action / status 累加得到等价结果。

并增加至少一个已知有成交、LP 在区间内的短窗口测试，确认累计 fee 单调非减且大于 0。

## Test Coverage Gap

本轮 `14 passed` 不能证明上述关键点已满足。现有 `TestBackwardMerge` 只检查源码字符串中存在 `ffill`，并没有验证 bar close 的真实时间因果性；也没有验证 LP 资金部署比例、NATR 尺度、生产 Binance 信号、4 天周期再平衡或累计 fee。

因此测试数量本身不能作为本轮通过依据。

## Accepted Parts

以下基础工作可以保留：

1. OOS 窗口固定为 2026-03-14 至 2026-08-21；
2. 未重新优化 `±8.13% / 0.57 / 4 天 / RSI 52/50 / NATR 2.0`；
3. 本轮没有链上写路径；
4. ETH / USDC / 50-50 简单持仓基准的计算框架可以继续使用；
5. Gross / Legacy-Cost 分离的结构可以继续使用；
6. 当前结果文件与每日净值曲线可保留作为 iteration 1 调试证据，但不得视为有效绩效结论。

## Iteration 2 Narrow Fix Scope

保持 `task_id = R0-T002`。允许修改：

- `research/r0_t002_post_freeze_oos.py`
- `tests/test_r0_t002_post_freeze_oos.py`
- `results/r0_t002/post_freeze_oos.json`
- `results/r0_t002/post_freeze_oos.md`
- `results/r0_t002/post_freeze_oos_equity.csv`
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止修改 legacy 策略文件、README、模型、协议和原始数据。

## Required Validation for Iteration 2

至少必须新增并通过以下验证：

1. Binance Production-like 信号覆盖完整 OOS 窗口，并报告数据文件/行数/缺口；
2. 15m / 4h bar-close causality test；
3. `pandas_ta` 特征 parity test，特别是 NATR 尺度；
4. LP deploy-capital invariant test（idle capital <1% 或明确解释）；
5. Frozen Legacy 4-day periodic rebalance test；
6. cumulative LP fee test；
7. 原任务已有测试继续通过；
8. 重新跑完整 OOS，并把 Production-like Binance 结果与 Pool-derived 对照分栏。

Harness Report 必须明确回答：修复后 `+21.38%` 是否仍成立、Always LP 是否仍接近 0%、Frozen Legacy 相对 ETH 的 +0.90% 是否仍存在，以及结论变化主要来自哪项修正。

## Next State

- decision: `CHANGES_REQUIRED`
- task_id: `R0-T002`
- next iteration: `2`
- owner: `harness`
- authorized_next: `[R0-T002]`
