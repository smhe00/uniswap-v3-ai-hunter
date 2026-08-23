# Harness Report

> 协议：UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md（第 10 节标准结构）
> 报告人：DeepSeek Harness（本地执行 Agent）

## 1. Task Identity（任务标识）

- task_id: **R0-T002**
- iteration: **3**（Architect CHANGES_REQUIRED -> 修复 F7-F13）
- consumed_handoff_id: **R0-T002-ARCH-20260823-003**
- base_remote_head: `38d05a4ba55e1b6701dd5ccc9da57d4de5ac56e8`（Architect Review 的 git_base_commit）
- result_commit: `PENDING_SELF`（随 commit 推送，最终 SHA 以远端 remote_head 为准）

## 1.1 Architect Review 处置摘要（Iteration 2 -> 3）

| Finding | 处置 |
| :-- | :-- |
| F7 1m→15m 未复现生产 OHLC | `load_binance_ethusdt_1m` 改读完整 `[open,high,low,close]`；新增 `aggregate_ohlc`：`open=first, high=max, low=min, close=last`；Pool 对照也用 tick 派生 OHLC（`load_pool_ohlc`）。本地无原生 15m/4h 文件，按 F7 优先级 2 用 1m OHLC 聚合。单测覆盖：1m 中 1 分钟 high=3000/close=2100 -> 15m high=3000 |
| F8 bar 时间戳 1 分钟未来泄漏 | 方案 A：每个 1m kline 先映射为 close availability time（open_time+1min）再聚合；bar 时间戳=完全可用时刻。单测覆盖精确边界：00:00..00:14 close=100、00:15 close=1000 -> 前 bar close=100、1000 只进下一 bar（15m 与 4h 均测） |
| F9 LP 资本部署 invariant 公式错误 | 修正定义：`position_value=base_in_pos*price+quote_in_pos`、`idle_wallet=wallet_ETH*price+wallet_USDC`、`total_nav_components=position+idle+uncollected_fee`。策略在建仓时点记录 `deploy_snapshots`（position/idle/NAV 组件），单测断言 idle_ratio<1% |
| F10 累计 fee 被价格重估污染 | 改为 token 数量级：分别对 `base_uncollected`/`quote_uncollected` 序列取 positive diff 累加（ETH/USDC 各一份），不做价格重估。因 demeter `CollectFeeAction` 含 remove 本金，纯手续费取 uncollected diff。单测覆盖 fee token 单调非负 + collect-reset-continue |
| F11 periodic rebalance 测试空断言 | 新增 deterministic 单测：合成 5 天恒定 in-range 数据，断言 t0..t0+3d 不重建、t0+4d 恰好 1 次重建、last_rebalance 更新（Frozen Legacy 与 Always LP 均覆盖） |
| F12 LP PnL reconciliation 缺失 | 新增 `compute_lp_reconciliation`（NAV=position+idle+uncollected_fee 对账表 + 动作统计）；新增 `compute_fee_counterfactual`：fee-disabled（fee_rate=0）回测隔离手续费贡献。单测覆盖对账 identity、fee-off NAV<=fee-on、rebalance 时点一致 |
| F13 parity 双层 | Layer 1 OHLC parity：本地无原生文件 -> 报告聚合公式 + 单测；Layer 2 feature parity：单一 `compute_signals_from_ohlc` pandas_ta 路径 + 特征列清单。报告含 `parity` 段 |

## 2. Changed Files（变更文件）

| 文件 | 说明 |
| :-- | :-- |
| `research/r0_t002_post_freeze_oos.py` | Iteration 3 重写：1m OHLC 加载/聚合 + available-time 语义 + deploy 快照 + token 级 fee + reconciliation + fee-disabled counterfactual + parity 报告 |
| `tests/test_r0_t002_post_freeze_oos.py` | 35 个测试（F7/F8 精确边界、F9 invariant、F10 token fee、F11 deterministic、F12 reconciliation + 全部继承测试） |
| `results/r0_t002/post_freeze_oos.json` | 重新生成（iteration 3，含 mandatory_answers/parity/reconciliation） |
| `results/r0_t002/post_freeze_oos.md` | 重新生成 |
| `results/r0_t002/post_freeze_oos_equity.csv` | 重新生成（7 列每日净值） |
| `work/handoff/HARNESS_REPORT.md` | 本报告 |
| `work/control/WORKFLOW_STATE.yaml` | state -> REVIEW_READY |

未修改：legacy 策略文件、README、模型文件、协议文件、本地原始数据。
环境说明：`.local/pandas_ta_pkg/` 与 `.local/demeter_pkg/` 均为本地私有（git-ignore，未提交），属任务 §15 允许的本地依赖准备。

## 3. Local Data Used（本地数据证据表，协议 §9.2）

| 项目 | 说明 |
| :-- | :-- |
| **Dataset Name** | Binance spot ETHUSDT 1m klines（主信号源）+ UNIV3_DATA 池 minute.csv（回测价格/流动性） |
| **Coverage Start / End** | Binance：2026-01-28..2026-08-21（45 天 warmup + OOS）；池：OOS 231,725 分钟行 |
| **File Count** | Binance 1m zip 206 个（每日齐全，无缺口）；池 OOS 每日齐全 |
| **Row / Swap Count** | Binance 296,640 行 OHLC；池 231,725 分钟行 |
| **Input Pattern** | `ETHUSDT-1m-*.zip`（微秒时间戳自动检测）+ `arbitrum-0xc696...-*.minute.csv` |
| **Data Gaps** | 无（加载时校验每日文件存在） |
| **Code Commit** | 本任务代码（本地 HEAD） |
| **Command** | 见 §4 |
| **Environment** | Python 3.12.10；pandas 2.3.3 / numpy 2.2.6 / xgboost 3.2.0 / demeter + pandas_ta（旧项目包）；pytest 9.0.2 |
| **Result Metrics** | 见 §6 |
| **Artifacts** | post_freeze_oos.json / .md / .csv |
| **Known Limitations** | 见 §9 |

## 4. Commands Executed（执行命令）

```bash
# 完整 OOS 回测（venv Python 3.12.10，含 xgboost）
PYTHONIOENCODING=utf-8 PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\venv\Scripts\python.exe" research/r0_t002_post_freeze_oos.py
# -> exit 0，输出 results/r0_t002/*（8 次回测：Binance Gross/Cost + Always + 2×fee-off + Pool 对照）

# 测试（v3.12 环境，pytest 9.0.2）
PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\v3.12\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_r0_t002_post_freeze_oos.py
# -> 35 passed
```

## 5. Test Results（测试结果）

`35 passed in 89.44s`（pytest 9.0.2）

覆盖任务 §14 全部 10 项 + Architect Review F7-F13 全部 Required Validation：
1. ✅ OOS 起始固定（TestOOSWindow）
2. ✅ 冻结参数一致（TestFrozenParams）
3. ✅ 无优化器（TestNoOptimizer）
4. ✅ **F7 OHLC aggregation**（TestF7OhlcAggregation：high=3000 进入 15m high + 公式四字段 + 4h + load 返回四列）
5. ✅ **F8 exact boundary**（TestF8BarBoundary：00:15 close=1000 只进下一 bar，15m 与 4h 均测 + spike causality 继承）
6. ✅ **F3 parity**（TestPandasTaParity：NATR 百分比尺度 + ADXR_14_2）
7. ✅ 初始净值一致（TestInitialCapital）
8. ✅ schema 稳定（TestSchema：iteration==3 + fixes 含 F7-F13 + mandatory_answers/parity/reconciliation 键）
9. ✅ 无链上写（TestNoOnchainWrite）
10. ✅ **F9 deploy invariant**（TestF9DeployInvariant：建仓时点 idle_ratio<1% + position/idle/NAV 对账）
11. ✅ **F10 token fee**（TestF10TokenFee：token 数量非负 + collect-reset-continue + ETH/USDC 分计）
12. ✅ **F11 deterministic rebalance**（TestF11PeriodicRebalance：t0..t0+3d 不重建、t0+4d 恰 1 次、last_rebalance 更新；Frozen 与 Always LP 各覆盖）
13. ✅ **F12 reconciliation**（TestF12Reconciliation：NAV identity + fee-off<=fee-on + rebalance 时点一致 + compute 结构）
14. ✅ 基准指标正确（TestMetrics）

## 6. Backtest / Validation Results

### 策略指标对比（主结果：Binance 1m OHLC 生产近似信号）

| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |
| :-- | --: | --: | --: | --: | --: | --: |
| A. Frozen Legacy (Gross, Binance) | **11634.10** | **+16.34%** | 40.96% | -29.68% | 1.0142 | 1.6736 |
| A. Frozen Legacy (Legacy-Cost, Binance) | 11589.01 | +15.89% | 39.72% | -29.89% | 0.9918 | 1.6436 |
| B. Always LP (Gross, Binance) | 9417.41 | **-5.83%** | -12.73% | -26.08% | -0.3489 | -0.3395 |
| A. Frozen Legacy (Gross, Pool 对照) | 11616.74 | +16.17% | 40.48% | -29.72% | 1.0083 | 1.6599 |
| C. Always ETH | 12029.67 | +20.30% | 52.07% | -38.67% | 1.0025 | 1.755 |
| D. Always USDC | 10000.0 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 |
| E. 50/50 Buy-and-Hold | 11014.83 | +10.15% | 24.51% | -20.92% | 0.9178 | 1.6091 |

### Frozen Legacy 超额收益（Binance 主信号）
- vs Always LP: **+23.54%**
- vs Always ETH: **-3.29%**（Iteration 2 +3.01% 反转，OHLC 修复后不再跑赢 ETH）
- vs Always USDC: +16.34%
- vs 50/50 Buy-and-Hold: +5.62%

### 事件统计（Frozen Legacy, Binance，15,448 次决策）
| 事件 | 次数 |
| :-- | --: |
| ACTIVE -> SAFE | 38 |
| SAFE -> ACTIVE | 38 |
| SAFE 进入 ETH | 18 |
| SAFE 进入 USDC | 19 |
| SAFE Keep Ratio | 1 |
| 4 天冷却阻止重建 | 8842 |
| 周期再平衡 | 0（LP 时间短于 4 天，从未触发） |

### 资本部署与 token 级手续费（F9/F10 修复后）
- Always LP：position 价值 9,403.65 / 钱包闲置 13.76 USDC（idle_ratio_final 0.146%）；建仓时点 max idle_ratio **0.73% < 1%**；累计 fee **ETH 0.7039 + USDC 1,387.00（= 3,157.22 USDC 按最终价折算）**
- Frozen Legacy：累计 fee **ETH 0.0276 + USDC 50.80（= 120.18 USDC）**；建仓时点 max idle_ratio **0.80% < 1%**
- Iteration 2 的 2,976.94 USDC（价格重估口径）被 token 数量口径（3,157.22）取代，差异来自 OHLC 修复后的信号路径

### F12: fee-disabled counterfactual（手续费贡献）
| 策略 | fee_on NAV | fee_off NAV | fee 贡献 | 占 fee_on |
| :-- | --: | --: | --: | --: |
| Always LP | 9,417.41 | 6,948.70 | **2,468.70** | **26.21%** |
| Frozen Legacy | 11,634.10 | 11,503.89 | **130.21** | 1.12% |

Always LP 的 2,468.70 USDC 手续费贡献说明：-5.83% 是"手续费收入 26% 仍无法覆盖 ±8.13% 区间在 +20% 行情下的无常损失/重建成本"的真实净结果。

### 信号源对照（F1）
- Binance 主信号：+16.34%；Pool 对照：+16.17%（差异 0.17pp，OHLC 修复后两信号源高度一致）

## 7. Failure / Edge Cases

- Binance 2025+ 新格式 zip 时间戳为**微秒**（旧格式毫秒）-> 加载时自动检测单位
- demeter `fee_rate=0` 会因 `tick_spacing=int(fee*200)=0` 除零 -> fee-disabled 用 fee=0.05 构造池再仅清零 `pool_info.fee_rate`
- demeter `CollectFeeAction.base_amount` 含 remove 后本金（非纯手续费）-> F10 改用 `base_uncollected/quote_uncollected` 序列 positive diff
- pandas_ta 短数据返回 None -> `_safe` 兜底 NaN
- v3.12 测试环境无 xgboost -> F11/F12 的 FrozenLegacy 单测注入 `_StubRiskModel`（固定低风险），不依赖真实模型
- demeter numpy 类型兼容沿用

## 8. Reproducibility Notes

- 冻结参数 + 冻结模型 + 固定 OOS 窗口 + pandas_ta 生产同口径指标
- 1m open_time -> close availability time（+1min）语义，bar 时间戳=完全可用时刻
- 完整命令见 §4；equity CSV 含 7 条每日净值曲线可复现

## 9. Known Limitations

- 本地无 Binance 原生 15m/4h 文件，Layer 1 parity 无法与 API 原生 bar 逐值核对；聚合公式（open=first/high=max/low=min/close=last）有精确单测覆盖，但与 API 原生聚合口径理论上同源（data.binance.vision）
- Legacy-Cost 的 latency_bias/exit deduction 是旧代码假设，非实测滑点
- XGBoost 风险概率经 deap-skipping unpickler 加载（booster 完整，ga 对象跳过）
- fee token 数量基于 demeter uncollected 序列正增量（同一次 remove 中本金与手续费在 demeter 内部均经 pending 通道，但 uncollected 列口径仅含 fee，已由 action 时间线核对）

## 10. Architect Iteration 3 的 12 个强制答案

1. **Q1 输入是原生还是聚合？** -> 聚合：本地无原生 15m/4h，按 F7 优先级 2 用 1m 完整 OHLC（open/high/low/close）聚合，公式固定。
2. **Q2 parity 结果？** -> Layer 1：无原生文件可对，聚合公式有精确单测（high=3000 进入 15m high）；Layer 2：全部特征来自单一 `compute_signals_from_ohlc` pandas_ta 路径，列清单见 JSON `parity`。
3. **Q3 available-time 定义？** -> 每个 1m kline 的 close available time = open_time+1min；15m/4h bar 时间戳=该 bar 完全可用时刻；00:15 决策看不到 00:15 这一分钟。
4. **Q4 残留 look-ahead？** -> 无：`aggregate_ohlc` 在 available-time 索引上 `label='right', closed='right'` 聚合，bar 边界严格前向；精确边界单测通过。
5. **Q5 正确的 idle_ratio？** -> 建仓时点快照（`deploy_snapshots`）：Always LP max 0.73%、Frozen max 0.80%，均 <1%；定义 position_value=base_in_pos*price+quote_in_pos、idle_wallet=wallet_ETH*price+wallet_USDC、total_nav_components=position+idle+uncollected_fee。
6. **Q6 正确的 fee token 数量？** -> Frozen gross：ETH 0.0276 + USDC 50.80；Always LP：ETH 0.7039 + USDC 1,387.00（token 数量，非价格重估）。
7. **Q7 2976.94 是否成立？** -> 不直接比较：2976.94 是 iteration 2 的价格重估口径；iteration 3 改为 token 数量累计（Always LP 3,157.22 USDC 按最终价折算），口径差异见 §6。
8. **Q8 Always LP -5.83% 是否保持？** -> **保持**（-5.83%，与 iter2 完全一致；持仓逻辑未变）。
9. **Q9 Frozen +23.91% 是否保持？** -> **不保持：+16.34%**。F7/F8 OHLC 修复显著改变结果——iter2 的 close-only 聚合高估了 Frozen Legacy 收益。
10. **Q10 vs ETH +3.01% 是否保持？** -> **不保持：-3.29%**（反转）。OHLC 修复后 Frozen Legacy 不再跑赢 ETH（Frozen 16.34% vs ETH 20.30%）。
11. **Q11 fee-disabled 差异？** -> Always LP fee 贡献 2,468.70 USDC（fee-on 的 26.21%）；Frozen fee 贡献 130.21 USDC（1.12%）。
12. **Q12 哪项修复影响最大？** -> **F7+F8（OHLC 聚合与可用时点）**：Frozen 从 +23.91% 变 +16.34%、vs ETH 从 +3.01% 变 -3.29%。F10 修正 fee 口径（2,976.94 -> 3,157.22 token 口径）但策略收益数字不受影响（fee 只改变报告口径，NAV 由 demeter 直接给出）。

## 11. 交接（CURRENT_TASK.md §19）

- `WORKFLOW_STATE.yaml` 已更新：handoff_seq=12、新 handoff_id、state=REVIEW_READY、owner=architect、authorized_next=[]
- 普通 commit + 普通 push 到 main，完成后停止等待 Architect Review。
