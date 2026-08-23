# R0-T002 Post-Freeze Strict OOS Validation - Iteration 2

- OOS 窗口：2026-03-14T00:00:00+00:00 -> 2026-08-21T23:59:59+00:00
- 初始资本：10000.0 USDC（全部策略一致）
- 主信号源：Binance spot ETHUSDT 1m (production-like, pandas_ta)
- 对照信号源：Pool-derived minute price (control only)

## 1. 策略指标对比（主结果：Binance 信号）

| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| A. Frozen Legacy (Gross, Binance) | 12391.39 | 23.91% | 62.64% | -25.21% | 1.3136 | 2.4299 |
| A. Frozen Legacy (Legacy-Cost, Binance) | 12350.77 | 23.51% | 61.43% | -25.39% | 1.2951 | 2.4026 |
| B. Always LP (Gross, Binance) | 9417.41 | -5.83% | -12.73% | -26.08% | -0.3489 | -0.3395 |
| A. Frozen Legacy (Gross, Pool 对照) | 12263.48 | 22.63% | 58.85% | -25.22% | 1.2609 | 2.3459 |
| C. Always ETH | 12029.67 | 20.30% | 52.07% | -38.67% | 1.0025 | 1.755 |
| D. Always USDC | 10000.0 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 |
| E. 50/50 Buy-and-Hold | 11014.83 | 10.15% | 24.51% | -20.92% | 0.9178 | 1.6091 |

## 2. Frozen Legacy 超额收益（Binance 主信号）

- frozen_vs_always_lp_binance: 31.58%
- frozen_vs_always_eth_binance: 3.01%
- frozen_vs_always_usdc_binance: 23.91%
- frozen_vs_buy_hold_5050_binance: 12.50%

## 3. 事件统计（Frozen Legacy, Binance 信号）

| 事件 | 次数 |
|---|---:|
| ACTIVE_TO_SAFE | 39 |
| SAFE_TO_ACTIVE | 39 |
| SAFE_ETH | 18 |
| SAFE_USDC | 17 |
| SAFE_KEEP | 4 |
| COOLDOWN_SKIP | 9102 |
| PERIODIC_REBALANCE | 0 |

- 总决策次数：15448
- LP 在池时间占比（相对总决策）：2.4%
- LP 期间在区间内占比：100.0% （活跃 373/373，出区间 0）

## 4. 累计 LP 手续费与资本部署

- Frozen Legacy (Gross, Binance)：88.28 USDC
- Always LP (Binance)：2976.94 USDC
- Always LP 最终部署价值：9353.14 USDC，钱包闲置：13.76 USDC

## 5. Iteration 2 修复（F1-F6）

- F1_binance_primary: Binance 1m -> 15m/4h signals (primary); pool-derived as control; separate columns
- F2_causality: resample label='right' closed='right'; bar timestamp = close time
- F3_pandas_ta: pandas_ta indicators (RSI/ADX+ADXR_14_2/NATR% /bbands); NATR in percent scale
- F4_capital_deploy: V3 range value-ratio capital deployment; idle tracked
- F5_periodic_rebalance: 4-day periodic rebalance in ACTIVE state (production semantics)
- F6_cumulative_fee: cumulative fee = positive diffs of uncollected fee series

## 6. 数据证据

- binance_files: ETHUSDT-1m zips 2026-01-28..2026-08-21
- binance_rows: 296640
- binance_gaps: checked at load; all daily files present
- pool_rows_oos: 231725
- pool_warmup_rows: 64490
- binance_signal_rows_oos: 15457
- pool_signal_rows_oos: 15457
