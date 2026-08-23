# R0-T002 Post-Freeze Strict OOS Validation - Iteration 3

- OOS 窗口：2026-03-14T00:00:00+00:00 -> 2026-08-21T23:59:59+00:00
- 初始资本：10000.0 USDC（全部策略一致）
- 主信号源：Binance spot ETHUSDT 1m OHLC (production-like, pandas_ta)
- 对照信号源：Pool-derived minute OHLC (control only)

## 1. 策略指标对比（主结果：Binance 信号）

| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| A. Frozen Legacy (Gross, Binance) | 11634.1 | 16.34% | 40.96% | -29.68% | 1.0142 | 1.6736 |
| A. Frozen Legacy (Legacy-Cost, Binance) | 11589.01 | 15.89% | 39.72% | -29.89% | 0.9918 | 1.6436 |
| B. Always LP (Gross, Binance) | 9417.41 | -5.83% | -12.73% | -26.08% | -0.3489 | -0.3395 |
| A. Frozen Legacy (Gross, Pool 对照) | 11616.74 | 16.17% | 40.48% | -29.72% | 1.0083 | 1.6599 |
| C. Always ETH | 12029.67 | 20.30% | 52.07% | -38.67% | 1.0025 | 1.755 |
| D. Always USDC | 10000.0 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 |
| E. 50/50 Buy-and-Hold | 11014.83 | 10.15% | 24.51% | -20.92% | 0.9178 | 1.6091 |

## 2. Frozen Legacy 超额收益（Binance 主信号）

- frozen_vs_always_lp_binance: 23.54%
- frozen_vs_always_eth_binance: -3.29%
- frozen_vs_always_usdc_binance: 16.34%
- frozen_vs_buy_hold_5050_binance: 5.62%

## 3. 事件统计（Frozen Legacy, Binance 信号）

| 事件 | 次数 |
|---|---:|
| ACTIVE_TO_SAFE | 38 |
| SAFE_TO_ACTIVE | 38 |
| SAFE_ETH | 18 |
| SAFE_USDC | 19 |
| SAFE_KEEP | 1 |
| COOLDOWN_SKIP | 8842 |
| PERIODIC_REBALANCE | 0 |

- 总决策次数：15448
- LP 在池时间占比（相对总决策）：3.9%
- LP 期间在区间内占比：100.0% （活跃 602/602，出区间 0）

## 4. 累计 LP 手续费（token 级，F10）与资本部署（F9）

- Frozen Legacy (Gross, Binance)：ETH 0.027588 + USDC 50.796542 （按最终价折 120.18 USDC）
- Always LP (Binance)：ETH 0.703875 + USDC 1387.003899 （按最终价折 3157.22 USDC）
- Always LP 最终仓位价值：9339.3793 USDC，钱包闲置价值：13.7582 USDC （idle_ratio_final=0.15%）
- 建仓时点快照数（Always LP）：41，最大 idle_ratio=0.7255%（invariant: <1%）

## 5. LP PnL Reconciliation 与 fee-disabled 反事实（F12）

### Always LP fee on/off 对比

- fee_on final_nav: 9417.407480111131 USDC
- fee_off final_nav: 6948.7040054899935 USDC
- fee 贡献（差值）: 2468.7 USDC （占 fee_on 26.21%）
- on 版对账：position=9339.3793 + idle=13.7582 + uncollected_fee=64.27 = 9417.4075（vs final_nav 9417.407480111131）
- on 版动作统计：add=41 remove=40 collect=40 swap=41
### Frozen Legacy fee on/off 对比

- fee_on final_nav: 11634.104670114772 USDC
- fee_off final_nav: 11503.894436917371 USDC
- fee 贡献（差值）: 130.21 USDC （占 fee_on 1.12%）
- on 版对账：position=11625.51 + idle=3.7932 + uncollected_fee=4.8015 = 11634.1047（vs final_nav 11634.104670114772）
- on 版动作统计：add=38 remove=37 collect=37 swap=75

## 6. Iteration 3 修复（F7-F13）

- F7_ohlc_aggregation: 1m 完整 OHLC -> 15m/4h：open=first, high=max, low=min, close=last；load_binance_ethusdt_1m 读 [open,high,low,close]，pool 用 tick 派生 OHLC
- F8_bar_available_time: 1m kline ts=open_time；方案 A：映射为 close availability time(+1min) 再聚合，bar 时间戳=完全可用时刻，无 1 分钟未来泄漏
- F9_deploy_invariant: position_value=base_in_pos*price+quote_in_pos；idle_wallet=wallet_ETH*price+wallet_USDC；total_nav_components=position+idle+uncollected_fee；idle_ratio 在建仓时点记录并断言 <1%
- F10_token_fee: 累计 fee 按 token 数量：action log collect 的 base/quote + 最终 uncollected；不再用价格重估 diff() 口径
- F11_periodic_rebalance_test: deterministic 单测：ACTIVE+持仓 t0..t0+3d 不重建，t0+4d 恰好一次重建，last_rebalance 更新（Frozen 与 Always LP 均覆盖）
- F12_lp_reconciliation: NAV=position+idle+uncollected_fee 幂等对账表 + fee-disabled counterfactual (fee_rate=0) 隔离手续费贡献
- F13_parity_two_layers: Layer1 OHLC parity（无原生文件→报告聚合公式）；Layer2 feature parity （单一 pandas_ta 聚合路径，列清单见 parity）

## 7. Architect 12 个强制答案

- **Q1_input_native_or_aggregated**: aggregated-from-1m-OHLC（本地无原生 15m/4h；F7 优先级 2）
- **Q2_parity_results**: Layer1: 无原生文件可对，仅聚合公式单测；Layer2: 特征列全部来自 compute_signals_from_ohlc 单一 pandas_ta 路径（见 tests）
- **Q3_available_time_definition**: 每个 1m kline 的 close available time = open_time + 1min；15m/4h bar 时间戳 = close available time（聚合完成即完全可用），00:15 决策看不到 00:15 这一分钟
- **Q4_residual_lookahead**: 无：aggregate_ohlc 用 label='right', closed='right' 在 available-time 索引上聚合，bar 边界严格前向
- **Q5_correct_idle_ratio**: 建仓时点快照验证 idle_ratio<1%（见 deploy_snapshots 与 F9 单测）；定义 position_value=base_in_pos*price+quote_in_pos, idle_wallet=wallet_ETH*price+wallet_USDC, total_nav_components=position+idle+uncollected_fee
- **Q6_fee_token_counts**: Frozen gross: ETH=0.027588, USDC=50.796542; Always LP: ETH=0.703875, USDC=1387.003899（token 数量，非价格重估）
- **Q7_does_2976_hold**: 2976.94 是 iteration 2 的价格重估口径累计 fee；iteration 3 改为 token 数量累计，按最终价格折算 value 见 event_stats.cum_fee_*_value（不再与旧口径直接比较）
- **Q8_always_lp_minus_583_holds**: Always LP total_return=-5.83% (iter2 -5.83%) -> 保持
- **Q9_frozen_plus_2391_holds**: Frozen gross total_return=16.34% (iter2 +23.91%) -> 有变化
- **Q10_vs_eth_plus_301_holds**: excess vs always_eth=-3.29% (iter2 +3.01%) -> 有变化
- **Q11_fee_disabled_difference**: Always LP fee 贡献=2468.7 USDC; Frozen fee 贡献=130.21 USDC
- **Q12_which_fix_matters_most**: F7+F8（OHLC 聚合与可用时点）改变信号 high/low 与边界，F10（token 级 fee）改变 fee 口径；最终以回测数字对比为准

## 8. 数据证据与 parity（F13）

- binance_files: ETHUSDT-1m zips 2026-01-28..2026-08-21
- binance_rows: 296640
- binance_ohlc_columns: ['open', 'high', 'low', 'close']
- binance_gaps: checked at load; all daily files present
- pool_rows_oos: 231725
- pool_ohlc_rows: 296215
- binance_signal_rows_oos: 15457
- pool_signal_rows_oos: 15457
- aggregation_rule: 1m OHLC -> 15m/4h: open=first, high=max, low=min, close=last; 1m open_time -> close availability time (+1min)
- native_15m_4h_available: False
- native_15m_4h_note: 本地 BINANCE_KDATA 无原生 15m/4h 文件（仅 1m/1s），按 F7 优先级 2 用 1m OHLC 聚合

### Layer 1 OHLC parity

- native_15m_4h_available: False
- method: 1m OHLC aggregation (open=first, high=max, low=min, close=last)
- note: 本地无原生 15m/4h 文件；聚合公式按 Architect Review F7 固定实现，并有单日 1m OHLC 聚合测试覆盖（见 tests）
- binance_rows_1m: 296640
- binance_signal_rows_15m_oos: 15457
- pool_signal_rows_15m_oos: 15457

### Layer 2 Feature parity

- feature_source: compute_signals_from_ohlc -> pandas_ta 同口径
- columns: ['RSI_14', 'ADX_14', 'ADXR_14_2', 'DMP_14', 'DMN_14', 'NATR_14', 'bb_width', 'macro_rsi', 'macro_ema', 'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag4', 'NATR_14_lag1', 'NATR_14_lag2', 'NATR_14_lag4', 'ADX_14_lag1', 'ADX_14_lag2', 'ADX_14_lag4', 'bb_width_lag1', 'bb_width_lag2', 'bb_width_lag4']
