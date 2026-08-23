# R0-T002 Post-Freeze Strict OOS Validation（冻结后严格样本外验证）

- OOS 窗口：2026-03-14T00:00:00+00:00 → 2026-08-21T23:59:59+00:00
- 初始资本：10000.0 USDC（全部策略一致）

## 1. 策略指标对比

| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |
|---|---:|---:|---:|---:|---:|---:|
| A. Frozen Legacy (Gross) | 12137.72 | 21.38% | 55.18% | -24.09% | 1.2561 | 2.0583 |
| A. Frozen Legacy (Legacy-Cost) | 12125.68 | 21.26% | 54.83% | -24.22% | 1.2489 | 2.0574 |
| B. Always LP (Gross) | 9995.75 | -0.04% | -0.10% | -0.13% | -0.8297 | -0.592 |
| C. Always ETH | 12029.67 | 20.30% | 52.07% | -38.67% | 1.0025 | 1.755 |
| D. Always USDC | 10000.0 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 |
| E. 50/50 Buy-and-Hold | 11014.83 | 10.15% | 24.51% | -20.92% | 0.9178 | 1.6091 |

## 2. Frozen Legacy 超额收益

- vs_always_lp: 21.43%
- vs_always_eth: 0.90%
- vs_always_usdc: 21.38%
- vs_buy_hold_5050: 10.19%

## 3. 事件统计（Frozen Legacy）

| 事件 | 次数 |
|---|---:|
| ACTIVE_TO_SAFE | 38 |
| SAFE_TO_ACTIVE | 37 |
| SAFE_ETH | 18 |
| SAFE_USDC | 19 |
| SAFE_KEEP | 1 |
| COOLDOWN_SKIP | 8795 |

- 总决策次数：15448
- LP 在池时间占比（相对总决策）：3.9%
- LP 期间在区间内占比：100.0% （活跃 607/607，出区间 0）

## 4. 累计 LP 手续费

- Frozen Legacy (Gross)：0.0 USDC
- Frozen Legacy (Legacy-Cost)：0.0 USDC
- Always LP：0.03 USDC

## 5. 数据与方法说明

- 引擎：demeter（Uniswap V3 回测框架）
- 池：Arbitrum WETH/USDC 0.05%（231725 分钟数据）
- 信号：15m/4h 技术指标（已收盘 bar，merge_asof backward），15456 个决策点
- 成本：Gross（无成本）/ Legacy-Cost（latency_bias=5bps + exit 扣 0.0002）
- 详细净值曲线见 post_freeze_oos_equity.csv
