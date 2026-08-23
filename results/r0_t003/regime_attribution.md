# R0-T003 — ETH Regime Attribution

- OOS 窗口：2026-03-14T00:00:00+00:00 -> 2026-08-21T23:59:59+00:00
- Regime 判定：固定 14 日窗口 + 段收益方向 (>= +5% bull / <= -5% bear / 其余 range)，相邻同 regime 合并
- 方法：ex-post attribution（事后归因，不用于策略决策）

## 上升 (bull) regime（35 天，ETH 阶段收益 64.13%）

| 阶段 | 起 | 止 | 天数 | ETH 收益 |
|---|---|---:|---:|---:|
| bull | 2026-03-28 | 2026-04-10 | 14 | 12.63% |
| bull | 2026-06-06 | 2026-06-19 | 14 | 9.00% |
| bull | 2026-08-15 | 2026-08-21 | 7 | 33.69% |

| 策略 | 阶段收益 | 最大回撤 | vs ETH | vs USDC |
|---|---:|---:|---:|---:|
| Frozen Legacy (Gross, Binance) | 33.01% | -7.07% | -31.12% | 33.01% |
| Frozen Legacy (Legacy-Cost, Binance) | 33.01% | -7.09% | -31.12% | 33.01% |
| Always LP (Gross, Binance) | 20.74% | -1.25% | -43.39% | 20.74% |
| Frozen Legacy (Gross, Pool 对照) | 33.22% | -7.17% | -30.91% | 33.22% |
| Always ETH | 64.13% | -4.75% | 0.00% | 64.13% |
| Always USDC | 0.00% | 0.00% | -64.13% | 0.00% |
| 50/50 Buy-and-Hold | 27.86% | -2.20% | -36.27% | 27.86% |

## 下降 (bear) regime（42 天，ETH 阶段收益 -35.37%）

| 阶段 | 起 | 止 | 天数 | ETH 收益 |
|---|---|---:|---:|---:|
| bear | 2026-03-14 | 2026-03-27 | 14 | -5.02% |
| bear | 2026-05-09 | 2026-06-05 | 28 | -31.95% |

| 策略 | 阶段收益 | 最大回撤 | vs ETH | vs USDC |
|---|---:|---:|---:|---:|
| Frozen Legacy (Gross, Binance) | -6.68% | -11.73% | 28.69% | -6.68% |
| Frozen Legacy (Legacy-Cost, Binance) | -6.88% | -11.71% | 28.49% | -6.88% |
| Always LP (Gross, Binance) | -29.66% | -23.46% | 5.71% | -29.66% |
| Frozen Legacy (Gross, Pool 对照) | -6.72% | -11.73% | 28.65% | -6.72% |
| Always ETH | -35.37% | -33.20% | 0.00% | -35.37% |
| Always USDC | 0.00% | 0.00% | 35.37% | 0.00% |
| 50/50 Buy-and-Hold | -18.92% | -17.64% | 16.45% | -18.92% |

## 震荡 (range) regime（84 天，ETH 阶段收益 9.05%）

| 阶段 | 起 | 止 | 天数 | ETH 收益 |
|---|---|---:|---:|---:|
| range | 2026-04-11 | 2026-05-08 | 28 | 0.89% |
| range | 2026-06-20 | 2026-08-14 | 56 | 8.08% |

| 策略 | 阶段收益 | 最大回撤 | vs ETH | vs USDC |
|---|---:|---:|---:|---:|
| Frozen Legacy (Gross, Binance) | -9.07% | -9.03% | -18.11% | -9.07% |
| Frozen Legacy (Legacy-Cost, Binance) | -9.26% | -9.10% | -18.31% | -9.26% |
| Always LP (Gross, Binance) | 8.09% | -7.03% | -0.96% | 8.09% |
| Frozen Legacy (Gross, Pool 对照) | -9.53% | -8.98% | -18.58% | -9.53% |
| Always ETH | 9.05% | -10.00% | 0.00% | 9.05% |
| Always USDC | 0.00% | 0.00% | -9.05% | 0.00% |
| 50/50 Buy-and-Hold | 4.15% | -4.54% | -4.89% | 4.15% |

## 结论

- **上升 (bull)**（35 天）：最优策略 **Always ETH** (64.13%)；Frozen 33.01% vs Always ETH 64.13% vs Always LP 20.74%
- **下降 (bear)**（42 天）：最优策略 **Always USDC** (0.00%)；Frozen -6.68% vs Always ETH -35.37% vs Always LP -29.66%
- **震荡 (range)**（84 天）：最优策略 **Always ETH** (9.05%)；Frozen -9.07% vs Always ETH 9.05% vs Always LP 8.09%
