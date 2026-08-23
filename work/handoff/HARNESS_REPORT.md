# Harness Report — R0-T003 ETH Regime Attribution

> 协议：UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md
> 报告人：DeepSeek Harness（本地执行 Agent）
> 说明：Architect 掉线未能补发任务书，按用户直接指示由 Harness 自拟任务书并执行。

## 1. Task Identity（任务标识）

- task_id: **R0-T003**
- iteration: 1
- 任务来源：用户直接指示（Architect 掉线）——"看看 ETH 在下降和震荡阶段不同策略的效果"
- 任务书：`work/control/CURRENT_TASK.md`（Harness 自拟，已推）
- 状态：HARNESS_READY ->（本报告）

## 2. 研究问题与方法

用户问题：**ETH 在下降和震荡阶段，不同策略的效果（收益/回撤/超额）分别如何？**

方法（ex-post attribution，透明可复现）：
1. 复用 R0-T002 iteration 4 已产出的 7 条逐日净值曲线（策略逻辑未改，P0-1 只加校验不改变净值）；
2. 用 ETH 池价日线，按**固定 14 日窗口 + 段收益方向**切分 regime（>=+5% bull / <=-5% bear / 其余 range），相邻同 regime 合并；
3. 对每个 regime，统计各策略收益（多段复利合成）、最大回撤、相对 Always ETH / USDC 超额。

Regime 划分结果（161 天 OOS 全覆盖，无 gap）：
| Regime | 覆盖 | ETH 阶段收益 |
| :-- | :-- | --: |
| 上升 bull | 03-28..04-10, 06-06..06-19, 08-15..08-21（35 天）| +64.13% |
| 下降 bear | 03-14..03-27, 05-09..06-05（42 天）| -35.37% |
| 震荡 range | 04-11..05-08, 06-20..08-14（84 天）| +9.05% |

## 3. Changed Files（变更文件）

| 文件 | 说明 |
| :-- | :-- |
| `research/r0_t003_regime_attribution.py` | 新增：regime 划分 + 策略表现归因 |
| `tests/test_r0_t003_regime_attribution.py` | 新增：10 个测试 |
| `results/r0_t003/regime_attribution.json` | 新增 |
| `results/r0_t003/regime_attribution.md` | 新增 |
| `work/control/CURRENT_TASK.md` | 重写为 R0-T003 任务书 |
| `work/control/WORKFLOW_STATE.yaml` | 更新 |

未修改：R0-T002 相关文件、legacy 策略、模型、README、协议、原始数据。

## 4. Commands Executed

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\venv\Scripts\python.exe" research/r0_t003_regime_attribution.py
# -> exit 0，输出 results/r0_t003/*

PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\v3.12\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_r0_t003_regime_attribution.py
# -> 10 passed
```

## 5. Test Results

`10 passed in 1.36s`（v3.12, pytest 9.0.2）
覆盖：regime 划分 sanity（三态均现/无 gap/天数=窗口/eth_ret 正确）、策略收益计算（单段/多段复利/超额）、schema、已生成结果 sanity（bear 阶段 Frozen > ETH）。

## 6. Backtest / Validation Results（核心）

### 下降 regime（42 天，ETH -35.37%）
| 策略 | 阶段收益 | 最大回撤 | vs ETH |
| :-- | --: | --: | --: |
| **Frozen Legacy (Gross)** | **-6.68%** | -11.73% | **+28.69%** |
| Frozen Legacy (Legacy-Cost) | -6.88% | -11.71% | +28.49% |
| Always LP | -29.66% | -23.46% | +5.71% |
| Always ETH | -35.37% | -33.20% | 0 |
| Always USDC | 0.00% | 0.00% | +35.37% |
| 50/50 | -18.92% | -17.64% | +16.45% |

### 震荡 regime（84 天，ETH +9.05%）
| 策略 | 阶段收益 | 最大回撤 | vs ETH |
| :-- | --: | --: | --: |
| Frozen Legacy (Gross) | **-9.07%** | -9.03% | -18.11% |
| Always LP | +8.09% | -7.03% | -0.96% |
| Always ETH | +9.05% | -10.00% | 0 |
| Always USDC | 0.00% | 0.00% | -9.05% |
| 50/50 | +4.15% | -4.54% | -4.89% |

### 上升 regime（35 天，ETH +64.13%）
| 策略 | 阶段收益 | 最大回撤 | vs ETH |
| :-- | --: | --: | --: |
| Frozen Legacy (Gross) | +33.01% | -7.07% | -31.12% |
| Always LP | +20.74% | -1.25% | -43.39% |
| Always ETH | +64.13% | -4.75% | 0 |
| Always USDC | 0.00% | 0.00% | -64.13% |
| 50/50 | +27.86% | -2.20% | -36.27% |

## 7. 结论（回答用户问题）

1. **下降阶段**：Frozen Legacy 完胜——只跌 6.68%（vs Always ETH -35.4% / Always LP -29.7%），超额 +28.7%。这是 R0-T002"超额主要来自 SAFE 避险择时"的直接体现：下跌期成功切换到 USDC/ETH 避险。
2. **震荡阶段**：Frozen Legacy 反而最差（-9.07%）——4 天防震荡冷却 + 频繁进出在震荡市来回止损；Always LP（+8.09%）和 Always ETH（+9.05%）均优于它。这暴露 Frozen 的**明确短板**：震荡市不赚钱、甚至亏钱。
3. **上升阶段**：Frozen 落后 ETH（+33% vs +64%），符合其避险/半仓特性，不追涨。

**整体画像**：Frozen Legacy 是"下降市护盾、上升市落后、震荡市亏损"的策略。它在 OOS 全窗口相对 Always LP +23.54%，主要靠下降段的巨大抗跌贡献；但相对 ETH -3.29%，因为上升段跟不上 + 震荡段亏损。

## 8. Known Limitations

- regime 划分为 ex-post attribution（事后归因），不用于策略决策；固定 14 日窗口 + 5% 阈值是透明固定规则，非优化；
- 震荡段 06-20..08-14（56 天 +8.1%）内部实际先跌后涨（07 月低点约 1700 再回升），14 日窗口净收益标 range 可接受；
- 复用 R0-T002 净值，未重跑策略（P0-1 只加校验不改净值，已核对 equity 曲线逐日一致）。

## 9. 交接

- `WORKFLOW_STATE.yaml`：见 commit（待定交接方式）。
- 推送后等待下一步指示（Architect 掉线，可能由用户直接评审）。
