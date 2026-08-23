# Harness Report

> 协议：UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md（第 10 节标准结构）
> 报告人：DeepSeek Harness（本地执行 Agent）

## 1. Task Identity（任务标识）

- task_id: **R0-T002**
- iteration: **4**（Architect CHANGES_REQUIRED，修订后范围 -> 修复 P0-1..3 + P1-1..2 + 保留 S1）
- consumed_handoff_id: **R0-T002-ARCH-20260823-005**
- base_remote_head: `3d615db209c9a484a74e2ebfac8ee23d275cedba`（Architect Review 的 git_base_commit）
- result_commit: `PENDING_SELF`（随 commit 推送，最终 SHA 以远端 remote_head 为准）

## 1.1 Architect Review 处置摘要（Iteration 3 -> 4）

| Finding | 处置 |
| :-- | :-- |
| S1 保留 4-day anti-churn cooldown | 明确为**策略意图**而非 bug：`ACTIVE->SAFE` 记录退出时点，`SAFE->ACTIVE` 需 >=4 天否则 `COOLDOWN_SKIP`；未删除/缩短/优化参数。新增行为测试 `TestS1ExitReentryCooldown`（4 天内不重入、4 天后允许） |
| P0-1 模型输入异常不得 fail-open | 新增 `validate_model_inputs`：OOS 前校验 features 存在+顺序+非有限+predict 试跑；策略 `strict_model_input=True` 时缺失/非有限/predict 异常一律 `raise RuntimeError`（不再降级 0.0）。输出 `model_input_audit`：required 19 / missing 0 / non_finite 0 / predict_errors 0 / first_valid 2026-03-14T00:00 / runtime 15448 决策异常 0。新增 fail-fast 测试 6 项 |
| P0-2 时间因果与 OHLC 保持 | Iteration 3 的 F7/F8 修复全保留且回归通过（OHLC 聚合、available-time、00:15/04:00 边界、NATR 百分比尺度）；`native_bar_parity=NOT_AVAILABLE`（无原生文件不阻塞） |
| P0-3 LP 主经济结果可对账 | `final NAV = position + idle + uncollected_fee` reconciliation error < 0.02；deploy idle_ratio<1%；fee-on/fee-off 事件路径一致（新增 `TestP03FeeOnOffActionEquality`） |
| P1-1 fee ledger 命名 | 规范为 `fee_accrued_eth/usdc`（token 数量累计）+ `fee_uncollected_final_eth/usdc`；`fee_collected=fee_accrued-fee_uncollected_final` 低风险拆出（demeter uncollected diff 口径）；旧 `cum_fee` 字段保留但 marked 不再用于结论 |
| P1-2 Legacy-Cost 诚实命名 | `latency_bias=5bps` / `exit_deduction=0.0002` 明确为 `Legacy heuristic cost assumption（旧启发式成本假设）`，非实际 Gas/滑点/历史成本 |

## 2. Changed Files（变更文件）

| 文件 | 说明 |
| :-- | :-- |
| `research/r0_t002_post_freeze_oos.py` | Iteration 4：新增 `validate_model_inputs` + 策略 `strict_model_input` fail-fast + `model_input_audit` 输出 + fee ledger 规范命名 + strategy_answers |
| `tests/test_r0_t002_post_freeze_oos.py` | 47 个测试（新增 P0-1 六项 + S1 cooldown 三项 + P0-3 action-equality 两项 + schema4） |
| `results/r0_t002/post_freeze_oos.json` | 重新生成（iteration 4，含 model_input_audit / strategy_answers / fee ledger / native_bar_parity） |
| `results/r0_t002/post_freeze_oos.md` | 重新生成 |
| `results/r0_t002/post_freeze_oos_equity.csv` | 重新生成（7 列每日净值） |
| `work/handoff/HARNESS_REPORT.md` | 本报告 |
| `work/control/WORKFLOW_STATE.yaml` | state -> REVIEW_READY |

未修改：legacy 策略文件、README、模型文件、协议文件、本地原始数据。未新增其他文件（遵守 Allowed Files）。
环境说明：`.local/pandas_ta_pkg/` 与 `.local/demeter_pkg/` 均为本地私有（git-ignore，未提交），属任务 §15 允许的本地依赖准备。

## 3. Local Data Used（本地数据证据表，协议 §9.2）

| 项目 | 说明 |
| :-- | :-- |
| **Dataset Name** | Binance spot ETHUSDT 1m klines（主信号源）+ UNIV3_DATA 池 minute.csv（回测价格/流动性） |
| **Coverage Start / End** | Binance：2026-01-28..2026-08-21（45 天 warmup + OOS）；池：OOS 231,725 分钟行 |
| **File Count** | Binance 1m zip 206 个（每日齐全）；池 OOS 每日齐全 |
| **Row / Swap Count** | Binance 296,640 行 OHLC；池 231,725 分钟行 |
| **Input Pattern** | `ETHUSDT-1m-*.zip`（微秒时间戳自动检测）+ `arbitrum-0xc696...-*.minute.csv` |
| **Data Gaps** | 无 |
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
# -> exit 0，输出 results/r0_t002/*（8 次回测，含 P0-1 预校验通过）

# 测试（v3.12 环境，pytest 9.0.2）
PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\v3.12\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_r0_t002_post_freeze_oos.py
# -> 47 passed
```

## 5. Test Results（测试结果）

`47 passed in 137.82s`（pytest 9.0.2）

覆盖任务 §14 全部 10 项 + Architect Iteration 4 全部核心测试：
1. ✅ **P0-1 missing feature -> RuntimeError**（`test_missing_model_feature_fails_fast`）
2. ✅ **P0-1 predict_proba raises -> RuntimeError**（`test_predict_exception_fails_fast` + `test_runtime_strict_raises_on_predict_error`）
3. ✅ **P0-1 non-finite -> RuntimeError**（`test_nonfinite_model_input_fails_fast`）
4. ✅ **P0-1 clean OOS -> audit 全 0**（`test_clean_oos_passes_with_audit` + `test_runtime_audit_counts_zero`）
5. ✅ **S1 cooldown 4 天**（`test_exit_reentry_cooldown_is_four_days` 系列：4 天内不重入 + COOLDOWN_SKIP + 4 天后允许）
6. ✅ 原 F7 OHLC aggregation
7. ✅ 原 F8 exact 15m / 4h boundary
8. ✅ 原 LP deploy idle<1% invariant
9. ✅ 原 NAV reconciliation
10. ✅ **fee-on / fee-off action-path equality**（`TestP03FeeOnOffActionEquality`：Always LP 计数一致 + Frozen 事件一致）

## 6. Backtest / Validation Results

### 策略指标对比（主结果：Binance 1m OHLC 生产近似信号，Iteration 4 重跑）

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
- vs Always ETH: **-3.29%**
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
| 4 天冷却阻止重建（COOLDOWN_SKIP） | 8842 |
| 周期再平衡 | 0（LP 时间短于 4 天，从未触发） |

### P0-1 模型输入审计（正式 OOS 无静默 fail-open）
```json
{
  "required_feature_count": 19,
  "missing_feature_count": 0,
  "non_finite_decision_rows": 0,
  "predict_errors": 0,
  "first_valid_oos_decision": "2026-03-14T00:00:00+00:00",
  "runtime_decision_rows": 15448,
  "runtime_non_finite_decision_rows": 0,
  "runtime_predict_errors": 0
}
```
首个有效 OOS 决策 = OOS 起点 00:00（warmup 45 天前置充分，无被静默剔除的 NaN 决策）。

### P1-1 fee ledger（token 数量，非价格重估）
| 指标 | Always LP | Frozen Gross |
| :-- | --: | --: |
| fee_accrued_eth | 0.7039 | 0.0276 |
| fee_accrued_usdc | 1,387.00 | 50.80 |
| fee_uncollected_final_eth | 0.0126 | 0.0010 |
| fee_uncollected_final_usdc | 32.65 | 2.30 |
| fee_collected（=accrued-final） | ETH 0.6913 + USDC 1,354.36 | ETH 0.0266 + USDC 48.50 |

### P0-3 fee-disabled counterfactual + reconciliation
| 策略 | fee_on NAV | fee_off NAV | fee 贡献 | 占 fee_on |
| :-- | --: | --: | --: | --: |
| Always LP | 9,417.41 | 6,948.70 | **2,468.70** | 26.21% |
| Frozen Legacy | 11,634.10 | 11,503.89 | **130.21** | 1.12% |

NAV 对账：`final_nav = position_value + idle_wallet_value + uncollected_fee_value`，error < 0.02 USDC（对账幂等）。

### 信号源对照（F1）
- Binance 主信号：+16.34%；Pool 对照：+16.17%（差异 0.17pp，高度一致）

## 7. Failure / Edge Cases

- Binance 2025+ 新格式 zip 时间戳为微秒（旧格式毫秒）-> 加载时自动检测单位
- demeter `fee_rate=0` 因 `tick_spacing=0` 除零 -> fee-disabled 用 fee=0.05 构造池再仅清零 `pool_info.fee_rate`
- demeter `CollectFeeAction` 含 remove 本金 -> F10/P1-1 用 uncollected 序列 positive diff
- v3.12 测试环境无 xgboost -> P0-1/S1/F11/F12 用 `_StubRiskModel`（不依赖真实模型）
- pandas_ta 短数据返回 None -> `_safe` 兜底 NaN

## 8. Reproducibility Notes

- 冻结参数 + 冻结模型 + 固定 OOS 窗口 + pandas_ta 生产同口径指标
- 1m open_time -> close availability time（+1min）语义，bar 时间戳=完全可用时刻
- P0-1 预校验在正式 OOS 前强制运行（features 存在/顺序/非有限/predict 试跑），失败即中止
- equity CSV 含 7 条每日净值曲线可复现

## 9. Known Limitations（含 Technical Debt）

**P2 / Technical Debt（不阻塞 R0-T002）：**
- 本地无 Binance 原生 15m/4h 文件，`native_bar_parity = NOT_AVAILABLE`（聚合公式有精确单测，但与 API 原生 bar 未逐值核对）
- Legacy-Cost 的 `latency_bias=5bps` / `exit_deduction=0.0002` 是旧启发式成本假设，非实测 Gas/滑点/历史成交成本（已诚实命名）
- `fee_collected = fee_accrued - fee_uncollected_final` 基于 demeter uncollected diff 口径拆分（低风险可靠），但会计级精确拆分未做（Architect P1-1 明确不要求为此重写 fee engine）
- XGBoost 风险概率经 deap-skipping unpickler 加载（booster 完整，ga 对象跳过）
- 旧 `cum_fee_*` 字段保留在 event_stats（兼容），但不再用于核心结论

## 10. Iteration 4 四个策略问题答案

1. **保留 4-day anti-churn cooldown 后，Frozen Legacy 的 OOS 收益是多少？**
   -> **+16.34%**（Gross）/ **+15.89%**（Legacy-Cost）。4 天防震荡冷却作为策略规则保留，收益与 Iteration 3 完全一致（P0-1 只加严格校验不改变策略逻辑）。
2. **是否战胜 Always LP？**
   -> **是：+23.54% 超额**（Frozen +16.34% vs Always LP -5.83%）。
3. **是否战胜 Always ETH？**
   -> **否：-3.29%**（Frozen +16.34% vs Always ETH +20.30%）。OHLC 修复后（Iteration 3 已确认）Frozen 不再跑赢 ETH。
4. **该旧策略是否值得作为后续新 LP/ETH/USDC routing 研究的 benchmark？**
   -> **值得，作为 benchmark 而非继续深挖旧模型**。Frozen 相对 Always LP 显著占优（+23.54%），其超额主要来自 SAFE 避险择时（COOLDOWN_SKIP 8842 次说明大部分时间在防震荡等待，即"少做比常驻 LP 好"）；但相对 ETH 落后，且 XGBoost 风险概率在 OOS 上普遍 <0.57（模型风险信号失效）。建议作为新 routing 研究的对照基准，而非继续调旧模型。

## 11. PASS 标准核对（Iteration 4）

- ✅ 时间因果测试保持通过（F7/F8 精确边界）
- ✅ LP capital/NAV 主对账正确（error<0.02）
- ✅ 模型输入正式 OOS 无静默 fail-open（model_input_audit 全 0 + fail-fast 测试）
- ✅ 4-day anti-churn cooldown 保留且有行为测试
- ✅ 同一冻结参数、同一 OOS 完整重跑（2026-03-14..08-21）
- ✅ 主要结果和限制说明清楚
- ✅ 剩余问题仅 P1/P2，不改变"旧策略 vs LP/ETH"方向性判断

## 12. 交接（CURRENT_TASK.md §19）

- `WORKFLOW_STATE.yaml` 已更新：handoff_seq=15、新 handoff_id、state=REVIEW_READY、owner=architect、authorized_next=[]
- 普通 commit + 普通 push 到 main，完成后停止等待 Architect Review。
