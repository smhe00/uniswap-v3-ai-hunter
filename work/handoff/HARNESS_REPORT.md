# Harness Report

> 协议：UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md（第 10 节标准结构）
> 报告人：DeepSeek Harness（本地执行 Agent）

## 1. Task Identity（任务标识）

- task_id: **R0-T002**
- iteration: 2（Architect CHANGES_REQUIRED -> 修复 F1-F6）
- consumed_handoff_id: **R0-T002-ARCH-20260823-002**
- base_remote_head: `a72b3f6fe1846c485cfb6c31d712e772607c19df`（本轮任务基线）
- result_commit: `PENDING_SELF`（随 commit 推送，最终 SHA 以远端 remote_head 为准）

## 1.1 Architect Review 处置摘要（Iteration 1 -> 2）

| Finding | 处置 |
| :-- | :-- |
| F1 未用 Binance 生产信号 | 主结果改用 Binance spot ETHUSDT 1m（296,640 行，2026-01-28..08-21）构造 15m/4h 信号；Pool-derived 仅作对照；JSON/MD 分栏（`*_binance` 主栏 + `*_pool_control` 对照栏） |
| F2 4h bar 未来泄漏 | `resample("4h", label="right", closed="right")`，bar 时间戳=收盘时刻；新增 4h causality test（尖峰合成数据验证收盘前不可见） |
| F3 NATR 尺度错误（差 100 倍） | 指标全部改用生产同口径 pandas_ta（`RSI_14/ADX_14/ADXR_14_2/DMP_14/DMN_14/NATR_14/bb_width`）；NATR 百分比尺度；新增 parity test（与 pandas_ta 原生输出 diff < 1e-6） |
| F4 LP 资本部署单位 bug | 重写为 V3 区间价值配比部署（`_v3_range_value_ratio`）；记录 deployed/idle；新增 invariant test（idle < 1%） |
| F5 未实现 4 天周期再平衡 | Frozen Legacy 与 Always LP 均实现 `PERIODIC_REBALANCE`（ACTIVE 超 4 天 remove+recenter）；事件计数输出 |
| F6 累计 fee 不完整 | 改为 uncollected fee 序列正增量累加（已实现+未领取）；新增 cumulative fee test（>0） |

## 2. Changed Files（变更文件）

| 文件 | 说明 |
| :-- | :-- |
| `research/r0_t002_post_freeze_oos.py` | Iteration 2 重写：Binance 主信号 + pandas_ta 同口径指标 + label-right 因果 + V3 配比部署 + 周期再平衡 + 累计 fee |
| `tests/test_r0_t002_post_freeze_oos.py` | 19 个测试（原 14 + 新增 causality/parity/deploy invariant/periodic rebalance/cumulative fee） |
| `results/r0_t002/post_freeze_oos.json` | 重新生成（分栏结果） |
| `results/r0_t002/post_freeze_oos.md` | 重新生成 |
| `results/r0_t002/post_freeze_oos_equity.csv` | 重新生成（7 列：Binance 主×3 + Pool 对照 + 3 基准） |
| `work/handoff/HARNESS_REPORT.md` | 本报告 |
| `work/control/WORKFLOW_STATE.yaml` | state -> REVIEW_READY |

未修改：legacy 策略文件、README、模型文件、协议文件、本地原始数据。
环境说明：`.local/pandas_ta_pkg/`（生产同版本 pandas_ta 纯 Python 包，来自本机旧项目 Linux venv）与 `.local/demeter_pkg/` 均为本地私有（git-ignore，未提交），属任务 §15 允许的本地依赖准备。

## 3. Local Data Used（本地数据证据表，协议 §9.2）

| 项目 | 说明 |
| :-- | :-- |
| **Dataset Name** | Binance spot ETHUSDT 1m klines（主信号源）+ UNIV3_DATA 池 minute.csv（回测价格/流动性） |
| **Coverage Start / End** | Binance：2026-01-28..2026-08-21（45 天 warmup + OOS）；池：OOS 231,725 分钟行 + warmup 64,490 行 |
| **File Count** | Binance 1m zip 206 个（每日齐全，无缺口）；池 OOS 161 天每日齐全 |
| **Row / Swap Count** | Binance 296,640 行；池 231,725 分钟行 |
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
# -> exit 0，输出 results/r0_t002/*（4 次回测：Binance Gross/Cost + Always LP + Pool 对照）

# 测试（v3.12 环境，pytest 9.0.2）
PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\v3.12\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_r0_t002_post_freeze_oos.py
# -> 19 passed
```

## 5. Test Results（测试结果）

`19 passed in 18.26s`（pytest 9.0.2）

覆盖任务 §14 全部 10 项 + Architect Required Validation：
1. ✅ OOS 起始固定（TestOOSWindow）
2. ✅ 冻结参数一致（TestFrozenParams）
3. ✅ 无优化器（TestNoOptimizer）
4. ✅ **F2 causality**（TestCausality：4h 尖峰合成数据收盘前不可见 + 15m 时间戳=收盘时刻）
5. ✅ 初始净值一致（TestInitialCapital）
6. ✅ **F3 parity**（TestPandasTaParity：NATR 百分比尺度 + 与 pandas_ta 原生 diff<1e-6 + ADXR_14_2 存在）
7. ✅ Gross/Legacy-Cost 分离（双实例）
8. ✅ 缺数据明确失败（RuntimeError，不伪造）
9. ✅ schema 稳定（TestSchema 含分栏字段）
10. ✅ 无链上写（TestNoOnchainWrite）
11. ✅ **F4 deploy invariant**（TestLPCapitalDeploy：idle < 1%）
12. ✅ **F5 periodic rebalance**（TestPeriodicRebalance）
13. ✅ **F6 cumulative fee**（TestCumulativeFee：fee > 0）
14. ✅ 基准指标正确（TestMetrics：USDC 恒定 / ETH 跟随价格）

## 6. Backtest / Validation Results

### 策略指标对比（主结果：Binance 生产近似信号）

| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |
| :-- | --: | --: | --: | --: | --: | --: |
| A. Frozen Legacy (Gross, Binance) | **12391.39** | **+23.91%** | 62.64% | -25.21% | 1.3136 | 2.4299 |
| A. Frozen Legacy (Legacy-Cost, Binance) | 12350.77 | +23.51% | 61.43% | -25.39% | 1.2951 | 2.4026 |
| B. Always LP (Gross, Binance) | 9417.41 | **-5.83%** | -12.73% | -26.08% | -0.3489 | -0.3395 |
| A. Frozen Legacy (Gross, Pool 对照) | 12263.48 | +22.63% | 58.85% | -25.22% | 1.2609 | 2.3459 |
| C. Always ETH | 12029.67 | +20.30% | 52.07% | -38.67% | 1.0025 | 1.755 |
| D. Always USDC | 10000.0 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 |
| E. 50/50 Buy-and-Hold | 11014.83 | +10.15% | 24.51% | -20.92% | 0.9178 | 1.6091 |

### Frozen Legacy 超额收益（Binance 主信号）
- vs Always LP: **+31.58%**
- vs Always ETH: **+3.01%**
- vs Always USDC: +23.91%
- vs 50/50 Buy-and-Hold: +12.50%

### 事件统计（Frozen Legacy, Binance，15,448 次决策）
| 事件 | 次数 |
| :-- | --: |
| ACTIVE -> SAFE | 39 |
| SAFE -> ACTIVE | 39 |
| SAFE 进入 ETH | 18 |
| SAFE 进入 USDC | 17 |
| SAFE Keep Ratio | 4 |
| 4 天冷却阻止重建 | 9102 |
| 周期再平衡 | 0（LP 时间短于 4 天，从未触发） |

### 资本部署与累计手续费（F4/F6 修复后）
- Always LP：部署 9,353.14 / 钱包闲置 13.76 USDC（**闲置率 0.15% < 1%**）；累计 fee **2,976.94 USDC**
- Frozen Legacy：部署 12,386.28 / 闲置 4.04 USDC；累计 fee **88.28 USDC**（LP 时间少）
- Iteration 1 的 Always LP ≈ 0% 与 fee=0.03 系 F4 单位 bug 导致资金闲置的假象，已修正

### 信号源对照（F1）
- Binance 主信号：+23.91%；Pool 对照：+22.63%（差异 1.28pp，信号源影响小，结论方向一致）

## 7. Failure / Edge Cases

- Binance 2025+ 新格式 zip 时间戳为**微秒**（旧格式毫秒）-> 加载时自动检测单位
- pandas_ta 短数据返回 None -> `_safe` 兜底 NaN（warmup 不足不崩溃）
- v3.12 测试环境无 pandas-ta 元数据 -> 复制 dist-info 到 .local
- Windows venv 自带 pandas_ta 0.4.71b0 导入超时（numpy 2.x 兼容问题）-> 改用旧项目 Linux venv 同版本包
- demeter numpy 类型兼容（iteration 1 已修）沿用

## 8. Reproducibility Notes

- 冻结参数 + 冻结模型 + 固定 OOS 窗口 + pandas_ta 生产同口径指标（parity test 保证）
- bar 时间戳=收盘时刻（label right），信号 ffill=backward 语义
- 完整命令见 §4；equity CSV 含 7 条每日净值曲线可复现

## 9. Known Limitations

- Binance 1m 聚合的 15m/4h bar 与 Binance API 原生 15m/4h klines 理论上同源（data.binance.vision），但未与 API 原生 bar 逐值核对
- Legacy-Cost 的 latency_bias/exit deduction 是旧代码假设，非实测滑点
- XGBoost 风险概率经 deap-skipping unpickler 加载（booster 完整，ga 对象跳过）

## 10. Requested Architect Decision（任务 §18 八问 + Iteration 2 要求回答）

**Architect 要求回答的四个对比问题：**

1. **修复后 +21.38% 是否仍成立？** -> **成立且更强：+23.91%**（Binance 主信号，Gross）。变化来自：F4 资金正确部署后 LP 行为真实化、F3 NATR 正确尺度使 GA filter 判定变化、F1 Binance 信号。
2. **Always LP 是否仍接近 0%？** -> **否，修正为 -5.83%**。Iteration 1 的 -0.04% 是资金闲置假象（F4 bug）；真实部署后 ±8.13% LP 在 +20% 行情下净亏（无常损失 + 周期重建成本超过 2,976.94 USDC 手续费）。
3. **Frozen Legacy 相对 ETH 的 +0.90% 是否仍存在？** -> **存在且扩大：+3.01%**。修复 NATR 尺度后 GA/避险时机变化，相对 ETH 优势更明显。
4. **结论变化主要来自哪项修正？** -> **F4（资本部署）**对基准影响最大（Always LP 从 -0.04% 变 -5.83%）；**F3（NATR 尺度）**对 Frozen Legacy 影响最大（GA filter 用正确百分比尺度判定，active 占比变化）。

**任务 §18 八问（更新）：**

1. Frozen Legacy OOS 最终收益：**+23.91%**（Gross, Binance）/ +23.51%（Legacy-Cost）
2. 战胜 Always LP：**+31.58%**（Always LP 真实结果 -5.83%）
3. 战胜 ETH：**+3.01%**；战胜 USDC：+23.91%
4. Gross vs Legacy-Cost：差 0.40pp（成本影响仍小）
5. 回撤：Frozen -25.21% vs ETH -38.67%（改善）、vs 50/50 -20.92%（略差）
6. 三态切换真实发生：39 次退出/39 次重建/18 ETH/17 USDC/4 Keep；LP 占比 ~2.4%（GA filter 大部分时间关闭）
7. ±8.13% 在 OOS 无优势：Always LP 用它 -5.83%，出区间重建成本高；但 Frozen 仅短时用它
8. **下一步建议维持结果 B（转向新经济驱动框架）**：Frozen 的超额主要来自 SAFE 避险择时；XGBoost 风险概率在 OOS 上仍普遍 <0.57（模型风险信号失效），LP 本身在上升趋势中为负贡献

## 11. 交接（CURRENT_TASK.md §19）

- `WORKFLOW_STATE.yaml` 已更新：handoff_seq=10、新 handoff_id、state=REVIEW_READY、owner=architect、authorized_next=[]
- 普通 commit + 普通 push 到 main，完成后停止等待 Architect Review。
