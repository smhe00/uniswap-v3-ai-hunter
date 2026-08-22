# R0-T001 Legacy Claim Audit（旧系统结论证据审计）

- 生成时间：2026-08-23T04:40:28.017483
- 消费 remote_head：04ac458c4154ceb5780980a1b2c2eb45c0f6b54b

## Claim Matrix（结论证据矩阵）

| Claim / 结论 | README 数值 | 代码来源 | 数据来源 | 分类 | OOS 状态 | 当前可信度 | 是否建议重跑 |
|---|---:|---|---|---|---|---|---|
| RANGE_PCT（做市区间）来源 | ±8.13%（约 25 倍资本效率） | wide_golden_params.pkl = {range: 0.081264, risk_thresh: 0.568, m_bull: 52, m_bear: 50}，由 wide_range_study.py 的 Optuna 搜索 | 本地 UNIV3_DATA minute.csv（598 天）；但脚本 DATA_DIR='uniswap_data/U | OPTIMIZER_OUTPUT | IN_SAMPLE | 部分可信（有优化器产物，但搜索=验证同源） | 是 |
| XGB_RISK_THRESHOLD（风险报警阈值）来源 | 0.57（过滤 90% 以上随机噪音） | wide_golden_params.pkl risk_thresh=0.568 来自 wide_range_study.py Optuna 搜索（risk_thresh∈[0.40,0.70]）；lp_smart_agent.py 硬编码 | 同上 | OPTIMIZER_OUTPUT | IN_SAMPLE | 部分可信 | 是 |
| 4 天再平衡冷却期来源 | 4 天（用于锁定手续费复利） | dual_engine_optimizer.py / wide_range_study.py / demeter_asymmetric_backtest.py 中硬编码 timedelta(days=4)；lp_smart_agent.py | 无（代码常量） | HARD_CODED | IN_SAMPLE | 可信（硬编码一致） | 否 |
| 最终净值 $29,270 来源 | $29,270（AI 猎手最终净值） | 仅 README.md 出现；所有旧脚本 / pkl / 结果文件均无 29,270 值。dual_engine_optimizer.py 有 ROI 输出但未保存该净值，且基准 hardcode 20863 | 无法溯源 | MANUAL_SUMMARY | UNKNOWN | 不可信 / 无法复现 | 是 |
| 总 ROI +40.3% 来源 | +40.3%（总 ROI） | 仅 README.md。dual_engine_optimizer.py: roi=(final_nav/20863-1)*100，基准 20863 为硬编码，非代码计算值；无 40.3 常量 | 无法溯源 | MANUAL_SUMMARY | UNKNOWN | 不可信 / 无法复现 | 是 |
| 相对 Alpha +45.3% 来源 | +45.3%（相对 Alpha） | 仅 README.md。代码中无 45.3 常量或输出 | 无法溯源 | MANUAL_SUMMARY | UNKNOWN | 不可信 / 无法复现 | 是 |
| Monte Carlo 胜率 91.7% 来源 | 91.7%（蒙特卡罗胜率） | 仅 README.md。v3_hunter_monte_carlo.py 只跑 10 次随机（range(10)），SUCCESS=(res_df>0).mean()*100；10 次随机不可能得 91.7%（9/10=90%, 10/10 | 本地 raw.csv（部分） | MANUAL_SUMMARY | OVERLAP | 不可信 / 与代码不匹配 | 是 |
| +40.44% 来源 | +40.44%（任务引用数值） | 当前仓库代码 / README / git 历史均无 40.44 | 无 | UNVERIFIED | UNKNOWN | 不可信 | 是 |
| +47.65% 来源 | +47.65%（Alpha，注释声称 bear market 验证） | 仅 lp_smart_agent.py 第 18 行注释 'Alpha +47.65% verified in bear market'，无任何计算代码支撑 | 无 | MANUAL_SUMMARY | UNKNOWN | 不可信 / 无计算支撑 | 是 |
| +32.88% 来源 | +32.88%（Raw Reality Check 输入） | v3_raw_reality_check.py: final_roi_raw = 32.88 * 0.85（硬编码）+ 0.85 'Reality Penalty' 经验系数 | raw.csv（但核心结果硬编码） | HARD_CODED | IN_SAMPLE | 不可信 / 估算冒充回测 | 是 |
| +24.15% 来源 | +24.15%（任务引用数值） | 当前仓库代码 / README / git 历史均无 24.15 | 无 | UNVERIFIED | UNKNOWN | 不可信 | 是 |
| 原子级 / Raw Log 回测是否真正逐笔计算 | README 声称解析几十 GB 链上原始 Swap Log 捕捉插针 | v3_raw_reality_check.py 读取 raw.csv 并循环 swap，但：p_entry/p_low/p_high/L 初始为 0 且从未更新；state 永远 POOL（state=='ETH' 分支不可达）；PnL 计 | 本地 raw.csv（598 天，~15 GB） | HEURISTIC_ADJUSTMENT | IN_SAMPLE | 不可信 / 非真正逐笔回测 | 是 |
| 现实约束（15 秒延迟 / 5bps 滑点 / Reality Penalty）是否真实进入计算 | README 声称显式引入 5s 采样 + 10s 上链确认延迟惩罚 | 5bps：dual_engine_optimizer.py / wide_range_study.py latency_bias=0.0005 作为固定滑点（经验值，非实测）；15 秒延迟：仅 v3_raw_reality_check.py | 无真实延迟 / 滑点数据 | HEURISTIC_ADJUSTMENT | IN_SAMPLE | 不可信 / 均为经验假设 | 是 |

## 训练 / 验证泄漏分析

| 脚本 | 搜索区间 | 验证窗口 | 重叠 | 未来数据 | 结论 |
|---|---|---|---|---|---|
| wide_range_study.py | range∈[0.08,0.12], risk_thresh∈[0.40,0.70], m_bull∈[50,65], m_bear∈[35,50] | 同一 full_minute_df 全量数据 + 2025-08-24 起 Peak Start Stress Test | 是（搜索与验证用同一份全量数据） | 否（未显式用未来，但无时间分割） | IN_SAMPLE |
| dual_engine_optimizer.py | range∈[0.02,0.05], risk_thresh∈[0.40,0.65] | search_df=full_minute_df.iloc[-260000:]（最近 ~6 个月），final=full_minute_df 全量 | 是（最终验证包含搜索用的最后 6 个月） | 否 | OVERLAP |
| demeter_asymmetric_backtest.py | 无搜索（固定 ±4% 区间 + 固定阈值 0.45） | 全量 365 天 | 不适用（无参数搜索，但无独立验证集） | 否 | IN_SAMPLE |
| v3_hunter_monte_carlo.py | 无参数搜索（固定 0.55 / EMA / RSI 阈值） | 10 次随机 25-35 天窗口 | 信号基于全量数据预计算，评估窗口内使用全量信号 → 存在信号泄漏风险 | 潜在（信号用全量数据，包含评估窗口） | OVERLAP |
| v3_raw_reality_check.py | 无 | 声称全量 raw，实际硬编码结果 | 不适用（结果非计算产生） | 否 | IN_SAMPLE |

## 数据路径映射

- 旧脚本引用：`uniswap_data/UNIV3_DATA`
- 本仓库内是否存在：False
- 本机实际数据根：D:\gitee\uniswap-data\UNIV3_DATA
- 结论：旧脚本引用的相对路径在本仓库 checkout 内不存在（.gitignore 忽略 uniswap_data/），因此从仓库根直接运行时无法读取数据；本机实际数据位于 D:\gitee\uniswap-data\UNIV3_DATA（Harness 配置根）

## 模型可复现性

- 模型文件存在：True
- 读取状态：DEGRADED_STATIC
- 顶层键：['xgb', 'ga', 'features']
- 模型类：xgboost.sklearn.XGBClassifier
- 特征数：19
- 特征：RSI_14, ADX_14, ADXR_14_2, DMP_14, DMN_14, NATR_14 ...
- GA 参数：[46.78085945837288, 80.70883111005968, 1.5875745741755496]
- 仓库内训练脚本：False
- 可完整重建：False
- 缺失项：
  - 模型训练脚本（仓库中不存在能生成 models_15m.pkl 的脚本）
  - 标签定义（LVR 风险标签如何从数据生成）
  - 训练窗口 / 数据版本
  - 随机种子 / 早停 / 验证划分

## 总体结论

- readme_403_trustworthy: False
- mc_917_reproducible: False
- raw_atomic_real: False
- model_retrainable: False
- next_rerun_recommendation: 最值得重跑：wide_range_study.py 的 wide-range（±8.13%）策略在严格 OOS 分割下做全量逐笔回测（该参数有优化器产物与脚本支撑，是唯一有完整代码路径的核心结论）；其次为 v3_hunter_monte_carlo.py 增加独立样本外信号重建。
