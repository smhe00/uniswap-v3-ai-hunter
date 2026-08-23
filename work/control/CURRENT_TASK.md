# R0-T003 — ETH Regime Attribution of Frozen Legacy vs Benchmarks

> 研究阶段：R0（旧系统可信度验证，延伸）
> 任务类型：ETH 行情阶段（下降/震荡/上升）归因分析
> 执行角色：DeepSeek Harness
> 安全边界：RESEARCH ONLY / NO ONCHAIN WRITE

## 1. 研究问题

R0-T002 已确认：Frozen Legacy 相对 Always LP 显著占优（+23.54%），相对 Always ETH 落后（-3.29%），其超额主要来自 SAFE 避险择时。

本任务进一步回答用户问题：

> ETH 在**下降**和**震荡**阶段，Frozen Legacy、Always LP、Always ETH、Always USDC、50/50 各策略的效果（收益 / 回撤 / 超额）分别是多少？

具体分解：

1. 把 OOS 窗口 `2026-03-14 .. 2026-08-21 UTC` 按 ETH 行情划分为 `上升 / 下降 / 震荡` regime；
2. 对每个 regime，统计各策略在该阶段内的收益、最大回撤、相对 Always ETH / Always USDC 的超额；
3. 回答：Frozen Legacy 的避险优势（相对 Always LP）是否主要来自下降/震荡阶段？它在下降阶段是否真的跑赢 Always ETH？

## 2. 方法与数据源

### 2.1 复用 R0-T002 已产出结果

本任务不重新训练、不重跑策略（策略净值曲线与 R0-T002 iteration 4 完全一致，P0-1 只加校验不改策略逻辑）。直接复用：

- `results/r0_t002/post_freeze_oos_equity.csv`：7 条每日净值曲线
- `results/r0_t002/post_freeze_oos.json`：完整指标（可交叉核对）
- 池 minute 价格（ETH 行情，用于 regime 划分）：`load_pool_minute()` 的 `price` 列

### 2.2 Regime 划分规则（ex-post attribution，透明且可复现）

用 ETH 日线收盘价（池 minute price 日线 last）定义 regime，逐日标注：

- 用**过去 N 日（默认 10 日）日收益**判定趋势：
  - `ret_10d >= +2%` → `上升（bull）`
  - `ret_10d <= -2%` → `下降（bear）`
  - 其余 → `震荡（range）`
- regime 为**状态序列**（连续同 regime 日合并为阶段），输出每个阶段的起止、天数、ETH 阶段收益。

这是**事后归因**（ex-post attribution），不用于任何策略决策，因此用未来确定阶段边界不构成 look-ahead（决策因果性已在 R0-T002 保证）。

### 2.3 必须输出的统计

按 regime（上升/下降/震荡）分三张表，每张表列各策略：

- 该 regime 内收益（`end_nav / start_nav - 1`）
- 该 regime 内最大回撤
- 相对 Always ETH 超额
- 相对 Always USDC 超额
- 该 regime 占 OOS 天数比例

另输出：

- regime 时间线（起止 / 天数 / ETH 收益）
- Frozen Legacy 在各 regime 的 ACTIVE/SAFE 状态占比（如果可从 equity 推断；否则只报收益）
- 汇总结论（下降/震荡阶段哪个策略最优）

## 3. 必须读取

- `results/r0_t002/post_freeze_oos_equity.csv`
- `results/r0_t002/post_freeze_oos.json`
- `research/r0_t002_post_freeze_oos.py`（复用 `load_pool_minute` / `OOS_START` / `OOS_END`）
- `work/handoff/HARNESS_REPORT.md`（R0-T002 结论）

## 4. 禁止项

- 禁止重新优化任何参数 / 阈值 / 冷却期；
- 禁止重新训练模型；
- 禁止修改 R0-T002 的策略逻辑或已产出净值；
- 禁止把 ex-post regime 划分用于任何"策略择时"结论（这是归因，不是交易信号）；
- 禁止用 README 旧数字校准结果。

## 5. Allowed Files

DeepSeek Harness 仅允许新增 / 修改：

- `research/r0_t003_regime_attribution.py`（新增）
- `tests/test_r0_t003_regime_attribution.py`（新增）
- `results/r0_t003/regime_attribution.json`（新增）
- `results/r0_t003/regime_attribution.md`（新增）
- `work/control/CURRENT_TASK.md`（本任务书）
- `work/handoff/HARNESS_REPORT.md`（R0-T003 报告）
- `work/control/WORKFLOW_STATE.yaml`

禁止修改：R0-T002 相关文件、legacy 策略文件、模型文件、README、协议、原始数据。

## 6. 必跑命令

```bash
# venv（含 pandas_ta / demeter / xgboost）
PYTHONIOENCODING=utf-8 PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\venv\Scripts\python.exe" research/r0_t003_regime_attribution.py

# 测试（v3.12）
PYTHONPATH=research
"C:\Users\peter\Documents\V3_Strategy\v3.12\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_r0_t003_regime_attribution.py
```

## 7. 验收标准

- regime 划分规则透明、可复现、无未来函数参与策略决策；
- 三张 regime 表（上升/下降/震荡）包含全部 5+ 策略（Frozen gross / Frozen legacy-cost / Always LP / Always ETH / Always USDC / 50/50）；
- 每个 regime 至少覆盖若干连续日（有实际数据，非空表）；
- 给出明确结论：下降/震荡阶段最优策略是谁、Frozen 相对各基准的超额方向；
- 测试通过（regime 划分 sanity + 表结构 + 收益计算正确性）。
