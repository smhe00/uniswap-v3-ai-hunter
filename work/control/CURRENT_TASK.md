# R0-T002 — Post-Freeze Strict OOS Validation of Legacy AI Hunter

> 研究阶段：R0（旧系统可信度验证）  
> 任务类型：冻结参数的严格时间外回测  
> 执行角色：DeepSeek Harness  
> 安全边界：RESEARCH ONLY / NO ONCHAIN WRITE

## 1. 研究问题

旧版 `uniswap-v3-ai-hunter` 在 2026-03-13 已经形成所谓“最终版本”。本任务不再重新优化旧参数，而是把旧系统视为一个已经冻结的候选策略，回答：

> 在旧代码、旧模型和旧参数冻结之后才发生的 2026-03-14 至 2026-08-21 数据上，旧版 LP / ETH / USDC 三态切换策略是否仍能产生有意义的样本外优势？

其中：

- **LP = Liquidity Provider / Liquidity Position，流动性提供者 / 流动性仓位**；
- **OOS = Out-of-Sample，样本外验证**；
- **ETH = Ether，以太币**；
- **USDC = USD Coin，美元稳定币**。

## 2. 冻结点与严格样本外窗口

旧项目冻结参考：2026-03-13。

本任务统一使用：

```text
OOS Start = 2026-03-14 00:00:00 UTC
OOS End   = 2026-08-21 23:59:59 UTC（或本地数据可用的最后完整时点）
```

必须满足：

1. 不允许使用 2026-03-14 之后的数据重新训练模型；
2. 不允许使用 2026-03-14 之后的数据重新优化任何阈值、区间、冷却期或成本参数；
3. 如果实际数据末端早于 2026-08-21 23:59:59 UTC，报告必须写明最后完整时点；
4. 如果模型或参数文件的 Git 历史证明其冻结时间晚于 2026-03-13，则立即报告 `BLOCKED`，不得伪称严格 OOS。

## 3. 必须读取

- `README.md`
- `lp_smart_agent.py`
- `dual_engine_optimizer.py`
- `wide_range_study.py`
- `demeter_asymmetric_backtest.py`
- `v3_experimental_15m_tag/models_15m.pkl`
- `results/r0_t001/legacy_claim_audit.json`
- `results/r0_t001/legacy_claim_audit.md`
- `work/handoff/ARCHITECT_REVIEW.md`
- `work/control/WORKFLOW_STATE.yaml`
- `UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md`

## 4. 冻结策略参数

本任务禁止优化，默认按生产脚本冻结值：

```text
RANGE_PCT            = 0.0813      # ±8.13%
REBALANCE_DELAY_DAYS = 4
XGB_RISK_THRESHOLD   = 0.57
MACRO_BULL_RSI       = 52
MACRO_BEAR_RSI       = 50
VOL_GUARD_NATR       = 2.0
```

其中：

- **XGBoost = Extreme Gradient Boosting，极端梯度提升模型**；
- **RSI = Relative Strength Index，相对强弱指数**；
- **NATR = Normalized Average True Range，归一化平均真实波幅**。

如果旧代码存在多个互相冲突的冻结值，以 `lp_smart_agent.py` 当前 main 的生产参数为主，同时在报告中列出冲突，不得自行调和成“更优参数”。

## 5. 数据源

优先使用本地只读数据：

### 5.1 Uniswap V3 池数据

Arbitrum WETH/USDC 0.05% 池：

```text
Pool = 0xC6962004f452bE9203591991D15f6b388e09E8D0
```

至少使用本地 `minute.csv`；如当前回测引擎确有需要，可使用 `raw.csv` 做抽样交叉验证，但本任务不要求全量逐笔 Raw 重放。

### 5.2 Binance 行情

生产 `lp_smart_agent.py` 使用 Binance ETHUSDT 15 分钟和 4 小时行情作为信号输入。若本地 `BINANCE_KDATA` 可完整提供本任务窗口，应优先复现生产信号来源。

如果由于历史数据结构无法精确复现生产输入，可增加一个“Pool-derived signal”对照，但必须与“Production-like Binance signal”分栏报告，禁止混成一个结果。

## 6. 必须比较的策略

所有策略从同一初始 USDC 等值资本开始，建议标准化为 `10,000 USDC`；如果回测框架需要其他金额，可使用不同数值，但各策略起始净值必须完全相同。

### A. Frozen Legacy AI Hunter

冻结旧模型和第 4 节参数，不调参。

状态逻辑按旧生产语义：

```text
ACTIVE → LP
SAFE + Bull → 100% ETH
SAFE + Bear → 100% USDC
SAFE + neither → Keep Ratio
```

### B. Always LP

始终做同样 `±8.13%` 宽度的 Uniswap V3 LP，不使用 AI 风险退出。

再平衡规则必须固定、明确，建议使用旧项目 4 天冷却 + 出区间后才允许重建；禁止为使结果更好临时修改。

### C. Always ETH

全程持有 ETH。

### D. Always USDC

全程持有 USDC。

### E. 50/50 ETH-USDC Buy-and-Hold

初始按价值 50/50 分配后不再平衡，用于区分 LP 收益与简单混合持仓收益。

其中：

- **Buy-and-Hold = 买入并持有**。

## 7. 成本处理

必须至少输出两套结果：

### 7.1 Gross Result（毛收益）

不扣人为假设的 Gas / latency penalty，用于观察纯策略结构。

- **Gas Fee = 链上执行手续费**。

### 7.2 Legacy-Cost Result（旧成本假设）

严格使用旧代码已经存在的成本假设，不得重新优化，例如：

```text
latency_bias = 0.0005  # 5 bps
exit balance deduction = 0.0002（若对应旧逻辑确实使用）
```

其中：

- **bps = basis points，基点；1 bps = 0.01%**；
- `latency_bias` 只能称为“旧延迟/滑点假设”，禁止称为真实历史 Gas 或真实成交滑点。

如 Demeter 或回测框架已经从池数据计算真实 LP 交易手续费收入，应保留该计算并在报告中说明。

## 8. 禁止重新优化

本任务禁止：

- Optuna 搜索；
- 网格搜索；
- 遗传算法重新寻参；
- 根据 2026-03-14 之后结果手工改变阈值；
- 选择性删除亏损区间；
- 根据结果挑选更好的开始日期；
- 用 README 的旧收益数字校准结果。

其中：

- **Optuna = 自动超参数优化框架**；
- **GA = Genetic Algorithm，遗传算法**。

## 9. 时间与信号因果性

所有信号在时间 `t` 的决策只能使用 `t` 或 `t` 之前已经完成的数据。

特别检查：

1. 15 分钟 bar 未收盘前不得使用该 bar 的最终 close/high/low；
2. 4 小时 bar 同理；
3. 使用 `merge_asof` 时必须 `direction='backward'`；
4. 禁止使用 centered rolling window；
5. 禁止使用未来填充；
6. 所有 forward-looking 标签在本任务中都不应参与决策。

## 10. 必须输出的指标

每个策略至少输出：

- 起始净值；
- 结束净值；
- Total Return，总收益率；
- Annualized Return，年化收益率；
- Maximum Drawdown，最大回撤；
- Sharpe Ratio，夏普比率；
- Sortino Ratio，索提诺比率；
- 交易 / 状态切换次数；
- LP 在池时间占比；
- 出区间时间占比；
- 累计 LP Fee，流动性手续费收入；
- 累计模拟成本；
- Frozen Legacy 相对 Always LP / Always ETH / Always USDC 的超额收益。

其中：

- **Sharpe Ratio = 夏普比率，每承担一单位波动获得的风险调整收益**；
- **Sortino Ratio = 索提诺比率，只惩罚下行波动的风险调整收益指标**；
- **Maximum Drawdown = 最大回撤，从历史高点到随后低点的最大跌幅**。

## 11. 必须输出的事件统计

Frozen Legacy 至少报告：

- ACTIVE → SAFE 次数；
- SAFE → ACTIVE 次数；
- SAFE 时进入 ETH 次数；
- SAFE 时进入 USDC 次数；
- SAFE 时 Keep Ratio 次数；
- 因 4 天冷却导致未重建次数；
- 因出区间停止赚手续费的累计时间。

这些统计用于检查策略是否真的发生状态切换，避免“回测跑了但实际一直在一个状态”。

## 12. 最小交叉验证

从 OOS 窗口随机或固定选择至少 3 个不同市场阶段，每段至少 24 小时，检查：

1. 策略状态序列；
2. 价格序列；
3. LP 是否在区间；
4. 手续费是否只在有效区间累计；
5. 资产净值变化是否与状态一致。

这不是参数优化，仅用于数值 sanity check（合理性检查）。

## 13. Allowed Files

DeepSeek Harness 仅允许新增 / 修改：

- `research/r0_t002_post_freeze_oos.py`
- `tests/test_r0_t002_post_freeze_oos.py`
- `results/r0_t002/post_freeze_oos.json`
- `results/r0_t002/post_freeze_oos.md`
- `results/r0_t002/post_freeze_oos_equity.csv`（仅允许小型汇总曲线，不上传原始数据）
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

如确需新增一个很小的通用辅助模块，必须先报告 `BLOCKED` 或 `USER_ACTION_REQUIRED`，不得自行扩大 Allowed Files。

禁止修改：

- 所有 legacy 策略文件；
- README；
- 模型文件；
- 协议文件；
- 本地原始数据。

## 14. 测试要求

至少覆盖：

1. OOS 起始日期固定为 2026-03-14，不可被配置漂移；
2. 冻结参数与 `lp_smart_agent.py` 一致；
3. 不存在任何优化器调用；
4. 信号只能 backward merge；
5. 各基准策略初始净值完全一致；
6. LP 出区间时不继续累计手续费；
7. 成本模型 Gross / Legacy-Cost 分离；
8. 缺少 Binance 或 Uniswap 数据时明确失败 / 降级，禁止用模拟数据冒充真实 OOS；
9. 输出 schema 稳定；
10. 无链上写路径。

## 15. 本地运行与依赖

允许 Harness 在本地研究环境安装本任务必要 Python 包，例如 Demeter，但：

- 不得修改全局系统环境以外的项目文件；
- 不得提交虚拟环境；
- 必须在报告记录 Python、Demeter、pandas、numpy、xgboost 等实际版本；
- 如依赖安装失败且无法完成可信回测，报告 `BLOCKED`，不得用手写收益数字替代。

## 16. 必跑命令

至少：

```bash
python research/r0_t002_post_freeze_oos.py
python -m pytest -q -p no:cacheprovider tests/test_r0_t002_post_freeze_oos.py
```

若 Windows 命令不同，可替换，但必须记录实际命令。

## 17. 结果裁决原则

本任务不是为了证明旧策略一定有效。

### 结果 A：Frozen Legacy 明显优于简单基准

保留旧三态架构作为后续研究 Benchmark（基准策略），进入更严格 Raw / 逐笔验证。

### 结果 B：优势很小或不稳定

旧系统只作为历史参考；下一阶段转向新的 Fee Yield + Volatility + Trend 经济驱动框架。

其中：

- **Fee Yield = 手续费收益率**；
- **Volatility = 波动率**；
- **Trend = 趋势**。

### 结果 C：明显落后

停止继续投入旧 AI Hunter 模型重构，直接进入新的 LP / ETH / USDC 状态研究框架。

## 18. Harness Report 必须回答

1. Frozen Legacy 在严格 post-freeze OOS 上最终收益是多少？
2. 它是否战胜 Always LP？差多少？
3. 它是否战胜 Always ETH / USDC？
4. Gross 和 Legacy-Cost 两套结果差多少？
5. 最大回撤是否改善？
6. 三态切换是否真实发生，还是大部分时间卡在某一状态？
7. `±8.13%` 是否在 post-freeze OOS 中仍具有合理性？
8. 下一步应该继续深挖旧模型，还是转向新的经济变量框架？

## 19. 交接规则

完成后：

1. 更新 `work/handoff/HARNESS_REPORT.md`；
2. 将 `WORKFLOW_STATE.yaml` 更新为新的 `REVIEW_READY`；
3. `handoff_seq + 1`；
4. 新唯一 `handoff_id`；
5. `owner=architect`；
6. `authorized_next=[]`；
7. 普通 commit + 普通 push；
8. 停止，等待 Architect Review。
