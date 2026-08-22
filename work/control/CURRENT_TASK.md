# R0-T001 — Legacy Claim Provenance & Reproducibility Audit

> 研究阶段：R0（旧系统可信度审计）  
> 任务类型：代码审计 + 本地数据可复现性检查  
> 执行角色：DeepSeek Harness  
> 安全边界：RESEARCH ONLY / NO ONCHAIN WRITE

## 1. 研究问题

对当前 `main` 中的旧版 Uniswap V3 AI Hunter 进行系统审计，回答：

1. README 中每一个核心收益、胜率、区间与风险结论，具体来自哪段代码、哪组数据、哪种计算；
2. 哪些数字是直接回测计算，哪些是近似、硬编码、经验修正或人工汇总；
3. 哪些结果存在训练集 / 参数搜索集与验证集重叠；
4. 哪些模型、参数和结果当前可以完整复现，哪些不可以；
5. 为下一任务决定：哪些旧策略值得做严格重跑，哪些旧结论应废弃。

本任务不要求重新设计策略，也不要求立即完成全年逐笔重型回测。

## 2. 必须读取

- `README.md`
- `lp_smart_agent.py`
- `dual_engine_optimizer.py`
- `wide_range_study.py`
- `demeter_asymmetric_backtest.py`
- `v3_raw_reality_check.py`
- `v3_hunter_monte_carlo.py`
- `v3_experimental_15m_tag/models_15m.pkl`
- `UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md`
- `LOCAL_HARNESS_INIT.md`
- `work/control/WORKFLOW_STATE.yaml`

## 3. 术语

报告中首次出现英文缩写时必须写全称和中文解释。例如：

- **LP = Liquidity Provider / Liquidity Position，流动性提供者 / 流动性仓位**；
- **ROI = Return on Investment，投资回报率**；
- **LVR = Loss Versus Rebalancing，相对再平衡损失**；
- **RSI = Relative Strength Index，相对强弱指数**；
- **EMA = Exponential Moving Average，指数移动平均线**；
- **NATR = Normalized Average True Range，归一化平均真实波幅**；
- **ADX = Average Directional Index，平均趋向指数**；
- **XGBoost = Extreme Gradient Boosting，极端梯度提升模型**；
- **GA = Genetic Algorithm，遗传算法**；
- **OOS = Out-of-Sample，样本外**。

## 4. 本地数据要求

允许读取 `.local/harness.yaml` 中配置的本地只读数据目录。

本任务只要求：

1. 验证旧脚本引用的数据路径是否能映射到本机已有数据；
2. 记录相关数据覆盖范围、文件数和缺口；
3. 如无需全量读取即可确定某项结论来源，不得为“凑全量回测”浪费计算资源；
4. 若需要执行旧脚本才能确认某项来源，可以运行，但必须记录完整命令、运行范围和实际输出；
5. 禁止修改原始数据。

## 5. 必须审计的 README / 旧系统核心结论

至少逐项建立证据链：

1. `RANGE_PCT = ±8.13%` 的来源；
2. `XGB_RISK_THRESHOLD = 0.57` 的来源；
3. `4 天再平衡冷却期` 的来源；
4. `最终净值 $29,270`；
5. `总 ROI +40.3%`；
6. `相对 Alpha +45.3%`；
7. `Monte Carlo 胜率 91.7%`；
8. 旧文档 / commit 中出现的 `+40.44%`、`+47.65%`、`+32.88%`、`+24.15%` 等关键数值；
9. `原子级 / Raw Log 回测` 是否真正逐笔计算；
10. `15 秒延迟`、`5 bps 滑点`、`Reality Penalty` 等现实约束是否真实进入计算。

其中：

- **Alpha = Excess Return，超额收益**；
- **Monte Carlo Simulation = 蒙特卡罗模拟**；
- **Raw Log = 原始链上事件日志**；
- **bps = basis points，基点，1 bps = 0.01%**。

## 6. 模型可复现性检查

对 `models_15m.pkl` 只做本地受控检查，至少报告：

1. pickle 内顶层键；
2. XGBoost 模型类型；
3. feature（特征）名称清单；
4. GA 参数内容；
5. 是否存在训练脚本；
6. 是否能从仓库代码和本地数据完整重建该模型；
7. 若不能，缺失什么：标签定义、训练窗口、随机种子、参数、数据版本等。

禁止为了“可复现”而猜测缺失训练流程。

## 7. 训练 / 验证泄漏检查

对每个优化 / 回测脚本明确标注：

- 参数搜索区间；
- 最终验证区间；
- 两者是否重叠；
- 是否使用未来数据；
- 是否属于严格 OOS（Out-of-Sample，样本外）验证。

只允许以下结论标签：

- `STRICT_OOS`：严格样本外；
- `OVERLAP`：搜索 / 训练和验证存在重叠；
- `IN_SAMPLE`：完全样本内；
- `UNKNOWN`：代码 / 数据不足以判断。

## 8. 数值来源分类

每个核心数值必须归入且只能归入以下一种：

- `DIRECT_COMPUTE`：代码直接从数据计算；
- `OPTIMIZER_OUTPUT`：优化器输出；
- `HARD_CODED`：硬编码常数或固定结果；
- `HEURISTIC_ADJUSTMENT`：经验系数 / 人工修正；
- `MANUAL_SUMMARY`：README 人工汇总，代码中找不到完整证据；
- `UNVERIFIED`：当前无法验证。

## 9. Allowed Files

DeepSeek Harness 仅允许修改 / 新增：

- `research/r0_t001_legacy_audit.py`
- `tests/test_r0_t001_legacy_audit.py`
- `results/r0_t001/legacy_claim_audit.json`
- `results/r0_t001/legacy_claim_audit.md`
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止修改：

- `README.md`
- 所有旧策略 / 回测脚本；
- `models_15m.pkl`；
- 协议文件；
- `LOCAL_HARNESS_INIT.md`；
- 本地原始数据。

## 10. 必须实现的审计工具

`research/r0_t001_legacy_audit.py` 至少要能：

1. 扫描上述旧脚本和 README 的关键数值；
2. 输出 Claim Matrix（结论证据矩阵）；
3. 标记数值来源分类；
4. 标记训练 / 验证重叠风险；
5. 输出 JSON 和 Markdown 两种结果；
6. 对模型元数据做受控读取；
7. 不依赖链上写权限。

## 11. 必跑命令

至少运行：

```bash
python research/r0_t001_legacy_audit.py
pytest -q tests/test_r0_t001_legacy_audit.py
```

如果 Windows 环境中的 Python 命令不同，可替换，但报告必须记录实际命令。

## 12. 测试要求

测试至少覆盖：

1. HARD_CODED 数值能被识别；
2. HEURISTIC_ADJUSTMENT 能被识别；
3. 训练 / 验证区间重叠能被标记为 `OVERLAP`；
4. README 中存在但代码无法建立证据的结论不能自动判为可信；
5. 输出 JSON schema 稳定；
6. 缺少本地模型 / 数据时脚本应明确降级为 `UNVERIFIED`，不能伪造 PASS。

## 13. 必须输出的结果

`legacy_claim_audit.md` 至少包含表格：

| Claim / 结论 | README 数值 | 代码来源 | 数据来源 | 分类 | OOS 状态 | 当前可信度 | 是否建议重跑 |
|---|---:|---|---|---|---|---|---|

并给出三张清单：

### A. 可直接继承
有充分代码与数据证据，可作为后续研究基准。

### B. 需要严格重跑
思想 / 代码有价值，但当前结果受重叠、近似或不完整验证影响。

### C. 应废弃旧数字
硬编码、经验修正冒充真实结果、无法找到证据链或不可复现。

## 14. Harness Report 必须额外回答

除协议标准结构外，明确回答：

1. 旧项目 README 的 +40.3% 是否可信；
2. 91.7% Monte Carlo 胜率是否可由当前代码复现；
3. Raw / 原子级结果是否真正由逐笔 Swap 计算；
4. `models_15m.pkl` 是否可重训；
5. 下一任务最值得花计算资源重跑哪一项。

## 15. 验收标准

任务 PASS 需要同时满足：

- Claim Matrix 覆盖第 5 节全部核心结论；
- 每个核心结论都有明确来源分类；
- OOS 状态分类完整；
- 模型可复现性有证据；
- 自动化审计工具与测试通过；
- 没有修改旧策略代码或原始数据；
- 没有把近似 / 抽样 / 硬编码结果描述成真实全量回测；
- `HARNESS_REPORT.md` 提供完整命令和本地数据证据。

## 16. 禁止项

- 禁止链上写操作；
- 禁止交易、Swap、增加 / 移除真实流动性；
- 禁止修改钱包 / 私钥 / RPC 凭据；
- 禁止上传原始历史数据；
- 禁止 force push、rebase、stash、reset；
- 禁止自行进入下一任务；
- 禁止修改本任务 Allowed Files 之外的文件。

## 17. 交接

完成后：

1. 更新 `work/handoff/HARNESS_REPORT.md`；
2. 将 `WORKFLOW_STATE.yaml` 更新为：
   - `handoff_seq: 2`
   - 新唯一 `handoff_id`
   - `state: REVIEW_READY`
   - `owner: architect`
   - `authorized_next: []`
3. 普通 commit + 普通 push 到 `main`；
4. 停止，等待 Architect Review。

从本任务开始，不再允许初始化阶段那种无正式 handoff 的 ad-hoc commit。
