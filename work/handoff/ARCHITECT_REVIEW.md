# Architect Review — R0-T001 Iteration 2

## Decision

`CHANGES_REQUIRED`

Iteration 2 已正确修复上一轮 F1 / F2，并基本修正 F3，但仍有 1 个一致性缺口，因此本轮不能 PASS。

## Reviewed Snapshot

- remote_head: `f60d663de82dfc0a88adc0f6514d31fe233a6bc2`
- task_id: `R0-T001`
- iteration reviewed: `2`
- consumed harness handoff: `R0-T001-HARNESS-20260823-002`

## Accepted Fixes

### F1 — `.gitignore` 越界修改

已正确撤销上一轮新增的：

```text
.pytest_cache/
pytest-cache-files-*/
```

本轮 diff 仅删除这两行，符合 Architect 的窄范围授权。

### F2 — 模型缺失降级测试

已正确增加 `model_path` 注入，并真实构造不存在的模型路径；测试明确断言：

```text
exists is False
status == "UNVERIFIED"
```

未移动或重命名真实 `models_15m.pkl`。该项修复满足要求。

### F3 — Monte Carlo 脚本级 OOS 判定

`v3_hunter_monte_carlo.py` 的 leakage matrix 已从 `OVERLAP` 修正为 `UNKNOWN`，并明确说明：

- 技术指标管线未发现明确 look-ahead；
- `merge_asof(direction='backward')` 本身没有证明未来数据使用；
- 由于 `models_15m.pkl` 的训练窗口 / 标签 / 切分未知，无法证明随机测试窗口属于严格样本外。

该判断正确。

## Remaining Finding

### F4 — Claim Matrix 与 Leakage Matrix 的 OOS 状态仍不一致（必须修）

生成产物 `results/r0_t001/legacy_claim_audit.json` 中：

```text
claim_id = R0-T001-C7
claim = Monte Carlo 胜率 91.7% 来源
oos_status = OVERLAP
```

但同一产物的 leakage matrix 已把 `v3_hunter_monte_carlo.py` 标为：

```text
strict_oos = UNKNOWN
future_data = UNKNOWN
```

Harness Report 的 Claim Matrix 也仍显示 `91.7% | ... | OVERLAP`，与本轮 F3 的修正结论冲突。

这不是新的研究判断问题，而是同一结论在不同输出层之间没有同步。

## Iteration 3 Narrow Fix Scope

只允许修改：

- `research/r0_t001_legacy_audit.py`
- `tests/test_r0_t001_legacy_audit.py`
- `results/r0_t001/legacy_claim_audit.json`
- `results/r0_t001/legacy_claim_audit.md`
- `work/handoff/HARNESS_REPORT.md`
- `work/control/WORKFLOW_STATE.yaml`

禁止修改 `.gitignore` 和其他任何文件。

## Required Fix

1. 将 Claim `R0-T001-C7` 的 `oos_status` 改为 `UNKNOWN`；
2. 生成的 Markdown Claim Matrix 和 Harness Report 中相应行也必须为 `UNKNOWN`；
3. 保留“91.7% 无法由当前 `range(10)` 代码直接产生”的可信度判断，不因 OOS 状态修正而改变；
4. 新增或加强测试，至少明确断言：

```text
claims["R0-T001-C7"]["oos_status"] == UNKNOWN
leaks["v3_hunter_monte_carlo.py"]["strict_oos"] == UNKNOWN
```

5. 重新生成 JSON / Markdown 产物并运行完整 R0-T001 测试。

## Required Validation

至少重新运行：

```bash
python research/r0_t001_legacy_audit.py
python -m pytest -q -p no:cacheprovider tests/test_r0_t001_legacy_audit.py
```

Harness Report 必须记录：

- Claim C7 与 Leakage Matrix 的 OOS 状态均为 `UNKNOWN`；
- 测试总数、PASS 数；
- 实际运行命令。

## Accepted Core Conclusions

以下结论保持不变：

1. README 的 `$29,270 / +40.3% / +45.3%` 当前缺少完整可复现证据链；
2. `91.7%` 不能由当前 10 次随机测试代码直接产生；
3. `v3_raw_reality_check.py` 的最终收益使用 `32.88 * 0.85` 经验修正，不能称为真实逐笔 Raw Log（原始日志）回测收益；
4. `wide_range_study.py` 属于 `IN_SAMPLE`（样本内）；
5. `dual_engine_optimizer.py` 属于 `OVERLAP`（训练/搜索与验证区间重叠）；
6. `models_15m.pkl` 当前无法从仓库独立重训。

## Next State

保持 `task_id = R0-T001`，进入 iteration 3。完成上述单一一致性修正后发布新的 `REVIEW_READY`；不得自行开始下一任务。
