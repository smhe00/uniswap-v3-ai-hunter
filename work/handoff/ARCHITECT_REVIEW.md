# Architect Review — R0-T001 Iteration 1

## Decision

`CHANGES_REQUIRED`

R0-T001 的主体审计方向正确，关键旧结论（+40.3%、91.7%、Raw/原子级结果、模型不可重训）基本识别到位，但本轮不能 PASS。原因包括 1 项范围违规、1 项测试覆盖缺口和 1 项 OOS（Out-of-Sample，样本外）判定过度。

## Reviewed Snapshot

- remote_head: `5ddaa3fba24f2f8bc4ec3f5fab0588a0a7ea2ab7`
- task_id: `R0-T001`
- iteration reviewed: `1`
- consumed harness handoff: `R0-T001-HARNESS-20260823-001`

## Findings

### F1 — Allowed Files 违规（必须修）

`04ac458c... -> 5ddaa3fb...` 的实际 diff 除任务允许文件外，还修改了：

- `.gitignore`

新增：

```text
.pytest_cache/
pytest-cache-files-*/
```

原任务 §9 明确限定 Allowed Files，不包含 `.gitignore`；§16 也禁止修改 Allowed Files 之外文件。因此必须撤销这两行，使 `.gitignore` 恢复到任务基线 `04ac458c4154ceb5780980a1b2c2eb45c0f6b54b` 的内容。

本轮 Iteration 2 **仅为撤销该越界改动**，特别授权 Harness 修改 `.gitignore`，且只允许删除本轮新增的上述两行，不得做其他整理。

### F2 — “模型缺失时降级”测试未真正覆盖（必须修）

`tests/test_r0_t001_legacy_audit.py::test_missing_model_degrades_to_unverified` 当前只是调用真实仓库模型并断言返回值包含 `exists`：

```python
meta = audit._analyze_model_metadata()
assert "exists" in meta
```

这没有构造“模型文件缺失”的场景，也没有验证状态为 `UNVERIFIED`，因此没有满足 CURRENT_TASK §12.6。

修复要求：

1. 让 `_analyze_model_metadata()` 支持可测试的模型路径注入，或以等价、无副作用方式临时指向不存在路径；
2. 新测试必须真实构造模型缺失；
3. 明确断言：`exists is False` 且 `status == "UNVERIFIED"`；
4. 不得重命名/移动真实 `models_15m.pkl` 来制造缺失场景。

### F3 — `v3_hunter_monte_carlo.py` 的 OOS 判定过度（必须修）

本轮报告将该脚本标为 `OVERLAP`，理由是“信号全量预计算含评估窗口”。这个理由本身不足以证明未来数据泄漏：

- 技术指标由历史价格滚动/重采样计算；
- 原脚本用 `pd.merge_asof(..., direction='backward')` 将最近的既有信号并入 Swap；
- “在整段历史上预先算出因果指标”不等于“使用未来值”。

真正无法确认的是 `models_15m.pkl` 的训练来源：训练脚本、标签、训练窗口和切分均缺失。因此无法证明随机测试窗口是否属于模型严格样本外，也无法证明一定重叠。

修复要求：

- `v3_hunter_monte_carlo.py` 的总体 OOS 状态从 `OVERLAP` 改为 `UNKNOWN`；
- `future_data` 改为 `UNKNOWN` 或等价明确表述；
- 说明：技术指标管线本身未发现明确 look-ahead，但模型训练窗口未知，因此无法判定严格 OOS；
- `91.7%` 仍维持“不可能由当前 10 次运行直接产生、当前代码不可复现”的结论，这一结论不受上述修改影响。

## Accepted Findings（本轮可保留）

以下核心审计判断有充分代码证据，可继续保留：

1. README 的 `$29,270 / +40.3% / +45.3%` 当前缺少完整可复现证据链；
2. `91.7%` 与当前 `range(10)` 的随机窗口代码不匹配；
3. `v3_raw_reality_check.py` 最终结果使用 `32.88 * 0.85`，不能称为真实逐笔 Raw Log（原始日志）收益结果；
4. `wide_range_study.py` 的参数搜索与所谓验证使用同源数据，属于 `IN_SAMPLE`（样本内）；
5. `dual_engine_optimizer.py` 最终验证包含参数搜索窗口，属于 `OVERLAP`（重叠）；
6. 当前仓库缺少 `models_15m.pkl` 的完整训练流程，因此模型无法从仓库独立重训。

## Iteration 2 Narrow Fix Scope

允许修改：

- `.gitignore` —— **仅撤销 Iteration 1 新增的两行**；
- `research/r0_t001_legacy_audit.py`；
- `tests/test_r0_t001_legacy_audit.py`；
- `results/r0_t001/legacy_claim_audit.json`；
- `results/r0_t001/legacy_claim_audit.md`；
- `work/handoff/HARNESS_REPORT.md`；
- `work/control/WORKFLOW_STATE.yaml`。

禁止修改其他文件，禁止顺手格式化旧代码。

## Required Validation

至少重新运行：

```bash
python research/r0_t001_legacy_audit.py
python -m pytest -q -p no:cacheprovider tests/test_r0_t001_legacy_audit.py
```

并在 Harness Report 中额外记录：

1. `.gitignore` 已恢复至基线，不再包含本任务越界新增项；
2. 缺失模型测试的真实断言结果；
3. Monte Carlo OOS 状态已改为 `UNKNOWN` 的理由；
4. 更新后的测试总数和 PASS 数。

## Next State

保持同一 `task_id = R0-T001`，进入 iteration 2。完成窄范围修正后发布新的 `REVIEW_READY`；不得自行开始 R0-T002。
