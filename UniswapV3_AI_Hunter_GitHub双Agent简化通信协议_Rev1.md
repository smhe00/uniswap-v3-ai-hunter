# Uniswap V3 AI Hunter GitHub 双 Agent 简化通信协议 Rev1

> 生效日期：2026-08-23  
> 权威仓库：`https://github.com/smhe00/uniswap-v3-ai-hunter`  
> 权威分支：`main`  
> 适用范围：研究、回测、数据验证、策略代码与测试  
> 默认本地轮询周期：180 秒

## 0. 命名约定

本文中的 **Uniswap V3** 仅表示 **Uniswap Version 3，Uniswap 第 3 版协议/资金池**，不是本项目的软件版本号。

为避免与 Uniswap V3 / Uniswap V4 混淆，本项目后续不使用 “V4” 表示研究版本。研究阶段使用 `R0 / R1 / R2 ...`；协议修订使用 `Rev1 / Rev2 ...`。

常用术语：

- **LP = Liquidity Provider / Liquidity Position，流动性提供者 / 流动性仓位**。
- **GitHub main**：本仓库的 `main` 主分支，是两个 Agent 之间唯一的跨机器权威状态。
- **Agent**：自动执行任务的软件代理。本协议有 Architect Agent 和 DeepSeek Harness Agent 两个角色。
- **YAML = YAML Ain't Markup Language，一种结构化配置文件格式**。
- **HEAD**：Git 当前指向的提交。本文的 `remote_head` 指当次读取到的远端 `main` 提交标识。

---

## 1. 目标

建立一个最小、可审计的双 Agent 协作闭环：

```text
Architect / ChatGPT
发布任务到 GitHub main
        ↓
DeepSeek Harness 本地检测新授权
        ↓
读取本地历史数据并实现 / 回测 / 验证
        ↓
只提交代码、测试和报告到 GitHub main
        ↓
用户在 ChatGPT 输入 fetch 或 f
        ↓
Architect 从同一 GitHub main 快照独立 Review
        ↓
PASS / CHANGES_REQUIRED / BLOCKED
        ↓
必要时发布下一任务
```

GitHub 只承担 **控制面和结果检查点**；几十 GB 的本地 Uniswap V3 原始数据、缓存和中间结果属于 **数据面**，继续留在本地，不上传 GitHub。

---

## 2. 角色

### 2.1 Architect Agent：ChatGPT

负责：

1. 总体研究架构与任务拆分；
2. 发布当前唯一授权任务；
3. 定义 Allowed Files（允许修改文件）、输入数据要求和验收标准；
4. 独立检查 Git diff、代码、测试和回测证据；
5. 裁决 `PASS`、`CHANGES_REQUIRED`、`BLOCKED`；
6. 决定下一研究阶段；
7. 维护协议、任务和 Architect Review。

Architect 不假设自己能读取 DeepSeek 机器上的本地数据；所有本地数据结论必须由可审计报告和可复现代码支撑。

### 2.2 DeepSeek Harness Agent：本地执行 Agent

负责：

1. 只执行 GitHub `main` 中明确授权的当前任务；
2. 访问本地 Uniswap V3 历史数据；
3. 编写任务允许范围内的代码与测试；
4. 运行本地回测、数据检查和验证；
5. 输出完整 Harness Report；
6. 普通非强制 push 到 `main`；
7. 完成交接后等待下一次唯一 handoff。

DeepSeek Harness 不负责自行扩大研究范围，不自行修改策略目标，不根据旧聊天记忆启动未授权工作。

---

## 3. GitHub 是唯一跨 Agent 权威状态

以下内容不是跨 Agent 权威状态：

- ChatGPT 旧对话记忆；
- DeepSeek Harness 本地聊天上下文；
- 本地未 push 的 commit；
- 本地未提交 Markdown；
- 本地数据文件本身；
- 任何其他分支上的实验结果。

每次双方做决定，都必须基于同一个远端 `main` 快照。

最低读取集合：

```text
remote_head
work/control/WORKFLOW_STATE.yaml
work/control/CURRENT_TASK.md
work/handoff/HARNESS_REPORT.md
work/handoff/ARCHITECT_REVIEW.md
当前任务涉及的代码和测试 diff
```

若读取过程中 `main` 的 HEAD 改变，本轮快照作废，重新读取；不得混用两个不同 commit 的状态和报告。

---

## 4. 最小文件结构

本协议采用最小化结构：

```text
work/
├── control/
│   ├── WORKFLOW_STATE.yaml
│   └── CURRENT_TASK.md
└── handoff/
    ├── HARNESS_REPORT.md
    └── ARCHITECT_REVIEW.md
```

文件所有权：

### Architect 所有

- `UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md`
- `work/control/CURRENT_TASK.md`
- `work/handoff/ARCHITECT_REVIEW.md`
- Architect 发布任务时的 `work/control/WORKFLOW_STATE.yaml`

### DeepSeek Harness 所有

- 当前任务 `Allowed Files` 中明确列出的实现文件；
- 当前任务 `Allowed Files` 中明确列出的测试文件；
- `work/handoff/HARNESS_REPORT.md`
- DeepSeek Harness 交付结果时的 `work/control/WORKFLOW_STATE.yaml`

`WORKFLOW_STATE.yaml` 是唯一由双方按阶段顺序写入的共享文件。任何时刻只有当前发送方允许修改。

---

## 5. WORKFLOW_STATE 最小格式

建议固定为：

```yaml
protocol: "UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md"
repository: "smhe00/uniswap-v3-ai-hunter"
branch: "main"

handoff_seq: 1
handoff_id: "R0-T001-ARCH-20260823-001"
task_id: "R0-T001"
iteration: 1

state: "HARNESS_READY"
owner: "harness"
authorized_next:
  - "R0-T001"

git_base_commit: "<architect observed remote_head before write>"
last_update: "2026-08-23T00:00:00+08:00"
```

### 规则

1. `handoff_seq` 每次正式交接只能增加，不能回退或复用；
2. `handoff_id` 每次新交接必须唯一；
3. `authorized_next` 是 DeepSeek Harness 唯一可以启动的任务列表；
4. 未被 `authorized_next` 授权的任务一律不得开始；
5. `iteration` 在同一任务要求修改时加 1；
6. `git_base_commit` 记录发送方开始写交接前看到的远端 `main`；
7. 不使用时间戳作为去重依据，真正的去重键是：

```text
handoff_seq + handoff_id + task_id + iteration + state + owner
```

---

## 6. 状态机

只保留六种 GitHub 状态：

### `ARCHITECT_PLANNING`

Architect 正在规划；DeepSeek Harness 不得执行新工作。

```text
owner = architect
authorized_next = []
```

### `HARNESS_READY`

Architect 已发布新任务，DeepSeek Harness 可以执行。

```text
owner = harness
authorized_next = [当前 task_id]
```

### `REVIEW_READY`

DeepSeek Harness 已完成代码 / 本地回测 / 验证并提交报告，等待 Architect Review。

```text
owner = architect
authorized_next = []
```

### `CHANGES_REQUIRED`

Architect 已完成 Review，但当前任务需要窄范围修正。

```text
owner = harness
authorized_next = [同一 task_id]
iteration = iteration + 1
```

### `BLOCKED`

任务因数据、环境、依赖、资源或无法满足验收标准而阻塞。不得伪造结果绕过。

### `USER_ACTION_REQUIRED`

需要用户提供数据路径、权限、环境配置或做明确决策。双方停止自动循环。

不提交 `RUNNING` heartbeat。DeepSeek Harness 开始执行后不需要额外 Git commit 表示“正在运行”。

---

## 7. Architect 发布任务规则

Architect 发布任务前：

1. 读取最新 GitHub `main` 并记录 `pre_write_head`；
2. 从同一 commit 读取 State、Task、Harness Report、Architect Review 和相关代码；
3. 确认没有未消费的 `REVIEW_READY`；
4. 在 `CURRENT_TASK.md` 中明确：
   - `task_id`；
   - 研究问题；
   - 必须读取的文件；
   - 本地数据要求；
   - Allowed Files；
   - 必跑命令；
   - 必须输出的统计指标；
   - 验收标准；
   - 禁止项；
5. 递增 `handoff_seq`，生成唯一 `handoff_id`；
6. 设置 `state=HARNESS_READY`、`owner=harness`、`authorized_next=[task_id]`；
7. 写入前再次确认远端 `main` 未变化；
8. 普通非强制 push。

如果远端已变化或 push 非 fast-forward，立即停止写入；禁止自动 force push、rebase 或 merge。

---

## 8. DeepSeek Harness 消费任务规则

DeepSeek Harness 只有在同时满足以下条件时才开始：

```text
owner = harness
state = HARNESS_READY 或 CHANGES_REQUIRED
task_id 位于 authorized_next
handoff_id 尚未消费
```

执行顺序：

1. `git fetch origin main`；
2. 确认本地工作区干净；
3. 只允许 `git merge --ff-only origin/main` 快进同步；
4. 从同步后的同一 commit 重新读取协议、State 和 Current Task；
5. 校验 `handoff_seq / handoff_id / task_id / iteration / owner / state`；
6. 只修改 Allowed Files；
7. 运行任务要求的测试和本地数据回测；
8. 写 `HARNESS_REPORT.md`；
9. push 前再次 `git fetch`，确认 `origin/main` 仍等于任务基线；
10. 设置新唯一 `handoff_id`、`handoff_seq + 1`、`state=REVIEW_READY`、`owner=architect`、`authorized_next=[]`；
11. 普通 commit + 普通 push 到 `main`；
12. push 成功后停止，不自动开始下一工作。

如果本地工作区有未知修改、远端变化、push 冲突或任务范围不清楚，停止写入，不 stash、不 reset、不 rebase、不 force push。

---

## 9. 本地历史数据规则

### 9.1 原始数据永不上传 GitHub

以下内容默认禁止 commit：

- `uniswap_data/`；
- `data/raw/`；
- `*.raw.csv`；
- `*.minute.csv`；
- 大型 Parquet / CSV / 数据库；
- 本地缓存；
- API key、钱包、私钥、助记词、RPC 私有凭据；
- `.local/` 配置目录。

### 9.2 Harness Report 必须记录数据证据

任何使用本地数据得出的结果，`HARNESS_REPORT.md` 至少包含：

```text
Dataset Name              数据集名称
Coverage Start / End      数据起止时间
File Count                文件数量
Row / Swap Count          行数或真实 Swap 数量
Approximate Size          数据量级
Input Pattern             输入文件匹配规则
Data Gaps                 缺失日期 / 缺块情况
Code Commit               运行使用的代码 commit
Command                    完整运行命令
Environment               Python 和关键包版本
Result Metrics            任务要求的全部结果指标
Artifacts                 允许提交的小型结果文件
Known Limitations         已知限制
```

若数据规模允许，可额外给出 manifest fingerprint（文件清单指纹），用于确认两次回测是否使用同一批本地文件；不要求对几十 GB 原始数据逐字节计算内容哈希。

### 9.3 不允许“估算冒充回测”

以下行为禁止：

- 用硬编码收益率替代真实回测；
- 用经验乘数修正后声称为逐笔验证；
- 因运行时间过长而跳过核心计算却报告 PASS；
- 把抽样结果写成全量结果；
- 把训练数据内结果写成严格样本外结果。

如任务无法在本地资源内完成，应报告 `BLOCKED` 或明确标注抽样范围。

---

## 10. HARNESS_REPORT 标准结构

每次交付固定包含：

```markdown
# Harness Report

## 1. Task Identity
- task_id:
- iteration:
- consumed_handoff_id:
- base_remote_head:
- result_commit:

## 2. Changed Files

## 3. Local Data Used
- 数据覆盖范围
- 文件数
- 行数 / Swap 数
- 缺口

## 4. Commands Executed

## 5. Test Results

## 6. Backtest / Validation Results

## 7. Failure / Edge Cases

## 8. Reproducibility Notes

## 9. Known Limitations

## 10. Requested Architect Decision
```

所有英文缩写首次出现时必须同时写出英文全称和中文解释。

---

## 11. Architect Review 规则

用户在 ChatGPT 中输入 `fetch` 或 `f` 时，含义固定为：

1. 刷新仓库 `main`；
2. 读取同一个 `remote_head` 的 State、Task、Harness Report 和代码 diff；
3. 若没有新 handoff，回复“无新交接”，不写空 commit；
4. 若发现新的 `REVIEW_READY`：
   - 独立检查实际 diff；
   - 检查代码是否真的执行了报告声称的算法；
   - 检查测试；
   - 核对数据范围和统计口径；
   - 检查是否存在未来数据泄漏；
   - 检查是否存在硬编码收益、代理收益或抽样冒充全量；
5. 裁决：
   - `PASS`
   - `CHANGES_REQUIRED`
   - `BLOCKED`
   - `USER_ACTION_REQUIRED`
6. 如需要 DeepSeek Harness 修正，保持同一 `task_id`、增加 `iteration` 并发布新 handoff；
7. 如 PASS 且下一任务明确，可以在同一次 Architect 提交中发布下一任务。

Architect 不得仅依据 Harness Report 的文字结论直接 PASS。

---

## 12. 冲突与停止写入规则

以下任一情况立即停止写入：

- 任务执行期间远端 `main` 改变；
- 非 fast-forward push；
- 本地存在未知未提交修改；
- `handoff_id` 已被消费；
- State 与 Task 不一致；
- 修改超出 Allowed Files；
- 本地数据路径不存在或数据覆盖与任务要求不符；
- 测试失败但无法在当前任务范围内修复；
- 结果无法复现。

停止后禁止自动：

- force push；
- rebase；
- merge commit；
- reset；
- stash 覆盖未知修改；
- cherry-pick；
- 盲目重复执行并覆盖证据。

应把证据写入 Harness Report，并设置 `BLOCKED` 或 `USER_ACTION_REQUIRED`。

---

## 13. Commit 纪律

- GitHub 只保存任务与验证检查点，不保存 heartbeat；
- 不产生空 commit；
- 一个任务尽量形成一个清晰实现 commit；
- commit message 推荐：

```text
research(R0-T001): implement <scope>
research(R0-T001): report local backtest
review(R0-T001): pass
review(R0-T001): request changes iteration 2
```

- 严禁提交私钥、钱包凭据、API key、本地绝对敏感路径和原始大数据。

---

## 14. 本地配置建议

DeepSeek Harness 本地建立以下文件，但 **绝不 commit**：

```text
.local/harness.yaml
```

建议内容：

```yaml
repository: "smhe00/uniswap-v3-ai-hunter"
branch: "main"
poll_interval_seconds: 180

local_data:
  uniswap_v3_data_root: "<你的本地历史数据目录>"

runtime:
  python_command: "python3"

safety:
  live_trading_allowed: false
  onchain_write_allowed: false
```

其中：

- `live_trading_allowed=false`：禁止真实交易；
- `onchain_write_allowed=false`：禁止任何会改变链上状态的调用；
- 本研究协议默认只覆盖研究、回测和验证，不授权修改或运行现有真实自动复投执行器。

本地建议执行：

```bash
mkdir -p .local
printf '.local/\n' >> .git/info/exclude
```

这样 `.local/` 仅在本 checkout 被忽略，不需要把个人数据路径写进公共 `.gitignore`。

---

## 15. DeepSeek Harness 启动提示词最小要求

DeepSeek Harness 新会话应首先读取：

```text
UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md
work/control/WORKFLOW_STATE.yaml
work/control/CURRENT_TASK.md
```

然后执行：

```text
1. fetch origin/main
2. 检查是否有新的、授权给 harness 的 handoff
3. 没有新授权则保持静默
4. 有授权则严格按 CURRENT_TASK 执行
5. 本地数据只用于计算，不上传
6. 完成后写 HARNESS_REPORT + REVIEW_READY
7. 普通 push 后停止并等待下一 handoff
```

不得依赖旧聊天摘要恢复任务状态。

---

## 16. 当前项目的默认研究安全边界

本协议当前默认：

```text
研究仓库                 = smhe00/uniswap-v3-ai-hunter
Uniswap 池版本           = Uniswap Version 3
真实交易                 = 禁止
链上写入                 = 禁止
本地历史数据读取         = 允许
本地回测 / 训练 / 验证   = 允许
GitHub 代码与报告提交    = 按任务授权允许
```

现有自动复投脚本属于独立执行层；除非未来任务明确扩展权限，否则 DeepSeek Harness 只研究，不触碰真实资金执行。

---

## 17. 最小协作原则

整个协议可以压缩为四句话：

1. **GitHub main 是唯一跨 Agent 权威状态。**
2. **DeepSeek Harness 只做当前明确授权任务，本地数据不上传。**
3. **Harness 提交代码 + 测试 + 数据证据报告，Architect 独立 Review。**
4. **任何冲突、数据不足或结果不可复现都停止写入，不猜、不强推、不伪造。**
