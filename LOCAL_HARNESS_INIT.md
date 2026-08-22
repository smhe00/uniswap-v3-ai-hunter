# Local DeepSeek Harness 初始化指南

> 适用仓库：`smhe00/uniswap-v3-ai-hunter`  
> 权威分支：`main`  
> 配套协议：`UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md`  
> 适用角色：DeepSeek Harness 本地执行 Agent  
> 本文只负责**本地初始化**，不构成任何研究任务授权。

---

## 0. 先读这个结论

DeepSeek Harness 第一次拿到仓库后，只做四件事：

1. 校验 Git 仓库与远端 `main`；
2. 建立本地私有 `.local/` 配置；
3. 发现并盘点本地 Uniswap V3 历史数据；
4. 建立本地 `last_seen` 状态，准备等待 GitHub 正式任务。

初始化阶段**禁止**：

- 修改研究代码；
- 修改策略参数；
- 运行正式回测；
- 训练模型；
- 进行链上写操作；
- 下单、Swap、Add Liquidity、Remove Liquidity；
- 创建 Git commit；
- push GitHub；
- 上传本地原始数据。

初始化成功只表示“本机可以作为 Harness 节点工作”，不表示已经获得任何研究任务授权。

---

## 1. 名词说明

- **Uniswap V3 = Uniswap Version 3，Uniswap 第 3 版协议/资金池**。本文中的 `V3` 不是本项目的软件版本号。
- **LP = Liquidity Provider / Liquidity Position，流动性提供者 / 流动性仓位**。
- **Agent = 自动执行任务的软件代理**。本文特指 DeepSeek Harness。
- **Git HEAD = Git 当前指向的提交**。`remote_head` 表示本轮读取到的远端 `main` 提交标识。
- **YAML = YAML Ain't Markup Language，一种结构化配置文件格式**。
- **JSON = JavaScript Object Notation，一种结构化数据文件格式**。
- **RPC = Remote Procedure Call，远程过程调用接口**。本研究初始化不需要私有 RPC 凭据。
- **API = Application Programming Interface，应用程序编程接口**。任何 API 密钥均不得提交到 GitHub。

---

## 2. 第一次启动的强制读取顺序

DeepSeek Harness 第一次启动必须按顺序读取：

```text
1. LOCAL_HARNESS_INIT.md
2. UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md
3. README.md
4. work/control/WORKFLOW_STATE.yaml            如果存在
5. work/control/CURRENT_TASK.md                如果存在
6. work/handoff/HARNESS_REPORT.md              如果存在
7. work/handoff/ARCHITECT_REVIEW.md            如果存在
```

如果 `work/` 控制文件尚未创建：

```text
初始化仍可继续；
完成本地初始化后进入 WAITING_FOR_ARCHITECT；
不得自行创建研究任务或开始旧脚本回测。
```

---

## 3. Git 仓库初始化检查

仓库必须满足：

```text
remote origin = https://github.com/smhe00/uniswap-v3-ai-hunter.git
或等价的 SSH 地址

branch = main
working tree = clean
```

推荐执行：

```bash
git remote -v
git status --short
git fetch origin main
git rev-parse origin/main
git branch --show-current
```

规则：

- 允许 `git fetch`；
- 需要同步时只允许 `git merge --ff-only origin/main`；
- 禁止自动 `rebase`；
- 禁止自动 merge commit；
- 禁止 `reset --hard`；
- 禁止 `stash` 来掩盖未知修改；
- 禁止 force push；
- 如果工作区已有未知修改，初始化状态必须为 `BLOCKED_LOCAL_WORKTREE`，先由用户处理。

初始化时记录：

```text
local_head
remote_head
branch
origin_url
working_tree_clean
```

保存到本地 `.local/init_report.md`，不要提交。

---

## 4. 创建本地私有目录

在仓库根目录创建：

```text
.local/
├── harness.yaml
├── data_inventory.json
├── init_report.md
├── last_seen.json
├── cache/
└── results/
```

`.local/` 永远是本机私有目录。

优先使用 `.git/info/exclude` 保证本地忽略，不要求因此修改仓库 `.gitignore`：

```bash
mkdir -p .local/cache .local/results
printf "\n.local/\n.venv/\n" >> .git/info/exclude
```

如果相同规则已经存在，不重复追加。

初始化完成后必须验证：

```bash
git status --short
```

输出仍应为空。

---

## 5. `.local/harness.yaml` 标准模板

用户只需要把本地数据路径改成真实绝对路径。

```yaml
repository:
  full_name: "smhe00/uniswap-v3-ai-hunter"
  branch: "main"
  protocol_file: "UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md"
  poll_interval_seconds: 180

local_data:
  # 必须改成真实绝对路径；Harness 对此目录只读。
  root: "/ABSOLUTE/PATH/TO/UNISWAP_DATA"

  # 旧项目常见数据格式。
  minute_glob: "**/*.minute.csv"
  raw_glob: "**/*.raw.csv"

  # Arbitrum WETH/USDC 0.05% Uniswap V3 Pool。
  pool_address: "0xc6962004f452bE9203591991D15f6b388e09E8D0"
  chain_id: 42161

  # 原始链上时间原则上按 UTC 解释；具体任务如有不同要求，以 CURRENT_TASK.md 为准。
  source_timezone: "UTC"
  read_only: true

runtime:
  python_command: "python3"
  preferred_python: "3.12+"
  package_manager: "uv"

local_state:
  inventory_file: ".local/data_inventory.json"
  init_report_file: ".local/init_report.md"
  last_seen_file: ".local/last_seen.json"
  cache_dir: ".local/cache"
  result_dir: ".local/results"

safety:
  live_trading_allowed: false
  onchain_write_allowed: false
  wallet_access_allowed: false
  force_push_allowed: false
  upload_raw_data_allowed: false
```

如果本机不用 `uv`，可把 `package_manager` 改成真实工具；但初始化阶段不要为了满足该字段自动改系统 Python 或全局安装大量依赖。

**uv = 一个 Python 包与环境管理工具。**

---

## 6. 本地数据发现与盘点

### 6.1 只读原则

Harness 对 `local_data.root` 必须视为只读数据源。

禁止：

- 重命名原始数据；
- 修改原始 CSV；
- 删除文件；
- 在原始数据目录里写缓存；
- 为节省空间自动压缩或覆盖用户数据。

所有缓存和临时结果写入：

```text
.local/cache/
.local/results/
```

### 6.2 初始化只做 Metadata Inventory

**Metadata Inventory = 元数据盘点。**

初始化阶段不要扫描几十 GB 文件的全部内容，也不要启动完整回测。

只盘点：

```text
数据根目录是否存在
minute.csv 文件数量
raw.csv 文件数量
文件名可推断的最早日期
文件名可推断的最晚日期
文件总大小
前若干文件名样例
是否存在明显日期缺口（仅基于文件名）
```

若无法从文件名判断时间范围，在 `data_inventory.json` 中标记：

```json
{
  "coverage_from_filename": "unknown"
}
```

不要为了初始化而全量解析数据。

### 6.3 `.local/data_inventory.json` 最小结构

```json
{
  "generated_at": "<ISO-8601 timestamp>",
  "data_root": "<absolute path>",
  "pool_address": "0xc6962004f452bE9203591991D15f6b388e09E8D0",
  "chain_id": 42161,
  "minute_files": {
    "pattern": "**/*.minute.csv",
    "count": 0,
    "approx_bytes": 0,
    "earliest_date_from_name": null,
    "latest_date_from_name": null
  },
  "raw_files": {
    "pattern": "**/*.raw.csv",
    "count": 0,
    "approx_bytes": 0,
    "earliest_date_from_name": null,
    "latest_date_from_name": null
  },
  "obvious_filename_gaps": [],
  "notes": []
}
```

该文件只存本地，不提交。

---

## 7. Python 环境检查

初始化只检查，不擅自重建环境。

至少记录：

```bash
python3 --version
python3 -c "import sys; print(sys.executable)"
```

如果已经安装，可记录这些包版本；不存在时标记 `NOT_INSTALLED`，不要因此判定整个 Harness 初始化失败：

```text
pandas
numpy
demeter
xgboost
optuna
pandas_ta
ccxt
```

其中：

- **XGBoost = Extreme Gradient Boosting，极端梯度提升模型**；
- **CCXT = CryptoCurrency eXchange Trading Library，加密货币交易所统一接口库**；
- `Demeter` 是项目旧回测代码使用的 Uniswap V3 回测框架。

具体依赖安装和版本冻结必须由后续正式任务决定，不能在初始化阶段自行升级整个项目。

---

## 8. 安全检查

初始化必须确认本地配置满足：

```text
live_trading_allowed = false
onchain_write_allowed = false
wallet_access_allowed = false
force_push_allowed = false
upload_raw_data_allowed = false
```

如果任一值为 `true`：

```text
LOCAL HARNESS INIT = FAIL
原因 = unsafe local configuration
```

本研究仓库当前只承担研究、回测、数据验证和策略层工作。

现有链上自动复投 / Swap / Add Liquidity 执行程序不属于本初始化授权范围。

禁止在仓库或报告中写入：

```text
私钥
助记词
交易所 API key
私有 RPC token
钱包密码
账户口令
本地敏感凭据
```

---

## 9. 建立 `last_seen` 去重状态

初始化结束时创建：

```text
.local/last_seen.json
```

如果 `work/control/WORKFLOW_STATE.yaml` 已存在，则记录当前已观察到的 GitHub 状态，但**初始化本身不得把当前任务自动视为已执行**。

建议结构：

```json
{
  "remote_head": "<current origin/main>",
  "handoff_seq": null,
  "handoff_id": null,
  "task_id": null,
  "iteration": null,
  "state": null,
  "owner": null,
  "consumed": false
}
```

如果 State 文件存在，则把对应字段填入。

`consumed=false` 表示“已观察但没有执行”。

正式是否可以消费任务，仍必须重新按协议检查全部授权条件。

---

## 10. 初始化完成后的任务消费规则

初始化 PASS 后，DeepSeek Harness 可以进入静默检测模式。

每个轮询周期只做：

```bash
git fetch --quiet origin main
git rev-parse origin/main
```

然后从**同一个 remote_head** 读取：

```text
work/control/WORKFLOW_STATE.yaml
work/control/CURRENT_TASK.md
```

只有同时满足以下全部条件才允许开始任务：

```text
owner = harness
state = HARNESS_READY 或 CHANGES_REQUIRED
task_id 位于 authorized_next
handoff_id 尚未消费
本地工作区干净
本地安全配置全部为 false
```

开始任务前必须再次：

```text
fetch
确认 remote_head 未变化
ff-only 同步
重新读取协议 / State / Task
重新校验 handoff
```

若条件不满足：保持等待，不写文件、不跑回测、不提交 heartbeat。

---

## 11. 正式任务中如何使用本地数据

当 `CURRENT_TASK.md` 明确要求本地回测或数据验证时，Harness 才可以读取完整数据。

正式任务报告 `work/handoff/HARNESS_REPORT.md` 必须按协议记录：

```text
Dataset Name              数据集名称
Coverage Start / End      数据起止时间
File Count                文件数量
Row / Swap Count          行数或真实 Swap 数量
Approximate Size          数据量级
Input Pattern             输入文件匹配规则
Data Gaps                 缺失日期 / 缺块情况
Code Commit               运行所用代码版本
Command                    完整运行命令
Environment               Python 与关键包版本
Result Metrics            全部要求结果
Artifacts                 允许提交的小型结果
Known Limitations         已知限制
```

注意：

- **Swap = 代币兑换交易**；
- 原始数据不上传；
- 抽样结果必须明确写“抽样”；
- 不能把抽样写成全量；
- 不能把经验估算写成真实回测；
- 不能把训练数据结果写成严格样本外验证结果。

---

## 12. 本地初始化报告格式

`.local/init_report.md` 建议固定为：

```markdown
# Local Harness Init Report

## Repository
- origin:
- branch:
- local_head:
- remote_head:
- working_tree_clean:

## Protocol
- init_file_read: true/false
- collaboration_protocol_read: true/false

## Runtime
- python_version:
- python_executable:
- package_manager:

## Local Data
- root:
- root_exists:
- minute_file_count:
- raw_file_count:
- approximate_total_size:
- earliest_date_from_filename:
- latest_date_from_filename:
- obvious_gaps:

## Safety
- live_trading_allowed: false
- onchain_write_allowed: false
- wallet_access_allowed: false
- force_push_allowed: false
- upload_raw_data_allowed: false

## Control Plane
- workflow_state_exists:
- current_task_exists:
- observed_state:
- observed_handoff_id:

## Final Status
- LOCAL HARNESS INIT: PASS / WARN / FAIL
- NEXT MODE: WAITING_FOR_ARCHITECT / BLOCKED_LOCAL_WORKTREE / USER_ACTION_REQUIRED
- notes:
```

---

## 13. PASS / WARN / FAIL 判定

### PASS

同时满足：

```text
正确仓库
main 可 fetch
工作区干净
.local 已隔离
harness.yaml 可解析
数据路径存在或用户明确暂未配置
安全开关全部关闭
协议已读取
```

如果控制文件尚未创建：

```text
PASS
NEXT MODE = WAITING_FOR_ARCHITECT
```

### WARN

允许进入等待状态但需要在正式任务前处理，例如：

```text
某些 Python 包未安装
数据覆盖时间无法仅从文件名确定
minute 或 raw 数据只有一种
本地数据存在少量未知日期缺口
```

### FAIL

以下任何一种都必须 FAIL：

```text
仓库错误
origin 错误且无法确认
工作区存在未知修改
.local 没有成功隔离
harness.yaml 解析失败
安全开关存在 true
Harness 需要写原始数据目录才能工作
发现凭据即将被 Git 跟踪
```

---

## 14. DeepSeek Harness 第一次启动的 Canonical Prompt

**Canonical Prompt = 规范启动提示词。**

用户可以在 clone 完仓库后直接给 DeepSeek Harness：

```text
你是本仓库的 DeepSeek Harness 本地执行 Agent。

当前仓库：smhe00/uniswap-v3-ai-hunter
权威分支：main

首次启动时：
1. 完整读取 LOCAL_HARNESS_INIT.md；
2. 完整读取 UniswapV3_AI_Hunter_GitHub双Agent简化通信协议_Rev1.md；
3. 严格按 LOCAL_HARNESS_INIT.md 完成本地初始化；
4. 只创建 .local/ 下的本地配置、盘点和状态文件；
5. 初始化阶段不得修改研究代码，不得跑正式回测，不得 commit/push，不得执行任何链上写操作；
6. 输出本地初始化结果：PASS / WARN / FAIL；
7. 如果 PASS 或 WARN，进入 WAITING_FOR_ARCHITECT；
8. 之后只有 GitHub main 中出现 owner=harness、state=HARNESS_READY 或 CHANGES_REQUIRED、task_id 位于 authorized_next 且 handoff_id 未消费时，才可以执行任务；
9. 所有本地历史数据保持本机只读，不上传 GitHub；
10. 若协议、State、Task、远端 HEAD 或工作区状态存在冲突，停止写入并报告，不自行猜测或扩大权限。
```

---

## 15. 初始化成功时的终端最终输出

DeepSeek Harness 应简洁输出：

```text
LOCAL HARNESS INIT: PASS
Repository          : smhe00/uniswap-v3-ai-hunter
Branch              : main
Remote HEAD         : <sha>
Working Tree        : CLEAN
Local Config        : .local/harness.yaml
Local Data Root     : <configured path>
Minute Files        : <count>
Raw Files           : <count>
Safety              : RESEARCH-ONLY / NO ONCHAIN WRITE
Control State       : <state or NOT_INITIALIZED>
Next Mode           : WAITING_FOR_ARCHITECT
```

初始化到此结束。

**不要继续执行旧回测脚本。不要自行创建下一任务。等待 GitHub 正式 handoff。**
