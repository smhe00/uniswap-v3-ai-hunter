# 🛡️ Uniswap V3 AI Dual-Engine Smart Hunter (Final Master Version)

这是一个针对 **Arbitrum ETH/USDC 0.05%** 池子开发的工业级量化管理系统。本系统旨在利用 AI 技术解决 Uniswap V3 极高杠杆下的 LVR (Loss Versus Rebalancing) 风险，并在各种极端行情下通过“非对称避险”逻辑守护本金并获取 Alpha 收益。

## 1. 系统设计原理 (Design Philosophy)

### A. 双引擎协作架构 (Dual-Engine MTF)
系统不再依赖单一周期，而是采用 **4H 宏观趋势 + 15M 微观风险** 的多时间框架策略：
- **宏观引擎 (4H Regime)**：基于 4 小时线的 RSI (52/50) 和 EMA 趋势判定当前市场是处于上涨、下跌还是震荡。
- **微观引擎 (15M Tactical)**：基于集成学习模型 (XGBoost + GA) 实时预测未来 2 小时的 LVR 风险概率。

### B. 非对称避险 (Asymmetric Hedging)
当微观风险触发“撤退”信号时，系统根据宏观判定决定持仓资产，彻底消除 LP 的“方向性焦虑”：
- **强势看多**：风险爆发但大趋势向上 -> 撤退并 **100% 持有 ETH**（享受现货涨幅，避开 LP 卖出损耗）。
- **强势看空**：大趋势向下 -> 撤退并 **100% 持有 USDC**（锁定美金价值，躲避瀑布）。
- **方向不明**：**原样持有 (Keep Ratio)**（减少 Swap 摩擦成本，待机而动）。

### C. 物理现实约束 (Latency & Gap)
本系统在研发时即考虑了最严苛的物理限制：
- **15秒延迟对冲**：显式引入了 5s 数据采样 + 10s 上链确认带来的延迟惩罚。
- **原子级验证**：回测直接解析几十 GB 的链上原始 Swap Log，捕捉分钟线掩盖的瞬时插针。

## 2. 关键黄金参数 (Verified Golden Params)

经过 Optuna 自动调优及 365 天压力测试锁定的实战参数：
- **做市区间 (`RANGE_PCT`)**: **±8.13%** (约 25 倍资本效率)。
- **风险报警阈值 (`XGB_THRESHOLD`)**: **0.57** (过滤 90% 以上随机噪音)。
- **再平衡冷却期**: **4 天** (用于锁定手续费复利，除非触发 AI 紧急报警)。
- **更新频率**: **15 分钟**。

## 3. 仓库资产清单 (File Descriptions)

| 文件 | 描述 |
| :--- | :--- |
| **`lp_smart_agent.py`** | **生产环境执行脚本**。实时抓取 Binance API 并输出避险/入场决策建议。 |
| **`models_15m.pkl`** | **AI 权重文件**。保存了在 Arbitrum 全年数据上训练的最优 XGBoost 模型。 |
| **`v3_raw_reality_check.py`** | **原子级数据引擎**。用于解析 `.raw.csv` 原始 Log 进行高精度复盘。 |
| **`dual_engine_optimizer.py`** | **参数搜索器**。利用 Optuna 进行大规模超参数演化。 |
| **`v3_hunter_monte_carlo.py`** | **压力测试工具**。通过蒙特卡罗模拟验证随机入场时的策略鲁棒性。 |
| **`best_magic_params.pkl`** | **参数备份**。记录了当前系统的核心黄金配置。 |

## 4. 1年期战报 (2025.03 - 2026.02)

| 测试项目 | 基准 (Hold 5 ETH+10k) | **AI 猎手 (本系统)** |
| :--- | :--- | :--- |
| **最终净值** | $19,823 | **$29,270** 🚀 |
| **总 ROI** | -5.0% (本金损耗) | **+40.3%** |
| **相对 Alpha** | 0.0% | **+45.3%** |
| **蒙特卡罗胜率** | N/A | **91.7%** |

## 5. 终端输出解读 (CLI Output Interpretation)

运行 `lp_smart_agent.py` 时，你将看到如下格式的输出。以下是各项指标的详细含义及取值范围：

### A. 状态汇总 (State Summary)
- **Price**: 当前实时 ETH/USDT 价格（来自 Binance Spot）。
- **Macro RSI**: 4小时级别 RSI 指标。
    - `> 52`: 进入 **BullRegime**（牛市环境）。
    - `< 50`: 进入 **BearRegime**（熊市环境）。
- **Mode**: 系统的当前持仓模式。
    - `ACTIVE`: 正在池子里提供流动性（LPing）。
    - `SAFE`: 已撤出流动性，处于避险状态。
- **Action**: 系统本次运行执行的操作。
    - `HOLD`: 维持现状，不做任何操作。
    - `EXIT -> HOLD ETH/USDC`: AI 检测到风险，正在执行撤退。
    - `RE-ENTER POOL`: 风险解除，正在重新部署资金。
    - `PERIODIC REBALANCE`: 触发 4 天强制再平衡逻辑。
- **Range**: 固定为 **±8.13%**，即 25 倍左右的资金效率。

### B. AI 信号细节 (Signals)
- **GA (Genetic Algorithm)**: 遗传算法环境过滤器。
    - `True`: 微观指标（RSI, NATR）处于“高胜率”震荡区间。
    - `False`: 市场微观特征异常，建议观望。
- **XGB (XGBoost Prob)**: 机器学习预测的 LVR 风险概率 (0.0 ~ 1.0)。
    - `> 0.57`: **高风险**，触发紧急撤退。
    - `< 0.57`: **安全**，风险在可控范围内。
- **Bull/Bear Regime**: 宏观引擎的大势判定。
    - 决定了当 `XGB > 0.57` 时，你应该持有 ETH 还是 USDC。

## 6. 最终应用与运维方案

### A. 部署建议
- **环境**：Python 3.12+，安装依赖 `pip install pandas pandas-ta xgboost ccxt optuna demeter`。
- **定时任务**：建议使用 `crontab` 每 15 分钟执行一次。
- **监控**：运行后会自动生成 `agent_state.json`，可通过监控该文件观察净值变动。

### B. 风险提示与免责
1. **Gas 成本**：高频避险策略需支付较多 Gas，请确保账户 ETH 充裕。
2. **免责声明**：本软件仅供学习与研究。DeFi 交易风险极高，开发者不对任何资金损失负责。**请先进行手动带跑观察，切勿盲目投入大额真实资产。**

---
**Tag: v3_ultimate_hunter_v3_dual_engine**  
**Status: Production Ready & High Robustness**
