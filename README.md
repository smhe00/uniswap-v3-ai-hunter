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

## 5. 最终应用与运维方案

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
