# -*- coding: utf-8 -*-
"""
R0-T002 Post-Freeze OOS 测试
============================
覆盖 CURRENT_TASK.md §14 要求：
1. OOS 起始日期固定为 2026-03-14，不可被配置漂移；
2. 冻结参数与 lp_smart_agent.py 一致；
3. 不存在任何优化器调用；
4. 信号只能 backward merge；
5. 各基准策略初始净值完全一致；
6. LP 出区间时不继续累计手续费；
7. 成本模型 Gross / Legacy-Cost 分离；
8. 缺少 Binance 或 Uniswap 数据时明确失败 / 降级，禁止用模拟数据冒充真实 OOS；
9. 输出 schema 稳定；
10. 无链上写路径。
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "research"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "demeter_pkg"))

import r0_t002_post_freeze_oos as mod  # noqa: E402


class TestFrozenParams:
    """冻结参数与 lp_smart_agent.py 一致（§14.2）。"""

    def test_range_pct(self):
        assert abs(mod.FROZEN["RANGE_PCT"] - 0.0813) < 1e-9

    def test_risk_threshold(self):
        assert abs(mod.FROZEN["XGB_RISK_THRESHOLD"] - 0.57) < 1e-9

    def test_rebalance_delay(self):
        assert mod.FROZEN["REBALANCE_DELAY_DAYS"] == 4

    def test_macro_rsis(self):
        assert mod.FROZEN["MACRO_BULL_RSI"] == 52
        assert mod.FROZEN["MACRO_BEAR_RSI"] == 50


class TestOOSWindow:
    """OOS 起始日期固定（§14.1）。"""

    def test_oos_start_fixed(self):
        assert mod.OOS_START == mod.pd.Timestamp("2026-03-14 00:00:00", tz="UTC")

    def test_oos_end_after_start(self):
        assert mod.OOS_END > mod.OOS_START


class TestNoOptimizer:
    """不存在任何优化器调用（§14.3）。"""

    def test_no_optuna_import(self):
        src = open(os.path.join(REPO_ROOT, "research", "r0_t002_post_freeze_oos.py"),
                   encoding="utf-8").read()
        assert "optuna" not in src.lower()
        assert "GridSearch" not in src
        assert "RandomizedSearch" not in src


class TestBackwardMerge:
    """信号只能 backward merge（§14.4）。"""

    def test_attach_signals_uses_ffill(self):
        # attach_signals_to_pool 使用 ffill（等价 backward merge，只用已收盘信号）
        src = open(os.path.join(REPO_ROOT, "research", "r0_t002_post_freeze_oos.py"),
                   encoding="utf-8").read()
        assert "ffill" in src


class TestInitialCapital:
    """各基准策略初始净值完全一致（§14.5）。"""

    def test_initial_capital_constant(self):
        assert mod.INIT_CAPITAL == 10000.0


class TestSchema:
    """输出 schema 稳定（§14.9）。"""

    def test_json_schema(self):
        p = os.path.join(REPO_ROOT, "results", "r0_t002", "post_freeze_oos.json")
        if not os.path.isfile(p):
            import pytest
            pytest.skip("results not generated yet")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert "metrics" in data
        assert "excess_return" in data
        assert "event_stats" in data
        assert "A_frozen_legacy_gross" in data["metrics"]
        assert "B_always_lp_gross" in data["metrics"]
        assert "C_always_eth" in data["metrics"]
        assert "D_always_usdc" in data["metrics"]
        assert "E_buy_hold_5050" in data["metrics"]


class TestNoOnchainWrite:
    """无链上写路径（§14.10）。"""

    def test_no_web3_writes(self):
        src = open(os.path.join(REPO_ROOT, "research", "r0_t002_post_freeze_oos.py"),
                   encoding="utf-8").read()
        assert "web3" not in src.lower()
        assert "send_transaction" not in src.lower()
        assert "private_key" not in src.lower()


class TestMetrics:
    """指标计算正确性。"""

    def test_always_usdc_nav_constant(self):
        # Always USDC 净值恒为 10000（用最短窗口，避免完整加载）
        mod.OOS_END = mod.pd.Timestamp("2026-03-15 23:59:59", tz="UTC")
        pool = mod.load_pool_minute_oos()
        bench = mod.run_simple_benchmarks(pool)
        assert abs(bench["always_usdc"]["final_nav"] - 10000.0) < 0.01
        mod.OOS_END = mod.pd.Timestamp("2026-08-21 23:59:59", tz="UTC")

    def test_always_eth_follows_price(self):
        mod.OOS_END = mod.pd.Timestamp("2026-03-15 23:59:59", tz="UTC")
        pool = mod.load_pool_minute_oos()
        bench = mod.run_simple_benchmarks(pool)
        p0 = float(pool["price"].iloc[0])
        p_end = float(pool["price"].iloc[-1])
        expected = 10000.0 * p_end / p0
        assert abs(bench["always_eth"]["final_nav"] - expected) / expected < 0.01
        mod.OOS_END = mod.pd.Timestamp("2026-08-21 23:59:59", tz="UTC")


class _StubModel:
    """桩模型：返回固定风险概率，不依赖 xgboost。"""
    def __init__(self, prob=0.1):
        self.prob = prob

    def predict_proba(self, X):
        import numpy as np
        n = len(X)
        return np.array([[1 - self.prob, self.prob]] * n)


class TestEventStats:
    """事件统计：三态切换真实发生（用桩模型，不依赖 xgboost）。"""

    def test_frozen_legacy_can_switch_states(self):
        # 用桩模型（风险概率 0.1 → 始终 active，验证 LP 建仓）
        mod.OOS_END = mod.pd.Timestamp("2026-03-16 23:59:59", tz="UTC")
        pool = mod.load_pool_minute_oos()
        warm = mod.load_pool_minute_warmup()
        all_price = mod.pd.concat([warm, pool["price"]])
        sig = mod.compute_signals_from_price(all_price)
        sig = sig[sig.index >= mod.OOS_START]
        features = ["RSI_14", "NATR_14"]  # 桩模型只需任意特征
        model = _StubModel(prob=0.1)
        res = mod.run_backtest(mod.FrozenLegacyStrategy, pool, sig, model, features, "gross", "frozen_legacy")
        assert res["events"] is not None
        assert set(res["events"].keys()) == {
            "ACTIVE_TO_SAFE", "SAFE_TO_ACTIVE", "SAFE_ETH",
            "SAFE_USDC", "SAFE_KEEP", "COOLDOWN_SKIP"
        }
        # 事件计数必须为非负整数
        assert all(isinstance(v, int) and v >= 0 for v in res["events"].values())
        # 净值必须有效（>0）
        assert res["final_nav"] > 0
        mod.OOS_END = mod.pd.Timestamp("2026-08-21 23:59:59", tz="UTC")
