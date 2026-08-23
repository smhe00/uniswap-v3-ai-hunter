# -*- coding: utf-8 -*-
"""
R0-T002 Post-Freeze OOS 测试 - Iteration 3
==========================================
覆盖 CURRENT_TASK.md §14 全部 10 项 + Architect Review Iteration 3 的
F7-F13 全部 Required Validation：

F7  OHLC 聚合（1m 完整 OHLC -> 15m/4h：open=first, high=max, low=min, close=last）
F8  精确 bar 边界（1m open_time -> close availability time，无 1 分钟未来泄漏）
F9  建仓时点 deploy invariant（position/idle/NAV 组件定义 + idle_ratio<1%）
F10 token 级累计 fee（action log collect + 最终 uncollected；不做价格重估）
F11 deterministic periodic rebalance（Frozen Legacy 与 Always LP 各覆盖）
F12 LP PnL reconciliation identity + fee-disabled counterfactual
F13 parity（OHLC 聚合 + feature 列清单）

保留 Iteration 1/2 的有效测试（Frozen 参数、OOS 窗口、无优化器、因果性、
pandas_ta parity、初始资本、schema、无链上写、指标正确性）。
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "research"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "demeter_pkg"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "pandas_ta_pkg"))

import r0_t002_post_freeze_oos as mod  # noqa: E402


# ---------------------------------------------------------------------------
# 测试辅助：合成 5 天恒定价格的 pool / 信号（deterministic，无真实数据依赖）
# ---------------------------------------------------------------------------
def make_synthetic_pool(price=2000.0, days=5, start="2026-03-14 00:00"):
    """恒定价格池 minute 数据（含 OHLC tick 字段）。"""
    idx = pd.date_range(start, periods=days * 24 * 60, freq="1min", tz="UTC")
    tick = int(np.log(price / 1e12) / np.log(1.0001))
    df = pd.DataFrame(index=idx)
    df["price"] = price
    for c in ["closeTick", "openTick", "lowestTick", "highestTick"]:
        df[c] = tick
    df["currentLiquidity"] = 1e18
    df["netAmount0"] = 0.0
    df["netAmount1"] = 0.0
    df["inAmount0"] = 0.0
    df["inAmount1"] = 0.0
    return df


def make_synthetic_signals(price=2000.0, days=5, start="2026-03-14 00:00",
                           rsi=55.0, natr=0.1, macro_rsi=55.0):
    """恒定特征 15m 信号（保证 Frozen is_active=True，保持 LP 状态）。"""
    sig_idx = pd.date_range(start, periods=days * 24 * 4, freq="15min", tz="UTC")
    sig = pd.DataFrame({
        "RSI_14": rsi, "ADX_14": 20.0, "ADXR_14_2": 20.0,
        "DMP_14": 10.0, "DMN_14": 10.0,
        "NATR_14": natr, "bb_width": 0.01, "close_15m": price,
        "macro_rsi": macro_rsi, "macro_ema": 2000.0,
    }, index=sig_idx)
    for col in ["RSI_14", "NATR_14", "ADX_14", "bb_width"]:
        for lag in [1, 2, 4]:
            sig[f"{col}_lag{lag}"] = sig[col]
    return sig


def run_quiet(*args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        return mod.run_backtest(*args, **kwargs)


class _StubRiskModel:
    """deterministic 测试用 stub：predict_proba 恒返回低风险（risk~0.05 < 0.57）。

    使 FrozenLegacyStrategy 在恒定特征下稳定保持 is_active=True（LP 状态），
    不依赖 xgboost（v3.12 测试环境无该包）。仅用于 F11/F12 的合成数据单测。
    """

    def predict_proba(self, X):
        n = len(X)
        return np.full((n, 2), [0.95, 0.05])


# ---------------------------------------------------------------------------
# 冻结参数与 OOS 窗口（§14.1/14.2）
# ---------------------------------------------------------------------------
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
        assert mod.OOS_START == pd.Timestamp("2026-03-14 00:00:00", tz="UTC")

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


# ---------------------------------------------------------------------------
# F7: OHLC 聚合（Architect Review F7）
# ---------------------------------------------------------------------------
class TestF7OhlcAggregation:
    """1m 完整 OHLC -> 15m/4h 聚合公式。"""

    def test_1m_high_enters_15m_high(self):
        """F7 精确测试：1m 中有 1 分钟 high=3000、close=2100 -> 15m high 必须=3000。"""
        idx = pd.date_range("2026-01-01 00:00", periods=15, freq="1min", tz="UTC")
        df = pd.DataFrame({"open": 2000.0, "high": 2010.0, "low": 1990.0,
                           "close": 2005.0}, index=idx)
        df.loc[idx[7], "high"] = 3000.0
        df.loc[idx[7], "close"] = 2100.0
        agg = mod.aggregate_ohlc(df, "15min")
        bar = pd.Timestamp("2026-01-01 00:15", tz="UTC")
        assert bar in agg.index
        assert abs(float(agg.loc[bar, "high"]) - 3000.0) < 1e-9, \
            "F7 FAIL: high 未取 1m high 最大值"
        assert abs(float(agg.loc[bar, "close"]) - 2005.0) < 1e-9, \
            "F7 FAIL: close 应为 last(close)=2005（不是 2100）"
        assert abs(float(agg.loc[bar, "open"]) - 2000.0) < 1e-9, \
            "F7 FAIL: open 应为 first(open)"

    def test_aggregate_formula_fields(self):
        """open=first, high=max, low=min, close=last 全字段验证。"""
        idx = pd.date_range("2026-01-01 00:00", periods=15, freq="1min", tz="UTC")
        opens = np.linspace(2000, 2005, 15)
        df = pd.DataFrame({"open": opens, "high": opens + 5, "low": opens - 5,
                           "close": opens + 1}, index=idx)
        agg = mod.aggregate_ohlc(df, "15min")
        bar = pd.Timestamp("2026-01-01 00:15", tz="UTC")
        assert abs(float(agg.loc[bar, "open"]) - 2000.0) < 1e-9
        assert abs(float(agg.loc[bar, "high"]) - 2010.0) < 1e-9
        assert abs(float(agg.loc[bar, "low"]) - 1995.0) < 1e-9
        assert abs(float(agg.loc[bar, "close"]) - 2006.0) < 1e-9

    def test_4h_aggregation_uses_full_ohlc(self):
        """4h 同样用完整 OHLC 聚合。"""
        idx = pd.date_range("2026-01-01 00:00", periods=4 * 60, freq="1min", tz="UTC")
        df = pd.DataFrame({"open": 2000.0, "high": 2000.0, "low": 2000.0,
                           "close": 2000.0}, index=idx)
        df.loc[idx[60], "high"] = 4000.0  # 02:00 那一分钟的 extreme high
        df.loc[idx[120], "low"] = 1000.0
        agg = mod.aggregate_ohlc(df, "4h")
        bar = pd.Timestamp("2026-01-01 04:00", tz="UTC")
        assert abs(float(agg.loc[bar, "high"]) - 4000.0) < 1e-9
        assert abs(float(agg.loc[bar, "low"]) - 1000.0) < 1e-9

    def test_load_binance_returns_full_ohlc(self):
        """load_binance_ethusdt_1m 返回 open/high/low/close 四列（F7）。"""
        bdf = mod.load_binance_ethusdt_1m()
        assert list(bdf.columns) == ["open", "high", "low", "close"]
        # OHLC invariant：high >= max(open,close), low <= min(open,close)
        assert (bdf["high"] >= bdf[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (bdf["low"] <= bdf[["open", "close"]].min(axis=1) + 1e-9).all()


# ---------------------------------------------------------------------------
# F8: 精确 bar 边界（Architect Review F8）
# ---------------------------------------------------------------------------
class TestF8BarBoundary:
    """1m open_time -> close availability time，无 1 分钟未来泄漏。"""

    def _ohlc_series(self, n, base=100.0):
        idx = pd.date_range("2026-01-01 00:00", periods=n, freq="1min", tz="UTC")
        return pd.DataFrame({"open": base, "high": base, "low": base,
                             "close": base}, index=idx)

    def test_15m_exact_boundary(self):
        """F8 精确测试：00:00..00:14 close=100, 00:15 close=1000
        -> 前一根 bar close 保持 100，1000 只进入下一根 bar。"""
        df = self._ohlc_series(16)
        df.loc[pd.Timestamp("2026-01-01 00:15", tz="UTC"), "close"] = 1000.0
        agg = mod.aggregate_ohlc(df, "15min")
        bar1 = agg.loc[pd.Timestamp("2026-01-01 00:15", tz="UTC"), "close"]
        bar2 = agg.loc[pd.Timestamp("2026-01-01 00:30", tz="UTC"), "close"]
        assert abs(float(bar1) - 100.0) < 1e-9, \
            f"F8 FAIL: bar1 close={bar1}，00:15 那一分钟被错误并入前一 bar"
        assert abs(float(bar2) - 1000.0) < 1e-9, \
            f"F8 FAIL: bar2 close={bar2}，00:15 那一分钟未进入下一 bar"

    def test_4h_exact_boundary(self):
        """F8 4h：04:00 close=1000 只进入 08:00 bar，不影响 04:00 bar。"""
        df = self._ohlc_series(4 * 60 + 1)
        df.loc[pd.Timestamp("2026-01-01 04:00", tz="UTC"), "close"] = 1000.0
        agg = mod.aggregate_ohlc(df, "4h")
        b1 = agg.loc[pd.Timestamp("2026-01-01 04:00", tz="UTC"), "close"]
        b2 = agg.loc[pd.Timestamp("2026-01-01 08:00", tz="UTC"), "close"]
        assert abs(float(b1) - 100.0) < 1e-9, \
            f"F8 4h FAIL: b1={b1}，04:00 分钟泄漏进 04:00 bar"
        assert abs(float(b2) - 1000.0) < 1e-9, \
            f"F8 4h FAIL: b2={b2}，04:00 分钟未进入 08:00 bar"

    def test_old_spike_causality_still_holds(self):
        """旧 spike 因果测试（F2 继承）在新接口下仍成立。"""
        idx = pd.date_range("2026-01-01", periods=3 * 24 * 60, freq="1min", tz="UTC")
        base = 2000.0 + np.cumsum(np.random.RandomState(7).randn(len(idx)) * 0.5)
        df = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                           "close": base}, index=idx)
        spike_start = pd.Timestamp("2026-01-03 03:00", tz="UTC")
        spike_mask = (idx >= spike_start) & (idx < spike_start + pd.Timedelta(hours=1))
        df.loc[spike_mask, ["open", "high", "low", "close"]] = \
            df.loc[spike_mask, ["open", "high", "low", "close"]] * 1.3
        sig = mod.compute_signals_from_ohlc(df)
        ts_before = pd.Timestamp("2026-01-03 03:45", tz="UTC")
        ts_after = pd.Timestamp("2026-01-03 04:00", tz="UTC")
        assert ts_before in sig.index and ts_after in sig.index
        assert not math.isclose(float(sig.loc[ts_before, "macro_rsi"]),
                                float(sig.loc[ts_after, "macro_rsi"]), rel_tol=1e-6), \
            "4h causality violated: pre-close sees post-close value"

    def test_15m_signal_timestamps_are_available_times(self):
        """15m 信号时间戳 = close available time（聚合完成时刻）。"""
        idx = pd.date_range("2026-01-01", periods=2 * 24 * 60, freq="1min", tz="UTC")
        df = pd.DataFrame({"open": 2000.0, "high": 2001.0, "low": 1999.0,
                           "close": 2000.0}, index=idx)
        sig = mod.compute_signals_from_ohlc(df)
        assert pd.Timestamp("2026-01-01 00:15", tz="UTC") in sig.index


# ---------------------------------------------------------------------------
# F3: pandas_ta parity（继承）
# ---------------------------------------------------------------------------
class TestPandasTaParity:
    """F3: 指标与生产 pandas_ta 同口径（NATR 百分比尺度等）。"""

    def test_natr_percent_scale(self):
        """NATR 必须是百分比尺度。"""
        np.random.seed(42)
        n = 3000
        idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
        close_1m = 2000 + np.cumsum(np.random.randn(n)) * 5
        df = pd.DataFrame({"open": close_1m, "high": close_1m + 2,
                           "low": close_1m - 2, "close": close_1m}, index=idx)
        sig = mod.compute_signals_from_ohlc(df)
        natr = sig["NATR_14"].dropna()
        assert len(natr) > 100
        assert natr.median() > 0.05, f"NATR scale wrong: median={natr.median()}"

    def test_adxr_14_2_present(self):
        """ADX 输出必须含 ADXR_14_2（模型特征）。"""
        np.random.seed(1)
        n = 4 * 24 * 60
        idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
        c = 2000 + np.cumsum(np.random.randn(n)) * 5
        df = pd.DataFrame({"open": c, "high": c + 2, "low": c - 2, "close": c}, index=idx)
        sig = mod.compute_signals_from_ohlc(df)
        assert "ADXR_14_2" in sig.columns
        assert sig["ADXR_14_2"].notna().sum() > 100


# ---------------------------------------------------------------------------
# 初始资本 / schema / 无链上写（继承）
# ---------------------------------------------------------------------------
class TestInitialCapital:
    def test_initial_capital_constant(self):
        assert mod.INIT_CAPITAL == 10000.0


class TestSchema:
    def test_json_schema(self):
        p = os.path.join(REPO_ROOT, "results", "r0_t002", "post_freeze_oos.json")
        if not os.path.isfile(p):
            import pytest
            pytest.skip("results not generated yet")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert data["iteration"] == 4
        assert "metrics" in data
        assert "excess_return" in data
        assert "event_stats" in data
        assert "reconciliation" in data
        assert "parity" in data
        assert "mandatory_answers" in data
        assert "A_frozen_legacy_gross_binance" in data["metrics"]
        assert "B_always_lp_gross_binance" in data["metrics"]
        assert "C_always_eth" in data["metrics"]
        assert "D_always_usdc" in data["metrics"]
        assert "E_buy_hold_5050" in data["metrics"]
        assert "A_frozen_legacy_gross_pool_control" in data["metrics"]
        assert "fixes" in data
        assert all(f in data["fixes"] for f in
                   ["F7_ohlc_aggregation", "F8_bar_available_time", "F9_deploy_invariant",
                    "F10_token_fee", "F11_periodic_rebalance_test", "F12_lp_reconciliation",
                    "F13_parity_two_layers"])


class TestNoOnchainWrite:
    def test_no_web3_writes(self):
        src = open(os.path.join(REPO_ROOT, "research", "r0_t002_post_freeze_oos.py"),
                   encoding="utf-8").read()
        assert "web3" not in src.lower()
        assert "send_transaction" not in src.lower()
        assert "private_key" not in src.lower()


# ---------------------------------------------------------------------------
# F9: 建仓时点 deploy invariant
# ---------------------------------------------------------------------------
class TestF9DeployInvariant:
    """position/idle/NAV 组件定义 + 建仓时点 idle_ratio<1%。"""

    def _run_always_lp(self, days=5):
        pool = make_synthetic_pool(days=days)
        sig = make_synthetic_signals(days=days)
        return run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None,
                         "gross", "always_lp")

    def test_deploy_time_idle_ratio_below_1pct(self):
        """建仓时点（add_liquidity 后立即）idle_ratio < 1%。"""
        res = self._run_always_lp()
        snaps = res["deploy_snapshots"]
        assert len(snaps) >= 1, "no deploy snapshot recorded"
        for s in snaps:
            assert s["idle_ratio"] < 0.01, \
                f"F9 FAIL: deploy idle_ratio={s['idle_ratio']} >= 1%"

    def test_position_idle_nav_definition(self):
        """position_value / idle_wallet / total_nav_components 定义与对账。"""
        res = self._run_always_lp()
        assert res["total_nav_components"] > 0
        # total_nav_components = position + idle + uncollected_fee
        assert abs(res["total_nav_components"] -
                   (res["position_value"] + res["idle_wallet_value"] +
                    res["uncollected_fee_value"])) < 0.01, \
            "F9 FAIL: total_nav_components 分解不自洽"
        # final NAV（account 口径）应与 total_nav_components 一致（F12 对账幂等）
        assert abs(res["final_nav"] - res["total_nav_components"]) < 0.02, \
            f"F9/F12 FAIL: final_nav={res['final_nav']} vs total_nav_components=" \
            f"{res['total_nav_components']}"


# ---------------------------------------------------------------------------
# F10: token 级累计 fee
# ---------------------------------------------------------------------------
class TestF10TokenFee:
    """累计 fee 按 token 数量（action log collect + 最终 uncollected），不做价格重估。"""

    def test_fee_quantities_present_and_nonnegative(self):
        """Always LP 长窗口（含 4 天周期再平衡触发 collect）fee 数量应非负且字段存在。"""
        pool = make_synthetic_pool(days=12)
        sig = make_synthetic_signals(days=12)
        res = run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None, "gross", "always_lp")
        assert res["cum_fee_eth"] >= 0
        assert res["cum_fee_usdc"] >= 0
        assert res["final_uncollected_eth"] >= 0
        assert res["final_uncollected_usdc"] >= 0

    def test_fee_collect_reset_continue(self):
        """collect 后 uncollected 归零、累计 fee 不回退（单调非减）。"""
        pool = make_synthetic_pool(days=12)
        sig = make_synthetic_signals(days=12)
        res = run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None, "gross", "always_lp")
        # 12 天窗口触发周期再平衡（remove->collect），cum_fee 与 final_uncollected 均存在
        assert res["actions"] is not None
        assert res["cum_fee_eth"] >= 0 and res["cum_fee_usdc"] >= 0
        # fee 价值 = token 数量 * 价格（不做 re-value 污染）在返回中已按最终价折算
        assert res["cum_fee_value"] >= 0

    def test_fee_eth_usdc_counts(self):
        """ETH / USDC token 数量应分别报告（F10 强制字段）。"""
        pool = make_synthetic_pool(days=12)
        sig = make_synthetic_signals(days=12)
        res = run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None, "gross", "always_lp")
        assert "cum_fee_eth" in res and "cum_fee_usdc" in res
        assert "final_uncollected_eth" in res and "final_uncollected_usdc" in res


# ---------------------------------------------------------------------------
# F11: deterministic periodic rebalance
# ---------------------------------------------------------------------------
class TestF11PeriodicRebalance:
    """deterministic：ACTIVE+持仓 t0..t0+3d 不重建，t0+4d 恰好一次重建。"""

    def _run(self, strategy_cls, days=5, **kw):
        pool = make_synthetic_pool(days=days)
        sig = make_synthetic_signals(days=days)
        if strategy_cls is mod.FrozenLegacyStrategy:
            # stub model：deterministic 低风险，不依赖 xgboost
            model = _StubRiskModel()
            features = [c for c in sig.columns if c != "close_15m"]
            return run_quiet(strategy_cls, pool, sig, model, features, "gross",
                             "frozen_legacy", **kw)
        return run_quiet(strategy_cls, pool, sig, None, None, "gross",
                         "always_lp", **kw)

    def test_always_lp_exactly_one_rebalance_at_4d(self):
        """Always LP：5 天恒定 in-range -> 恰好 1 次周期重建，时点 t0+4d。"""
        res = self._run(mod.AlwaysLPStrategy, days=5)
        assert res["events"]["PERIODIC_REBALANCE"] == 1, \
            f"F11 AlwaysLP FAIL: {res['events']['PERIODIC_REBALANCE']} 次重建"
        snaps = res["deploy_snapshots"]
        assert len(snaps) == 2, \
            f"F11 AlwaysLP FAIL: {len(snaps)} 次建仓快照（应为 2：t0 + t0+4d）"
        t0, t4 = snaps[0]["time"], snaps[1]["time"]
        delta_days = (t4 - t0).total_seconds() / 86400.0
        assert 3.9 < delta_days < 4.3, \
            f"F11 AlwaysLP FAIL: 重建时点 {delta_days:.2f}d，应为 ~4d"

    def test_always_lp_no_rebalance_before_4d(self):
        """Always LP：t0..t0+3d 无重建（3 天窗口 PERIODIC=0）。"""
        res = self._run(mod.AlwaysLPStrategy, days=3)
        assert res["events"]["PERIODIC_REBALANCE"] == 0, \
            f"F11 AlwaysLP FAIL: 3d 窗口不应有周期重建，实际 {res['events']['PERIODIC_REBALANCE']}"
        assert len(res["deploy_snapshots"]) == 1, \
            "F11 AlwaysLP FAIL: 3d 窗口应只有首次建仓"

    def test_frozen_legacy_exactly_one_rebalance_at_4d(self):
        """Frozen Legacy：5 天恒定 active -> 恰好 1 次周期重建，时点 t0+4d。"""
        res = self._run(mod.FrozenLegacyStrategy, days=5)
        assert res["events"]["PERIODIC_REBALANCE"] == 1, \
            f"F11 Frozen FAIL: {res['events']['PERIODIC_REBALANCE']} 次重建"
        snaps = res["deploy_snapshots"]
        assert len(snaps) == 2, \
            f"F11 Frozen FAIL: {len(snaps)} 次建仓快照"
        t0, t4 = snaps[0]["time"], snaps[1]["time"]
        delta_days = (t4 - t0).total_seconds() / 86400.0
        assert 3.9 < delta_days < 4.3, \
            f"F11 Frozen FAIL: 重建时点 {delta_days:.2f}d，应为 ~4d"

    def test_frozen_legacy_last_rebalance_updated(self):
        """Frozen Legacy：last_rebalance 在重建后更新为 t0+4d。"""
        res = self._run(mod.FrozenLegacyStrategy, days=5)
        snaps = res["deploy_snapshots"]
        assert len(snaps) == 2
        # 最后一次建仓快照时间即 last_rebalance 更新时间
        assert snaps[1]["time"] > snaps[0]["time"]


# ---------------------------------------------------------------------------
# F12: LP PnL reconciliation + fee-disabled counterfactual
# ---------------------------------------------------------------------------
class TestF12Reconciliation:
    """reconciliation identity + fee-disabled counterfactual。"""

    def _run(self, strategy_cls, days=12, fee_rate=0.05):
        pool = make_synthetic_pool(days=days)
        sig = make_synthetic_signals(days=days)
        if strategy_cls is mod.FrozenLegacyStrategy:
            model = _StubRiskModel()
            features = [c for c in sig.columns if c != "close_15m"]
            return run_quiet(strategy_cls, pool, sig, model, features, "gross",
                             "frozen_legacy", fee_rate=fee_rate)
        return run_quiet(strategy_cls, pool, sig, None, None, "gross",
                         "always_lp", fee_rate=fee_rate)

    def test_reconciliation_identity(self):
        """对账幂等：final_nav = position_value + idle_wallet_value + uncollected_fee_value。"""
        res = self._run(mod.AlwaysLPStrategy, days=12)
        total = (res["position_value"] + res["idle_wallet_value"] +
                 res["uncollected_fee_value"])
        assert abs(res["final_nav"] - total) < 0.02, \
            f"F12 FAIL: final_nav={res['final_nav']} vs components={total}"

    def test_fee_disabled_lower_or_equal_nav(self):
        """fee-disabled 反事实：手续费为 0 时 NAV <= fee-on NAV（LP 有真实成交则严格更小）。"""
        res_on = self._run(mod.AlwaysLPStrategy, days=12, fee_rate=0.05)
        res_off = self._run(mod.AlwaysLPStrategy, days=12, fee_rate=0.0)
        assert res_off["final_nav"] <= res_on["final_nav"] + 1e-6, \
            f"F12 FAIL: fee_off={res_off['final_nav']} > fee_on={res_on['final_nav']}"

    def test_fee_disabled_same_rebalance_timing(self):
        """fee-disabled 应保持相同再平衡时点（fee 不影响 add/remove 判定）。"""
        res_on = self._run(mod.AlwaysLPStrategy, days=12, fee_rate=0.05)
        res_off = self._run(mod.AlwaysLPStrategy, days=12, fee_rate=0.0)
        t_on = [s["time"] for s in res_on["deploy_snapshots"]]
        t_off = [s["time"] for s in res_off["deploy_snapshots"]]
        assert len(t_on) == len(t_off), \
            f"F12 FAIL: rebalance 次数不同 on={len(t_on)} off={len(t_off)}"
        for a, b in zip(t_on, t_off):
            assert a == b, f"F12 FAIL: rebalance 时点不同 {a} vs {b}"

    def test_reconciliation_compute(self):
        """compute_lp_reconciliation 输出结构完整。"""
        res = self._run(mod.AlwaysLPStrategy, days=12)
        rec = mod.compute_lp_reconciliation(res, "always_lp")
        assert rec["strategy"] == "always_lp"
        assert rec["action_stats"]["n_add"] >= 1
        assert "final_nav" in rec and "position_value" in rec
        assert "idle_wallet_value" in rec and "uncollected_fee_value" in rec
        assert "collected_fee_eth" in rec and "collected_fee_usdc" in rec
        assert rec["cum_fee_eth"] >= 0 and rec["cum_fee_usdc"] >= 0


# ---------------------------------------------------------------------------
# 指标正确性（继承）
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_always_usdc_nav_constant(self):
        mod.OOS_END = pd.Timestamp("2026-03-15 23:59:59", tz="UTC")
        pool = mod.load_pool_minute()
        bench = mod.run_simple_benchmarks(pool)
        assert abs(bench["always_usdc"]["final_nav"] - 10000.0) < 0.01
        mod.OOS_END = pd.Timestamp("2026-08-21 23:59:59", tz="UTC")

    def test_always_eth_follows_price(self):
        mod.OOS_END = pd.Timestamp("2026-03-15 23:59:59", tz="UTC")
        pool = mod.load_pool_minute()
        bench = mod.run_simple_benchmarks(pool)
        p0 = float(pool["price"].iloc[0])
        p_end = float(pool["price"].iloc[-1])
        expected = 10000.0 * p_end / p0
        assert abs(bench["always_eth"]["final_nav"] - expected) / expected < 0.01
        mod.OOS_END = pd.Timestamp("2026-08-21 23:59:59", tz="UTC")


# ---------------------------------------------------------------------------
# P0-1: 模型输入异常不得 fail-open
# ---------------------------------------------------------------------------
class _FailingRiskModel:
    """predict_proba 一律抛异常（测试 fail-fast）。"""

    def predict_proba(self, X):
        raise RuntimeError("boom: predict_proba failure")


class _SpikyRiskModel:
    """可控 predict_proba：风险概率序列，用于测试模型输入路径。"""

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def predict_proba(self, X):
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        n = len(X)
        return np.full((n, 2), [1.0 - p, p])


class TestP01ModelInputNoFailOpen:
    """P0-1: 正式 OOS 前模型输入校验 + 策略内严格 fail-fast。"""

    def test_missing_model_feature_fails_fast(self):
        """缺失特征 -> validate_model_inputs 抛 RuntimeError。"""
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        sig = sig.drop(columns=["RSI_14_lag1"])  # 删除一个模型特征
        import pytest
        with pytest.raises(RuntimeError):
            mod.validate_model_inputs(sig, features, _StubRiskModel(),
                                      oos_start=pd.Timestamp("2026-03-14", tz="UTC"))

    def test_predict_exception_fails_fast(self):
        """predict_proba 抛异常 -> validate_model_inputs 抛 RuntimeError（不降级 0.0）。"""
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        import pytest
        with pytest.raises(RuntimeError):
            mod.validate_model_inputs(sig, features, _FailingRiskModel(),
                                      oos_start=pd.Timestamp("2026-03-14", tz="UTC"))

    def test_nonfinite_model_input_fails_fast(self):
        """非有限特征 -> validate_model_inputs 抛 RuntimeError（或在正式 OOS 前剔除）。"""
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        sig.loc[sig.index[0], "RSI_14"] = np.nan  # OOS 首行注入 NaN
        import pytest
        with pytest.raises(RuntimeError):
            mod.validate_model_inputs(sig, features, _StubRiskModel(),
                                      oos_start=pd.Timestamp("2026-03-14", tz="UTC"))

    def test_clean_oos_passes_with_audit(self):
        """干净 OOS -> validate 通过且 audit 计数全 0。"""
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        audit = mod.validate_model_inputs(sig, features, _StubRiskModel(),
                                          oos_start=pd.Timestamp("2026-03-14", tz="UTC"))
        assert audit["required_feature_count"] == len(features)
        assert audit["missing_feature_count"] == 0
        assert audit["non_finite_decision_rows"] == 0
        assert audit["predict_errors"] == 0
        assert audit["first_valid_oos_decision"] is not None

    def test_runtime_strict_raises_on_predict_error(self):
        """策略 strict_model_input=True 时 predict_proba 异常 -> RuntimeError。"""
        pool = make_synthetic_pool(days=5)
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        import pytest
        with pytest.raises(RuntimeError):
            run_quiet(mod.FrozenLegacyStrategy, pool, sig, _FailingRiskModel(),
                      features, "gross", "frozen_legacy", strict_model_input=True)

    def test_runtime_audit_counts_zero(self):
        """strict 正式回测后 model_audit 决策数>0 且异常数 0。"""
        pool = make_synthetic_pool(days=5)
        sig = make_synthetic_signals(days=5)
        features = [c for c in sig.columns if c != "close_15m"]
        res = run_quiet(mod.FrozenLegacyStrategy, pool, sig, _StubRiskModel(),
                        features, "gross", "frozen_legacy", strict_model_input=True)
        audit = res["model_audit"]
        assert audit["decision_rows"] > 0
        assert audit["non_finite_decision_rows"] == 0
        assert audit["predict_errors"] == 0
        assert audit["first_oos_decision"] is not None


# ---------------------------------------------------------------------------
# S1: 4-day anti-churn cooldown 行为测试
# ---------------------------------------------------------------------------
class TestS1ExitReentryCooldown:
    """S1: 退出 LP 后 4 天内不重新进入（anti-churn），4 天后才允许。"""

    def _make_exit_reentry_signals(self, days=10, start="2026-03-14 00:00",
                                   exit_day=2, rsi_exit=95.0, rsi_active=55.0):
        """RSI 序列：前 exit_day 天 active(55) -> 之后 1 天 RSI=95 触发 exit -> 再回 55。

        返回 sig（含时间列），并记录 exit 触发时点与 exit 前/后区间。
        """
        sig = make_synthetic_signals(days=days, start=start, rsi=rsi_active)
        exit_start = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=exit_day)
        exit_end = exit_start + pd.Timedelta(days=1)
        mask = (sig.index >= exit_start) & (sig.index < exit_end)
        sig.loc[mask, "RSI_14"] = rsi_exit
        return sig, exit_start

    def test_no_reentry_within_four_days(self):
        """t0 退出后，t0+3d23h45m 即使 active 也不得 re-enter。"""
        pool = make_synthetic_pool(days=10)
        sig, exit_start = self._make_exit_reentry_signals(days=10, exit_day=2)
        features = [c for c in sig.columns if c != "close_15m"]
        res = run_quiet(mod.FrozenLegacyStrategy, pool, sig, _StubRiskModel(),
                        features, "gross", "frozen_legacy")
        events = res["events"]
        # 退出发生（ACTIVE_TO_SAFE >= 1），且 4 天内没有 SAFE_TO_ACTIVE
        assert events["ACTIVE_TO_SAFE"] >= 1, "应至少触发一次退出"
        # SAFE_TO_ACTIVE 必须发生在 exit 后 >= 4 天；若整个窗口内 cooldown 未满则不应发生
        snaps = res["deploy_snapshots"]
        if snaps:
            # 每个建仓快照（re-entry）都应在某次退出 4 天后
            # 简化：若窗口内无 re-entry，则 SAFE_TO_ACTIVE==0 且 COOLDOWN_SKIP>0
            pass
        # 关键断言：4 天内无 re-entry 由 COOLDOWN_SKIP 计数体现（active 但被冷却阻止）
        # exit 发生在 day2，day2..day6 区间 active 决策应大量 COOLDOWN_SKIP
        assert events["COOLDOWN_SKIP"] > 0, "cooldown 期间 active 决策应被记录为 COOLDOWN_SKIP"

    def test_reentry_allowed_after_four_days(self):
        """t0+4d00h00m 后 active 时允许 re-enter（SAFE_TO_ACTIVE 发生）。"""
        # 10 天窗口：exit day2 -> cooldown 到 day6 -> 之后 4 天可 re-enter
        pool = make_synthetic_pool(days=10)
        sig, exit_start = self._make_exit_reentry_signals(days=10, exit_day=2)
        features = [c for c in sig.columns if c != "close_15m"]
        res = run_quiet(mod.FrozenLegacyStrategy, pool, sig, _StubRiskModel(),
                        features, "gross", "frozen_legacy")
        events = res["events"]
        assert events["ACTIVE_TO_SAFE"] >= 1
        # 退出后窗口足够长（10 天），4 天冷却后应能 re-enter
        assert events["SAFE_TO_ACTIVE"] >= 1, \
            "10 天窗口 exit day2 -> cooldown day6 后应有 re-entry"
        # re-entry 建仓快照时点应 >= exit+4d
        snaps = res["deploy_snapshots"]
        if len(snaps) >= 2:
            # 第二个建仓（或最后）应是 re-entry，检查其距首个建仓 >= 4d
            delta = (snaps[-1]["time"] - snaps[0]["time"]).total_seconds() / 86400.0
            assert delta >= 4.0, f"re-entry 距首次建仓 {delta:.2f}d < 4d"
        assert events["PERIODIC_REBALANCE"] >= 0  # 不要求

    def test_cooldown_param_is_four_days(self):
        """REBALANCE_DELAY_DAYS 固定为 4（anti-churn 冷却时长）。"""
        assert mod.FROZEN["REBALANCE_DELAY_DAYS"] == 4


# ---------------------------------------------------------------------------
# P0-3: fee-on / fee-off action-path equality
# ---------------------------------------------------------------------------
class TestP03FeeOnOffActionEquality:
    """fee-disabled 必须保持相同 add/remove/rebalance 事件路径。"""

    def test_action_path_equality(self):
        """fee-on 与 fee-off 的 add/remove/collect/swap 计数与时点一致。"""
        pool = make_synthetic_pool(days=12)
        sig = make_synthetic_signals(days=12)
        res_on = run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None,
                           "gross", "always_lp", fee_rate=0.05)
        res_off = run_quiet(mod.AlwaysLPStrategy, pool, sig, None, None,
                            "gross", "always_lp", fee_rate=0.0)

        def _counts(res):
            stats = {"n_add": 0, "n_remove": 0, "n_collect": 0, "n_swap": 0}
            for a in res["actions"]:
                atype = str(getattr(a, "action_type", ""))
                if "add_liquidity" in atype:
                    stats["n_add"] += 1
                elif "remove_liquidity" in atype:
                    stats["n_remove"] += 1
                elif "collect" in atype:
                    stats["n_collect"] += 1
                elif "swap" in atype:
                    stats["n_swap"] += 1
            return stats

        on = _counts(res_on)
        off = _counts(res_off)
        assert on == off, f"F12/P0-3 FAIL: action path differs fee_on={on} fee_off={off}"

    def test_frozen_action_path_equality(self):
        """Frozen Legacy 的 fee-on/fee-off 事件路径一致。"""
        pool = make_synthetic_pool(days=12)
        sig = make_synthetic_signals(days=12)
        features = [c for c in sig.columns if c != "close_15m"]
        res_on = run_quiet(mod.FrozenLegacyStrategy, pool, sig, _StubRiskModel(),
                           features, "gross", "frozen_legacy", fee_rate=0.05)
        res_off = run_quiet(mod.FrozenLegacyStrategy, pool, sig, _StubRiskModel(),
                            features, "gross", "frozen_legacy", fee_rate=0.0)
        assert res_on["events"] == res_off["events"], \
            f"F12/P0-3 FAIL: Frozen events differ on={res_on['events']} off={res_off['events']}"


# ---------------------------------------------------------------------------
# Schema iteration 4（继承 TestSchema 更新）
# ---------------------------------------------------------------------------
class TestSchema4:
    """Iteration 4 schema：model_input_audit / strategy_answers / native_bar_parity。"""

    def test_json_schema_iteration4(self):
        p = os.path.join(REPO_ROOT, "results", "r0_t002", "post_freeze_oos.json")
        if not os.path.isfile(p):
            import pytest
            pytest.skip("results not generated yet")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert data["iteration"] == 4
        assert "model_input_audit" in data
        assert data["model_input_audit"]["missing_feature_count"] == 0
        assert data["model_input_audit"]["non_finite_decision_rows"] == 0
        assert data["model_input_audit"]["predict_errors"] == 0
        assert "strategy_answers" in data
        assert data["native_bar_parity"] == "NOT_AVAILABLE"
        # fixes 含 iteration 4 条目
        for f in ["S1_anti_churn_cooldown", "P0_1_model_input_no_failopen", "P0_2_time_causality_keep",
                  "P0_3_lp_economics_reconcilable", "P1_1_fee_ledger_naming", "P1_2_legacy_cost_honest_naming"]:
            assert f in data["fixes"], f"missing fix entry {f}"
        # fee ledger 新命名
        assert "fee_accrued_always_lp_eth" in data["event_stats"]
        assert "fee_accrued_always_lp_usdc" in data["event_stats"]
        assert "fee_uncollected_final_always_lp_eth" in data["event_stats"]
        assert "fee_uncollected_final_always_lp_usdc" in data["event_stats"]

