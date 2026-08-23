# -*- coding: utf-8 -*-
"""
R0-T003 — ETH Regime Attribution 测试
=====================================
覆盖：
1. regime 划分 sanity（固定窗口法：bull/bear/range 均出现、连续完整覆盖、无 gap）
2. regime 内策略收益计算正确性（手工构造小数据验证）
3. 表结构完整（三个 regime 各含全部策略 + 超额字段）
4. 收益复利合成正确（多段复利）
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "research"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "demeter_pkg"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "pandas_ta_pkg"))

import r0_t003_regime_attribution as mod  # noqa: E402


def _daily_prices(values, start="2026-03-14"):
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx)


class TestFixedWindowSegmentation:
    """固定窗口 regime 划分。"""

    def test_simple_bull_bear_range(self):
        """构造 3 段：涨、跌、平 -> 各产生一个 regime。"""
        # 30 天：前 10 天 +15%，中间 10 天 -15%，后 10 天平
        prices = ([1000 * 1.015 ** i for i in range(10)] +
                  [1150 * 0.985 ** i for i in range(10)] +
                  [980] * 10)
        eth = _daily_prices(prices)
        segs = mod.fixed_window_regime_segments(eth, window=10, direction_thresh=0.05)
        regimes = [s["regime"] for s in segs]
        assert "bull" in regimes and "bear" in regimes and "range" in regimes, \
            f"expected all 3 regimes, got {regimes}"

    def test_full_coverage_no_gap(self):
        """所有段必须连续完整覆盖整个窗口（无 gap、无重叠）。"""
        rng = np.random.RandomState(3)
        n = 50
        prices = 2000 * (1 + np.cumsum(rng.randn(n) * 0.01))
        eth = _daily_prices(prices)
        segs = mod.fixed_window_regime_segments(eth, window=14, direction_thresh=0.05)
        assert segs, "should produce segments"
        # 首段从窗口首日开始，末段到窗口末日
        assert segs[0]["start"] == eth.index[0]
        assert segs[-1]["end"] == eth.index[-1]
        # 段之间连续无 gap
        for i in range(1, len(segs)):
            assert segs[i]["start"] == segs[i - 1]["end"] + pd.Timedelta(days=1), \
                f"gap between segments {i-1} and {i}"

    def test_days_sum_to_window(self):
        """所有段天数之和 = 窗口总天数。"""
        rng = np.random.RandomState(5)
        n = 40
        prices = 2000 * (1 + np.cumsum(rng.randn(n) * 0.015))
        eth = _daily_prices(prices)
        segs = mod.fixed_window_regime_segments(eth, window=14, direction_thresh=0.05)
        assert sum(s["days"] for s in segs) == len(eth)

    def test_eth_ret_correct(self):
        """段 eth_ret = 段末/段初 - 1。"""
        prices = [1000, 1100, 1200, 1150, 1050, 1000]  # 6 天
        eth = _daily_prices(prices)
        segs = mod.fixed_window_regime_segments(eth, window=6, direction_thresh=0.05)
        assert len(segs) == 1
        assert abs(segs[0]["eth_ret"] - (1000 / 1000 - 1)) < 1e-9  # 尾 1000 头 1000


class TestRegimeStrategyStats:
    """regime 内策略表现计算。"""

    def _make_equity(self, n=30):
        idx = pd.date_range("2026-03-14", periods=n, freq="D", tz="UTC")
        cols = mod.STRATEGY_COLUMNS
        data = {c: 10000.0 * (1.001 ** np.arange(n)) for c in cols}
        df = pd.DataFrame(data, index=idx)
        return df

    def test_single_bear_segment_returns(self):
        """单 regime 段：策略收益 = 段内净值首尾比。"""
        eth = _daily_prices([1000, 900, 850, 800, 760], "2026-03-14")  # 明确下跌
        segs = mod.fixed_window_regime_segments(eth, window=5, direction_thresh=0.05)
        assert len(segs) == 1 and segs[0]["regime"] == "bear"
        equity = self._make_equity(5)
        stats = mod.regime_strategy_stats(equity, segs)
        bear = stats["bear"]
        for col in mod.STRATEGY_COLUMNS:
            nav0 = float(equity[col].iloc[0])
            nav1 = float(equity[col].iloc[-1])
            expected = nav1 / nav0 - 1
            assert abs(bear["strategies"][col]["ret"] - expected) < 1e-6, \
                f"{col}: got {bear['strategies'][col]['ret']}, expected {expected}"

    def test_multi_segment_compounding(self):
        """多段同 regime 收益按复利合成。"""
        # 两个 bull 段（各 5 天），中间隔一个 range 段
        eth = _daily_prices(
            [1000, 1050, 1100, 1150, 1200,   # bull +20%
             1200, 1210, 1220, 1215, 1210,   # range ~0%
             1200, 1260, 1300, 1340, 1400],  # bull +16.7%
            "2026-03-14")
        segs = mod.fixed_window_regime_segments(eth, window=5, direction_thresh=0.05)
        bull_segs = [s for s in segs if s["regime"] == "bull"]
        assert len(bull_segs) == 2, f"expected 2 bull segments, got {len(bull_segs)}"
        # 手工构造策略净值：始终 1%/天
        n = 15
        idx = pd.date_range("2026-03-14", periods=n, freq="D", tz="UTC")
        equity = pd.DataFrame({"C_always_eth": 10000 * (1.01 ** np.arange(n))}, index=idx)
        stats = mod.regime_strategy_stats(equity, segs)
        # bull 合成 = (1+seg1)(1+seg2)-1
        seg1_bull = bull_segs[0]
        seg2_bull = bull_segs[1]
        sub1 = equity.loc[(equity.index >= seg1_bull["start"]) & (equity.index <= seg1_bull["end"]), "C_always_eth"]
        sub2 = equity.loc[(equity.index >= seg2_bull["start"]) & (equity.index <= seg2_bull["end"]), "C_always_eth"]
        r1 = float(sub1.iloc[-1] / sub1.iloc[0] - 1)
        r2 = float(sub2.iloc[-1] / sub2.iloc[0] - 1)
        expected = (1 + r1) * (1 + r2) - 1
        got = stats["bull"]["strategies"]["C_always_eth"]["ret"]
        assert abs(got - expected) < 1e-6, f"compounding: got {got}, expected {expected}"

    def test_excess_vs_eth_usdc(self):
        """excess_vs_eth = ret - eth_ret，excess_vs_usdc = ret - usdc_ret。"""
        eth = _daily_prices([1000, 900, 850, 800, 760], "2026-03-14")
        segs = mod.fixed_window_regime_segments(eth, window=5, direction_thresh=0.05)
        equity = self._make_equity(5)
        stats = mod.regime_strategy_stats(equity, segs)
        bear = stats["bear"]
        eth_ret = bear["strategies"]["C_always_eth"]["ret"]
        usdc_ret = bear["strategies"]["D_always_usdc"]["ret"]
        for col in mod.STRATEGY_COLUMNS:
            st = bear["strategies"][col]
            assert abs(st["excess_vs_eth"] - (st["ret"] - eth_ret)) < 1e-6
            assert abs(st["excess_vs_usdc"] - (st["ret"] - usdc_ret)) < 1e-6

    def test_all_regimes_present_and_strategies(self):
        """结构完整性：三个 regime 各含全部策略 + 超额字段。"""
        rng = np.random.RandomState(7)
        n = 60
        prices = 2000 * (1 + np.cumsum(rng.randn(n) * 0.02))
        eth = _daily_prices(prices)
        segs = mod.fixed_window_regime_segments(eth, window=14, direction_thresh=0.05)
        equity = self._make_equity(n)
        stats = mod.regime_strategy_stats(equity, segs)
        for regime in ["bull", "bear", "range"]:
            rs = stats[regime]
            assert "days" in rs and "eth_ret" in rs and "strategies" in rs
            for col in mod.STRATEGY_COLUMNS:
                st = rs["strategies"][col]
                for field in ["ret", "mdd", "excess_vs_eth", "excess_vs_usdc", "segments"]:
                    assert field in st, f"{regime}/{col} missing {field}"


class TestSchema:
    """输出 JSON schema。"""

    def test_json_schema(self):
        p = os.path.join(REPO_ROOT, "results", "r0_t003", "regime_attribution.json")
        if not os.path.isfile(p):
            import pytest
            pytest.skip("results not generated yet")
        import json
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert data["task_id"] == "R0-T003"
        assert "method" in data
        assert "regime_stats" in data
        for regime in ["bull", "bear", "range"]:
            assert regime in data["regime_stats"], f"missing regime {regime}"
            rs = data["regime_stats"][regime]
            assert "strategies" in rs
            # 至少一个策略有收益
            assert any("ret" in st for st in rs["strategies"].values())


class TestResultsSane:
    """对已生成结果做 sanity 检查（若存在）。"""

    def test_bear_frozen_beats_eth(self):
        """下降阶段 Frozen 应显著优于 Always ETH（R0-T002 核心发现）。"""
        p = os.path.join(REPO_ROOT, "results", "r0_t003", "regime_attribution.json")
        if not os.path.isfile(p):
            import pytest
            pytest.skip("results not generated yet")
        import json
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        bear = data["regime_stats"]["bear"]["strategies"]
        frozen = bear["A_frozen_legacy_gross_binance"]["ret"]
        eth = bear["C_always_eth"]["ret"]
        assert frozen > eth, f"Frozen({frozen}) should beat ETH({eth}) in bear regime"
