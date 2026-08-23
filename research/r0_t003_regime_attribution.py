# -*- coding: utf-8 -*-
"""
R0-T003 — ETH Regime Attribution of Frozen Legacy vs Benchmarks
================================================================
复用 R0-T002 已产出的逐日净值曲线，把 OOS 窗口按 ETH 行情划分为
上升(bull) / 下降(bear) / 震荡(range) regime，统计各策略在每个 regime
内的收益、最大回撤、相对 Always ETH / Always USDC 的超额。

方法（ex-post attribution，透明可复现）：
  - regime 用 ETH 日线收盘价的过去 N 日(默认 10)收益判定：
      ret_10d >= +2%  -> bull
      ret_10d <= -2%  -> bear
      其余            -> range
  - regime 为状态序列，连续同 regime 日合并为阶段。
  - 事后归因不用于任何策略决策，因此用未来确定阶段边界不构成 look-ahead
    （决策因果性已在 R0-T002 保证）。

只读数据：
  - results/r0_t002/post_freeze_oos_equity.csv（7 条每日净值曲线）
  - 池 minute 价格（load_pool_minute 的 price 列，ETH 行情）
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "research"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "demeter_pkg"))
sys.path.insert(0, os.path.join(REPO_ROOT, ".local", "pandas_ta_pkg"))

import r0_t002_post_freeze_oos as r2  # noqa: E402

OOS_START = r2.OOS_START
OOS_END = r2.OOS_END

# regime 判定参数（透明固定，非优化）
REGIME_WINDOW_DAYS = 14    # 固定子窗口长度（日）
REGIME_THRESHOLD = 0.05    # 段收益阈值：>= +5% bull / <= -5% bear / 其余 range

EQUITY_CSV = os.path.join(REPO_ROOT, "results", "r0_t002", "post_freeze_oos_equity.csv")

STRATEGY_COLUMNS = [
    "A_frozen_legacy_gross_binance",
    "A_frozen_legacy_legacy_cost_binance",
    "B_always_lp_gross_binance",
    "A_frozen_legacy_gross_pool_control",
    "C_always_eth",
    "D_always_usdc",
    "E_buy_hold_5050",
]

STRATEGY_LABELS = {
    "A_frozen_legacy_gross_binance": "Frozen Legacy (Gross, Binance)",
    "A_frozen_legacy_legacy_cost_binance": "Frozen Legacy (Legacy-Cost, Binance)",
    "B_always_lp_gross_binance": "Always LP (Gross, Binance)",
    "A_frozen_legacy_gross_pool_control": "Frozen Legacy (Gross, Pool 对照)",
    "C_always_eth": "Always ETH",
    "D_always_usdc": "Always USDC",
    "E_buy_hold_5050": "50/50 Buy-and-Hold",
}

REGIME_LABELS = {"bull": "上升 (bull)", "bear": "下降 (bear)", "range": "震荡 (range)"}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_equity_curves():
    """加载 R0-T002 逐日净值曲线（7 列，Daily index UTC）。"""
    df = pd.read_csv(EQUITY_CSV, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_eth_daily():
    """ETH 日线收盘价（池 minute price 日线 last）。"""
    pool = r2.load_pool_minute()
    return pool["price"].astype(float).resample("D").last().dropna()


# ---------------------------------------------------------------------------
# Regime 划分（固定窗口 + 方向阈值，连续完整覆盖）
# ---------------------------------------------------------------------------
def fixed_window_regime_segments(eth_daily, window=14, direction_thresh=0.05):
    """按固定子窗口 + 阶段收益方向切分 regime。

    把窗口切成固定长度子段（最后一段可短），每段按首尾收益方向标 regime：
      - 收益 >= +5%  -> bull
      - 收益 <= -5%  -> bear
      - 其余          -> range（震荡）
    相邻同 regime 段合并。

    透明、可复现、无参数优化；自然产生震荡(range)段，符合"下降/震荡阶段"
    分析需求。

    Returns:
        list[dict]: {start, end, regime, days, eth_ret}
    """
    if len(eth_daily) == 0:
        return []
    idx = list(eth_daily.index)
    n = len(idx)
    segments = []
    for i in range(0, n, window):
        end_i = min(i + window, n)
        sub = eth_daily.iloc[i:end_i]
        if len(sub) < 2:
            continue
        ret = float(sub.iloc[-1] / sub.iloc[0] - 1)
        if ret >= direction_thresh:
            regime = "bull"
        elif ret <= -direction_thresh:
            regime = "bear"
        else:
            regime = "range"
        segments.append({"start": idx[i], "end": idx[end_i - 1], "regime": regime,
                         "days": len(sub), "eth_ret": ret})

    # 合并相邻同 regime 段
    merged = []
    for s in segments:
        if merged and merged[-1]["regime"] == s["regime"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["days"] += s["days"]
            sub = eth_daily[(eth_daily.index >= merged[-1]["start"]) &
                            (eth_daily.index <= merged[-1]["end"])]
            if len(sub) >= 2:
                merged[-1]["eth_ret"] = float(sub.iloc[-1] / sub.iloc[0] - 1)
        else:
            merged.append(dict(s))
    return merged


# ---------------------------------------------------------------------------
# Regime 内策略表现
# ---------------------------------------------------------------------------
def _max_drawdown(eq):
    if len(eq) < 2:
        return 0.0
    running_max = eq.cummax()
    return float((eq / running_max - 1).min())


def regime_strategy_stats(equity, segments, benchmark="C_always_eth"):
    """对每个 regime 聚合阶段，统计各策略表现。

    Returns:
        dict: {
          'bull' / 'bear' / 'range': {
             days, eth_ret, segments: [...],
             strategies: {col: {ret, mdd, excess_vs_eth, excess_vs_usdc}}
          }
        }
    """
    out = {}
    for regime in ["bull", "bear", "range"]:
        segs = [s for s in segments if s["regime"] == regime]
        strat = {}
        for col in STRATEGY_COLUMNS:
            if col not in equity.columns:
                continue
            # 各段独立计算收益/回撤，再复利合成该 regime 总收益
            seg_rets = []
            seg_mdds = []
            total_factor = 1.0
            for s in segs:
                mask = (equity.index >= s["start"]) & (equity.index <= s["end"])
                sub = equity.loc[mask, col].dropna()
                if len(sub) >= 2:
                    r = float(sub.iloc[-1] / sub.iloc[0] - 1)
                    seg_rets.append(r)
                    seg_mdds.append(_max_drawdown(sub))
                    total_factor *= (1 + r)
            total_ret = total_factor - 1 if seg_rets else 0.0
            strat[col] = {
                "ret": round(total_ret, 6),
                "mdd": round(min(seg_mdds) if seg_mdds else 0.0, 6),
                "segments": len(segs),
                "segment_rets": [round(r, 6) for r in seg_rets],
            }
        # 超额
        for col in strat:
            eth = strat.get("C_always_eth", {}).get("ret", 0.0)
            usdc = strat.get("D_always_usdc", {}).get("ret", 0.0)
            strat[col]["excess_vs_eth"] = round(strat[col]["ret"] - eth, 6)
            strat[col]["excess_vs_usdc"] = round(strat[col]["ret"] - usdc, 6)
        out[regime] = {
            "days": sum(s["days"] for s in segs),
            "segments": segs,
            "eth_ret": float(np.prod([1 + s["eth_ret"] for s in segs]) - 1) if segs else 0.0,
            "strategies": strat,
        }
    return out


def summarize(equity, segments):
    """生成 JSON 报告 + 简单文本结论。"""
    report = {
        "task_id": "R0-T003",
        "status": "COMPLETE",
        "oos_window": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
        "method": {
            "segmentation": f"fixed {REGIME_WINDOW_DAYS}-day window, per-segment return "
                            f"direction (+/-{REGIME_THRESHOLD*100:.0f}%), merged same-regime",
            "note": "ex-post attribution; 固定窗口切分连续完整覆盖，按段首尾收益方向标 regime，"
                    "事后归因不用于策略决策",
        },
        "regime_stats": regime_strategy_stats(equity, segments),
    }
    return report


def render_markdown(report):
    lines = []
    lines.append("# R0-T003 — ETH Regime Attribution\n")
    lines.append(f"- OOS 窗口：{report['oos_window']['start']} -> {report['oos_window']['end']}")
    lines.append(f"- Regime 判定：固定 {REGIME_WINDOW_DAYS} 日窗口 + 段收益方向 "
                 f"(>= +{REGIME_THRESHOLD*100:.0f}% bull / <= -{REGIME_THRESHOLD*100:.0f}% bear / 其余 range)，"
                 f"相邻同 regime 合并")
    lines.append(f"- 方法：ex-post attribution（事后归因，不用于策略决策）\n")

    for regime in ["bull", "bear", "range"]:
        rs = report["regime_stats"][regime]
        lines.append(f"## {REGIME_LABELS[regime]} regime（{rs['days']} 天，ETH 阶段收益 {rs['eth_ret']*100:.2f}%）\n")
        if rs["segments"]:
            lines.append("| 阶段 | 起 | 止 | 天数 | ETH 收益 |")
            lines.append("|---|---|---:|---:|---:|")
            for s in rs["segments"]:
                lines.append(f"| {s['regime']} | {s['start'].date()} | {s['end'].date()} | "
                             f"{s['days']} | {s['eth_ret']*100:.2f}% |")
        lines.append("\n| 策略 | 阶段收益 | 最大回撤 | vs ETH | vs USDC |")
        lines.append("|---|---:|---:|---:|---:|")
        for col, st in rs["strategies"].items():
            lines.append(f"| {STRATEGY_LABELS[col]} | {st['ret']*100:.2f}% | "
                         f"{st['mdd']*100:.2f}% | {st['excess_vs_eth']*100:.2f}% | "
                         f"{st['excess_vs_usdc']*100:.2f}% |")
        lines.append("")

    # 结论
    lines.append("## 结论\n")
    for regime in ["bull", "bear", "range"]:
        rs = report["regime_stats"][regime]
        strat = rs["strategies"]
        if not strat:
            continue
        best = max(strat.items(), key=lambda kv: kv[1]["ret"])
        frozen = strat.get("A_frozen_legacy_gross_binance", {}).get("ret")
        eth = strat.get("C_always_eth", {}).get("ret")
        lp = strat.get("B_always_lp_gross_binance", {}).get("ret")
        lines.append(f"- **{REGIME_LABELS[regime]}**（{rs['days']} 天）：最优策略 **{STRATEGY_LABELS[best[0]]}** "
                     f"({best[1]['ret']*100:.2f}%)；Frozen {frozen*100:.2f}% vs Always ETH {eth*100:.2f}% "
                     f"vs Always LP {lp*100:.2f}%")
    lines.append("")
    return "\n".join(lines)


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "r0_t003")
    os.makedirs(out_dir, exist_ok=True)

    print("loading equity curves and ETH daily price...")
    equity = load_equity_curves()
    eth_daily = load_eth_daily()
    print(f"  equity rows: {len(equity)}; eth daily rows: {len(eth_daily)}")

    print("computing regime labels (ex-post, fixed-window)...")
    segments = fixed_window_regime_segments(eth_daily, window=REGIME_WINDOW_DAYS,
                                            direction_thresh=REGIME_THRESHOLD)
    print(f"  segments: {len(segments)}")
    for s in segments:
        print(f"    {s['regime']:<6} {s['start'].date()} -> {s['end'].date()}  {s['days']}d  ETH {s['eth_ret']*100:.1f}%")

    print("computing regime strategy stats...")
    report = summarize(equity, segments)

    with open(os.path.join(out_dir, "regime_attribution.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = render_markdown(report)
    with open(os.path.join(out_dir, "regime_attribution.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("COMPLETE written (R0-T003)")


if __name__ == "__main__":
    main()
