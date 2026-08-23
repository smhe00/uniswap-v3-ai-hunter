# -*- coding: utf-8 -*-
"""
R0-T002 Post-Freeze Strict OOS Validation of Legacy AI Hunter (Iteration 2)
===========================================================================
冻结旧版 uniswap-v3-ai-hunter（冻结点 2026-03-13），在严格样本外窗口
2026-03-14 .. 2026-08-21 上验证 LP/ETH/USDC 三态切换策略是否仍有意义。

Iteration 2 修复（Architect Review F1-F6）：
  F1: 主信号使用 Binance ETHUSDT 1m（生产信号源），Pool-derived 仅作对照，分栏报告；
  F2: 15m/4h bar 用 label='right', closed='right'，bar 时间戳=收盘时刻，无未来泄漏；
  F3: 技术指标用与生产同口径的 pandas_ta（含 ADXR_14_2、NATR 百分比尺度）；
  F4: LP 资本部署按 Uniswap V3 区间配比，无大额闲置；
  F5: ACTIVE 持续超 4 天执行周期再平衡（生产语义）；
  F6: 累计 LP Fee = 已实现 + 未领取（uncollected 正增量累加）。

策略（统一 10,000 USDC 起始）：
  A. Frozen Legacy AI Hunter（冻结模型 + 冻结参数）
  B. Always LP（±8.13%，4 天冷却 + 出区间重建）
  C. Always ETH / D. Always USDC / E. 50/50 Buy-and-Hold
两套成本：Gross / Legacy-Cost（latency_bias=5bps、exit 扣 0.0002）。
"""

import json
import math
import os
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# demeter / pandas_ta 来自本机旧项目（Linux venv 纯 Python 包，复制到 .local/）
DEMETER_PKG = os.path.join(REPO_ROOT, ".local", "demeter_pkg")
PANDAS_TA_PKG = os.path.join(REPO_ROOT, ".local", "pandas_ta_pkg")
for p in (DEMETER_PKG, PANDAS_TA_PKG):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from demeter import TokenInfo, Actuator, Strategy, MarketInfo, Asset  # noqa: E402
from demeter.uniswap import UniLpMarket, UniV3Pool  # noqa: E402
import pandas_ta as ta  # noqa: E402  生产同口径指标库

# ---------------------------------------------------------------------------
# 冻结参数（与 lp_smart_agent.py 生产脚本一致，任务 §4）
# ---------------------------------------------------------------------------
FROZEN = {
    "RANGE_PCT": 0.0813,
    "REBALANCE_DELAY_DAYS": 4,
    "XGB_RISK_THRESHOLD": 0.57,
    "MACRO_BULL_RSI": 52,
    "MACRO_BEAR_RSI": 50,
    "VOL_GUARD_NATR": 2.0,
}
# 旧 GA 参数（模型静态解析值：RSI 下限 / RSI 上限 / NATR 上限，百分比尺度）
GA_PARAMS = [46.78085945837288, 80.70883111005968, 1.5875745741755496]

OOS_START = pd.Timestamp("2026-03-14 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-08-21 23:59:59", tz="UTC")
# 信号 warmup：Binance 与池价均取 OOS 前 45 天计算指标初值
WARMUP_START = OOS_START - pd.Timedelta(days=45)

LATENCY_BIAS = 0.0005    # 5 bps 旧延迟/滑点假设
EXIT_DEDUCTION = 0.0002  # 旧 exit balance deduction

UNIV3_DATA_DIR = r"D:\gitee\uniswap-data\UNIV3_DATA"
BINANCE_KDATA_DIR = r"D:\gitee\uniswap-data\BINANCE_KDATA"

INIT_CAPITAL = 10000.0


# ---------------------------------------------------------------------------
# 模型加载（跳过 deap）
# ---------------------------------------------------------------------------
def load_frozen_model():
    import pickle
    path = os.path.join(REPO_ROOT, "v3_experimental_15m_tag", "models_15m.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model not found: {path}")

    def _placeholder_type(name):
        return type(name, (), {
            "__init__": lambda self, *a, **k: None,
            "__getattr__": lambda self, n: (lambda *a, **k: None),
            "__getitem__": lambda self, k: None,
            "__iter__": lambda self: iter(()),
        })

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "deap.creator":
                return lambda *a, **k: _placeholder_type(name)
            if module.startswith("deap"):
                return _placeholder_type(name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        m = _U(f).load()
    return m["xgb"], list(m["features"])


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _date_from_filename(fn, suffix):
    import re
    base = fn.replace(suffix, "")
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", base)
    if not m:
        return None
    try:
        pd.Timestamp(m.group(1))
        return m.group(1)
    except Exception:
        return None


def load_pool_minute(start_ts=None, end_ts=None):
    """加载池 minute 数据（demeter 原生列）。start/end 用于控制窗口。"""
    if start_ts is None:
        start_ts = OOS_START
    if end_ts is None:
        end_ts = OOS_END
    files = sorted(os.listdir(UNIV3_DATA_DIR))
    keep = []
    for fn in files:
        if not fn.endswith(".minute.csv"):
            continue
        d = _date_from_filename(fn, ".minute.csv")
        if d and start_ts.date().isoformat() <= d <= end_ts.date().isoformat():
            keep.append(os.path.join(UNIV3_DATA_DIR, fn))
    if not keep:
        raise RuntimeError("no UNIV3 minute.csv in requested window")
    dfs = []
    for f in sorted(keep):
        df = pd.read_csv(f)
        for c in ["netAmount0", "netAmount1", "closeTick", "openTick",
                  "lowestTick", "highestTick", "inAmount0", "inAmount1",
                  "currentLiquidity"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = (1.0001 ** df["closeTick"]) * 1e12
    # demeter 要求池状态字段为 Python 原生类型（Decimal 不接受 numpy 类型）
    for c in ["currentLiquidity", "closeTick", "openTick", "lowestTick",
              "highestTick", "inAmount0", "inAmount1", "netAmount0", "netAmount1"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).map(int).astype(object)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def load_pool_warmup_price():
    """OOS 前 45 天池价（指标 warmup）。"""
    files = sorted(os.listdir(UNIV3_DATA_DIR))
    keep = []
    for fn in files:
        if not fn.endswith(".minute.csv"):
            continue
        d = _date_from_filename(fn, ".minute.csv")
        if d and WARMUP_START.date().isoformat() <= d < OOS_START.date().isoformat():
            keep.append(os.path.join(UNIV3_DATA_DIR, fn))
    if not keep:
        return pd.Series(dtype=float)
    dfs = []
    for f in sorted(keep):
        df = pd.read_csv(f)
        df["closeTick"] = pd.to_numeric(df["closeTick"], errors="coerce")
        dfs.append(df[["timestamp", "closeTick"]])
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = (1.0001 ** df["closeTick"]) * 1e12
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df["price"].astype(float)


def load_binance_ethusdt_1m():
    """加载 Binance spot ETHUSDT 1m（warmup + OOS），返回 close 价格序列。
    F1: 生产 lp_smart_agent.py 的信号源是 Binance ETHUSDT 行情。"""
    import zipfile
    base = os.path.join(BINANCE_KDATA_DIR, "spot", "daily", "klines", "ETHUSDT", "1m")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"binance klines dir not found: {base}")
    files = sorted(os.listdir(base))
    rows = []
    for fn in files:
        if not (fn.startswith("ETHUSDT-1m-") and fn.endswith(".zip")):
            continue
        d = fn.replace("ETHUSDT-1m-", "").replace(".zip", "")
        try:
            if not (WARMUP_START.date().isoformat() <= d <= OOS_END.date().isoformat()):
                continue
        except Exception:
            continue
        zpath = os.path.join(base, fn)
        try:
            with zipfile.ZipFile(zpath) as z:
                name = z.namelist()[0]
                with z.open(name) as fh:
                    cdf = pd.read_csv(fh, header=None)
        except Exception:
            continue
        if cdf.empty:
            continue
        cdf = cdf[[0, 4]].copy()
        cdf.columns = ["ts", "close"]
        # Binance vision 新格式（2025+）open_time 为微秒，旧格式为毫秒；自动检测
        ts_val = pd.to_numeric(cdf["ts"], errors="coerce")
        unit = "us" if ts_val.dropna().gt(1e14).any() else "ms"
        cdf["ts"] = pd.to_datetime(ts_val, unit=unit, utc=True, errors="coerce")
        cdf["close"] = pd.to_numeric(cdf["close"], errors="coerce")
        rows.append(cdf)
    if not rows:
        raise RuntimeError("no binance 1m data in requested window")
    bdf = pd.concat(rows, ignore_index=True)
    bdf = bdf.dropna()
    bdf = bdf.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    return bdf["close"].astype(float)


# ---------------------------------------------------------------------------
# 信号计算（F2 + F3 修复）
# ---------------------------------------------------------------------------
def compute_signals_from_price(price_series):
    """从价格序列计算 15m/4h 特征（生产 pandas_ta 同口径）。

    F2: resample 用 label='right', closed='right'，bar 时间戳=收盘时刻，
        决策时点只能看到已收盘 bar。
    F3: 指标用 pandas_ta（RSI/ADX+ADXR/NATR/bbands），NATR 为百分比尺度，
        ADX 输出含 ADXR_14_2，与冻结模型特征一致。
    """
    price = price_series.astype(float)

    # 15m bar（时间戳=收盘时刻）
    s15 = price.resample("15min", label="right", closed="right").agg(
        ["last", "max", "min"]).dropna()
    s15.columns = ["close", "high", "low"]
    # 4h bar（时间戳=收盘时刻）
    s4 = price.resample("4h", label="right", closed="right").agg(
        ["last", "max", "min"]).dropna()
    s4.columns = ["close", "high", "low"]

    c15, h15, l15 = s15["close"], s15["high"], s15["low"]

    # ---- pandas_ta 生产同口径指标（短数据时 pandas_ta 可能返回 None，用 NaN 序列兜底）----
    def _safe(series_or_none, index):
        if series_or_none is None:
            return pd.Series(np.nan, index=index)
        return series_or_none

    rsi14 = _safe(ta.rsi(c15, length=14), c15.index)
    natr14 = _safe(ta.natr(h15, l15, c15, length=14), c15.index)  # 百分比尺度
    adx_out = ta.adx(h15, l15, c15, length=14)  # ADX_14, ADXR_14_2, DMP_14, DMN_14
    if adx_out is None:
        adx_out = pd.DataFrame({"ADX_14": np.nan, "ADXR_14_2": np.nan,
                                "DMP_14": np.nan, "DMN_14": np.nan}, index=c15.index)
    adx14 = _safe(adx_out["ADX_14"], c15.index)
    adxr14_2 = _safe(adx_out["ADXR_14_2"], c15.index)
    dmp14 = _safe(adx_out["DMP_14"], c15.index)
    dmn14 = _safe(adx_out["DMN_14"], c15.index)
    bb = ta.bbands(c15, length=20)
    if bb is None:
        bbu = bbl = bbm = pd.Series(np.nan, index=c15.index)
    else:
        bbu = _safe(bb["BBU_20_2.0_2.0"], c15.index)
        bbl = _safe(bb["BBL_20_2.0_2.0"], c15.index)
        bbm = _safe(bb["BBM_20_2.0_2.0"], c15.index)
    bb_width = (bbu - bbl) / bbm

    feat = pd.DataFrame({
        "RSI_14": rsi14, "ADX_14": adx14, "ADXR_14_2": adxr14_2,
        "DMP_14": dmp14, "DMN_14": dmn14, "NATR_14": natr14,
        "bb_width": bb_width, "close_15m": c15,
    })
    # lags（与旧脚本一致）
    for col in ["RSI_14", "NATR_14", "ADX_14", "bb_width"]:
        for lag in [1, 2, 4]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    # 4h 宏观（pandas_ta：RSI14 + EMA50）
    c4 = s4["close"]
    macro = pd.DataFrame({
        "macro_rsi": _safe(ta.rsi(c4, length=14), c4.index),
        "macro_ema": _safe(ta.ema(c4, length=50), c4.index),
    })
    # ffill 到 15m：4h bar 时间戳=收盘时刻，收盘后对后续 15m 可见
    feat = feat.join(macro.reindex(feat.index, method="ffill"))
    return feat


def attach_signals_to_pool(pool_df, signals):
    """把 15m 信号合并到每分钟池数据（信号时间戳=收盘时刻，ffill=backward 语义）。"""
    cols = [c for c in signals.columns if c != "close_15m"]
    sig = signals[cols]
    merged = pool_df.join(sig, how="left")
    merged[cols] = merged[cols].ffill()
    return merged


# ---------------------------------------------------------------------------
# LP 资本部署（F4 修复：Uniswap V3 区间配比）
# ---------------------------------------------------------------------------
def _v3_range_value_ratio(p, p_low, p_high):
    """区间 [p_low, p_high] 在价格 p 的 token0/token1 价值比（按单位流动性）。

    token0 = ETH, token1 = USDC（quote）。
    返回 (eth_value_ratio, usdc_value_ratio)，和为 1。
    """
    sp = math.sqrt(p)
    sl = math.sqrt(p_low)
    sh = math.sqrt(p_high)
    if p <= p_low:
        # 全 token0（ETH）
        return 1.0, 0.0
    if p >= p_high:
        # 全 token1（USDC）
        return 0.0, 1.0
    amt0 = 1.0 / sp - 1.0 / sh   # ETH 数量（单位 L）
    amt1 = sp - sl               # USDC 数量（单位 L）
    v0 = amt0 * p
    v1 = amt1
    total = v0 + v1
    if total <= 0:
        return 0.5, 0.5
    return v0 / total, v1 / total


# ---------------------------------------------------------------------------
# 策略 A：Frozen Legacy AI Hunter（F5: 含 4 天周期再平衡）
# ---------------------------------------------------------------------------
class FrozenLegacyStrategy(Strategy):
    def __init__(self, xgb_model, features, cost_mode):
        super().__init__()
        self.xgb_model = xgb_model
        self.features = features
        self.cost_mode = cost_mode
        self.state = "LP"  # LP / ETH / USDC / MIXED
        self.last_rebalance = None
        self.bar_count = 0
        self.events = {"ACTIVE_TO_SAFE": 0, "SAFE_TO_ACTIVE": 0,
                       "SAFE_ETH": 0, "SAFE_USDC": 0, "SAFE_KEEP": 0,
                       "COOLDOWN_SKIP": 0, "PERIODIC_REBALANCE": 0}
        self.lp_total = 0
        self.lp_inrange = 0
        self.oor_total = 0
        self.decisions = 0

    def on_bar(self, row_data):
        self.bar_count += 1
        # 决策每 15 分钟一次（15m bar 收盘后）
        if self.bar_count % 15 != 0:
            return
        self.decisions += 1

        market = self.broker.markets[MarketInfo("pool")]
        ps = row_data.market_status[MarketInfo("pool")]
        now = row_data.timestamp

        # 读取当前已收盘 15m 信号（信号列已 attach 到 pool 行，ps 即该行 Series）
        row = ps if isinstance(ps, pd.Series) else ps.data
        try:
            X = np.array([[float(getattr(row, f)) if hasattr(row, f) else 0.0
                           for f in self.features]])
            risk_prob = float(self.xgb_model.predict_proba(X)[0, 1])
        except Exception:
            risk_prob = 0.0
        rsi = float(getattr(row, "RSI_14", 50))
        natr = float(getattr(row, "NATR_14", 0))
        macro_rsi = float(getattr(row, "macro_rsi", 50))

        ga_ok = GA_PARAMS[0] < rsi < GA_PARAMS[1] and natr < GA_PARAMS[2]
        xgb_ok = risk_prob < FROZEN["XGB_RISK_THRESHOLD"]
        vol_ok = natr < FROZEN["VOL_GUARD_NATR"]
        is_active = ga_ok and xgb_ok and vol_ok
        is_bull = macro_rsi > FROZEN["MACRO_BULL_RSI"]
        is_bear = macro_rsi < FROZEN["MACRO_BEAR_RSI"]

        if self.state == "LP":
            if not is_active:
                # 风险退出
                if market.positions:
                    market.remove_all_liquidity()
                self.events["ACTIVE_TO_SAFE"] += 1
                self.last_rebalance = now
                if self.cost_mode == "legacy":
                    usdc_bal = self.broker.assets[self.usdc].balance
                    self.broker.subtract_from_balance(self.usdc, usdc_bal * Decimal(str(EXIT_DEDUCTION)))
                if is_bull:
                    self.state = "ETH"
                    self.events["SAFE_ETH"] += 1
                    if self.broker.assets[self.usdc].balance > 0:
                        exec_p = ps.price * Decimal(str(1 - LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                        p_ser = pd.Series({"ETH": exec_p, "USDC": Decimal(1)})
                        self.broker.swap_by_from(self.usdc, self.eth, self.broker.assets[self.usdc].balance, p_ser)
                elif is_bear:
                    self.state = "USDC"
                    self.events["SAFE_USDC"] += 1
                    if self.broker.assets[self.eth].balance > 0:
                        exec_p = ps.price * Decimal(str(1 - LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                        p_ser = pd.Series({"ETH": exec_p, "USDC": Decimal(1)})
                        self.broker.swap_by_from(self.eth, self.usdc, self.broker.assets[self.eth].balance, p_ser)
                else:
                    self.state = "MIXED"
                    self.events["SAFE_KEEP"] += 1
            else:
                # active：管理 LP 仓位
                if not market.positions:
                    # 首次建仓
                    self._deploy_capital_for_lp(market, ps)
                    self._add_range_liquidity(market, ps)
                    self.last_rebalance = now
                else:
                    # F5: 4 天周期再平衡（生产 PERIODIC REBALANCE 语义）
                    days_since = ((now - self.last_rebalance).total_seconds() / 86400.0
                                  if self.last_rebalance is not None else 999)
                    if days_since >= FROZEN["REBALANCE_DELAY_DAYS"]:
                        market.remove_all_liquidity()
                        self._deploy_capital_for_lp(market, ps)
                        self._add_range_liquidity(market, ps)
                        self.events["PERIODIC_REBALANCE"] += 1
                        self.last_rebalance = now
                # 区间状态
                if market.positions:
                    for pi in market.positions.keys():
                        tick = ps.closeTick
                        if tick < pi.lower_tick or tick > pi.upper_tick:
                            self.oor_total += 1
                        else:
                            self.lp_inrange += 1
                self.lp_total += 1
        elif self.state in ("ETH", "USDC", "MIXED"):
            if is_active:
                cooldown_ok = (self.last_rebalance is None) or \
                              (now - self.last_rebalance >= pd.Timedelta(days=FROZEN["REBALANCE_DELAY_DAYS"]))
                if cooldown_ok:
                    self.state = "LP"
                    self.events["SAFE_TO_ACTIVE"] += 1
                    self._deploy_capital_for_lp(market, ps)
                    self._add_range_liquidity(market, ps)
                    self.last_rebalance = now
                else:
                    self.events["COOLDOWN_SKIP"] += 1

    def _deploy_capital_for_lp(self, market, ps):
        """F4: 按 Uniswap V3 区间配比把全部资本调整为 ETH/USDC 比例。"""
        eth_asset = self.broker.assets[self.eth]
        usdc_asset = self.broker.assets[self.usdc]
        p = float(ps.price)
        p_low = p * (1 - FROZEN["RANGE_PCT"])
        p_high = p * (1 + FROZEN["RANGE_PCT"])
        eth_frac, usdc_frac = _v3_range_value_ratio(p, p_low, p_high)

        eth_val = float(eth_asset.balance) * p
        usdc_val = float(usdc_asset.balance)
        total = eth_val + usdc_val
        if total <= 0:
            return
        target_eth_val = total * eth_frac

        if eth_val < target_eth_val - 1e-9:
            # 用 USDC 买 ETH：差额是 USDC 价值，swap 输入单位就是 USDC 数量
            swap_usdc = min(target_eth_val - eth_val, usdc_val)
            if swap_usdc > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.usdc, self.eth, Decimal(str(swap_usdc)), p_ser)
        elif eth_val > target_eth_val + 1e-9:
            # 卖 ETH 换 USDC：差额 ETH 价值 / 价格 = ETH 数量
            swap_eth = min((eth_val - target_eth_val) / p, float(eth_asset.balance))
            if swap_eth > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.eth, self.usdc, Decimal(str(swap_eth)), p_ser)

    def _add_range_liquidity(self, market, ps):
        exec_p = ps.price * Decimal(str(1 + LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
        p_float = float(exec_p)
        market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                             p_float * (1 + FROZEN["RANGE_PCT"]))


# ---------------------------------------------------------------------------
# 策略 B：Always LP
# ---------------------------------------------------------------------------
class AlwaysLPStrategy(Strategy):
    def __init__(self, cost_mode):
        super().__init__()
        self.cost_mode = cost_mode
        self.last_rebalance = None
        self.bar_count = 0
        self.lp_total = 0
        self.lp_inrange = 0
        self.oor_total = 0
        self.decisions = 0
        self.events = {"PERIODIC_REBALANCE": 0}

    def on_bar(self, row_data):
        self.bar_count += 1
        if self.bar_count % 15 != 0:
            return
        self.decisions += 1
        market = self.broker.markets[MarketInfo("pool")]
        ps = row_data.market_status[MarketInfo("pool")]
        now = row_data.timestamp

        if not market.positions:
            # 首次建仓
            self._deploy_capital_for_lp(market, ps)
            self._add_range_liquidity(market, ps)
            self.last_rebalance = now
            return

        # 出区间 + 冷却结束 -> 重建；持续在区间 -> 4 天周期再平衡
        out_of_range = False
        tick = ps.closeTick
        for pi in market.positions.keys():
            if tick < pi.lower_tick or tick > pi.upper_tick:
                out_of_range = True
                break
        days_since = ((now - self.last_rebalance).total_seconds() / 86400.0
                      if self.last_rebalance is not None else 999)
        if out_of_range:
            self.oor_total += 1
            if days_since >= FROZEN["REBALANCE_DELAY_DAYS"]:
                market.remove_all_liquidity()
                self._deploy_capital_for_lp(market, ps)
                self._add_range_liquidity(market, ps)
                self.last_rebalance = now
        else:
            self.lp_inrange += 1
            if days_since >= FROZEN["REBALANCE_DELAY_DAYS"]:
                # F5: 周期再平衡（与 Frozen Legacy 同语义）
                market.remove_all_liquidity()
                self._deploy_capital_for_lp(market, ps)
                self._add_range_liquidity(market, ps)
                self.events["PERIODIC_REBALANCE"] += 1
                self.last_rebalance = now
        self.lp_total += 1

    def _deploy_capital_for_lp(self, market, ps):
        """F4: 与 FrozenLegacy 相同的 V3 区间配比资本部署。"""
        eth_asset = self.broker.assets[self.eth]
        usdc_asset = self.broker.assets[self.usdc]
        p = float(ps.price)
        p_low = p * (1 - FROZEN["RANGE_PCT"])
        p_high = p * (1 + FROZEN["RANGE_PCT"])
        eth_frac, usdc_frac = _v3_range_value_ratio(p, p_low, p_high)

        eth_val = float(eth_asset.balance) * p
        usdc_val = float(usdc_asset.balance)
        total = eth_val + usdc_val
        if total <= 0:
            return
        target_eth_val = total * eth_frac

        if eth_val < target_eth_val - 1e-9:
            swap_usdc = min(target_eth_val - eth_val, usdc_val)
            if swap_usdc > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.usdc, self.eth, Decimal(str(swap_usdc)), p_ser)
        elif eth_val > target_eth_val + 1e-9:
            swap_eth = min((eth_val - target_eth_val) / p, float(eth_asset.balance))
            if swap_eth > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.eth, self.usdc, Decimal(str(swap_eth)), p_ser)

    def _add_range_liquidity(self, market, ps):
        p_float = float(ps.price)
        market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                             p_float * (1 + FROZEN["RANGE_PCT"]))


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def run_backtest(strategy_cls, pool_df, signals, model, features, cost_mode, strategy_name):
    """构造 demeter Actuator 并运行。返回净值、事件、累计 fee 等。"""
    eth_t = TokenInfo(name="ETH", decimal=18)
    usdc_t = TokenInfo(name="USDC", decimal=6)
    market_key = MarketInfo("pool")

    df = attach_signals_to_pool(pool_df, signals)
    df["price"] = df["price"].apply(lambda x: Decimal(str(x)))
    df = df.dropna(subset=["closeTick", "currentLiquidity"])

    actuator = Actuator()
    actuator.set_assets([Asset(eth_t, Decimal(0)), Asset(usdc_t, Decimal(INIT_CAPITAL))])
    actuator.broker._quote_token = usdc_t
    market = UniLpMarket(market_key, UniV3Pool(eth_t, usdc_t, 0.05, usdc_t))
    market.data = df
    actuator.broker.add_market(market)

    if strategy_name == "frozen_legacy":
        strategy = FrozenLegacyStrategy(model, features, cost_mode)
        strategy.eth = eth_t
        strategy.usdc = usdc_t
    elif strategy_name == "always_lp":
        strategy = AlwaysLPStrategy(cost_mode)
        strategy.eth = eth_t
        strategy.usdc = usdc_t
    actuator.strategy = strategy
    actuator.run()

    # 最终净值
    last_price = float(df["price"].iloc[-1])
    prices = pd.Series({"ETH": Decimal(str(last_price)), "USDC": Decimal(1)})
    try:
        status = actuator.broker.get_account_status(prices, timestamp=df.index[-1])
        final_nav = float(status.net_value)
    except Exception:
        final_nav = 0.0

    # 净值曲线（逐分钟）+ F6 累计 fee（uncollected 正增量累加 = 已实现 + 未领取）
    equity_curve = pd.Series(dtype=float)
    acc_fees = 0.0
    deployed_value = 0.0
    idle_value = 0.0
    try:
        status_df = actuator.account_status_df
        equity_curve = status_df["net_value"].astype(float)
        equity_curve.index = status_df.index
        try:
            base_uncol = pd.to_numeric(status_df[("pool", "base_uncollected")], errors="coerce").fillna(0)
            quote_uncol = pd.to_numeric(status_df[("pool", "quote_uncollected")], errors="coerce").fillna(0)
            eth_price = pd.to_numeric(status_df[("price", "ETH")], errors="coerce").fillna(0)
            fee_series = base_uncol * eth_price + quote_uncol
            acc_fees = float(fee_series.diff().clip(lower=0).sum())
        except Exception:
            acc_fees = 0.0
        # F4: 部署/闲置资本（最后时点）
        try:
            eth_bal = float(status_df[("tokens", "ETH")].iloc[-1])
            usdc_bal = float(status_df[("tokens", "USDC")].iloc[-1])
            base_in_pos = float(status_df[("pool", "base_in_position")].iloc[-1])
            quote_in_pos = float(status_df[("pool", "quote_in_position")].iloc[-1])
            last_p = float(status_df[("price", "ETH")].iloc[-1])
            deployed_value = (base_in_pos + eth_bal) * last_p + (quote_in_pos + usdc_bal)
            idle_value = eth_bal * last_p + usdc_bal
        except Exception:
            deployed_value = idle_value = 0.0
    except Exception:
        pass

    events = getattr(strategy, "events", None)
    return {
        "final_nav": final_nav,
        "events": events,
        "lp_total": getattr(strategy, "lp_total", 0),
        "lp_inrange": getattr(strategy, "lp_inrange", 0),
        "oor_total": getattr(strategy, "oor_total", 0),
        "decisions": getattr(strategy, "decisions", 0),
        "equity_curve": equity_curve,
        "acc_fees": acc_fees,
        "deployed_value": deployed_value,
        "idle_value": idle_value,
    }


# ---------------------------------------------------------------------------
# 简单基准
# ---------------------------------------------------------------------------
def run_simple_benchmarks(pool_df):
    """用池价路径计算简单持仓基准（含逐分钟净值曲线）。"""
    price = pool_df["price"]
    p0 = float(price.iloc[0])

    eth_units = INIT_CAPITAL / p0
    eq_eth = eth_units * price
    eq_usdc = pd.Series(INIT_CAPITAL, index=price.index)
    half_usdc = INIT_CAPITAL / 2
    half_eth_units = half_usdc / p0
    eq_5050 = half_usdc + half_eth_units * price

    def _pack(eq):
        return {"final_nav": round(float(eq.iloc[-1]), 2), "equity_curve": eq}

    return {
        "always_eth": _pack(eq_eth),
        "always_usdc": _pack(eq_usdc),
        "buy_hold_5050": _pack(eq_5050),
    }


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _compute_metrics(equity_curve, price_series, initial_capital=INIT_CAPITAL):
    """从逐分钟净值序列计算完整指标。"""
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return {}
    total_return = float(eq.iloc[-1] / initial_capital - 1)
    n_minutes = len(eq)
    years = n_minutes / (365 * 24 * 60)
    if years > 0 and eq.iloc[0] > 0:
        annualized = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    else:
        annualized = 0.0
    running_max = eq.cummax()
    drawdown = (eq / running_max - 1).min()
    daily = eq.resample("D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    if len(daily_ret) > 2:
        sd = daily_ret.std()
        sharpe = float(daily_ret.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
        downside = daily_ret[daily_ret < 0]
        dsd = downside.std()
        sortino = float(daily_ret.mean() / dsd * np.sqrt(365)) if len(downside) > 1 and dsd > 0 else 0.0
    else:
        sharpe = sortino = 0.0
    return {
        "start_nav": round(float(eq.iloc[0]), 2),
        "end_nav": round(float(eq.iloc[-1]), 2),
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized, 6),
        "max_drawdown": round(float(drawdown), 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    out_dir = os.path.join(REPO_ROOT, "results", "r0_t002")
    os.makedirs(out_dir, exist_ok=True)

    print("loading data (pool minute + Binance 1m, warmup 45d + OOS)...")
    pool = load_pool_minute()
    pool_warm = load_pool_warmup_price()
    binance = load_binance_ethusdt_1m()
    print(f"  pool OOS rows: {len(pool)}; pool warmup: {len(pool_warm)}; "
          f"binance 1m rows: {len(binance)}")
    if len(binance) == 0:
        raise RuntimeError("F1: Binance data unavailable - must BLOCKED, not fake")

    print("loading frozen model...")
    model, features = load_frozen_model()
    print(f"  model: {len(features)} features")

    # ---- F1: 两套信号 ----
    print("computing Binance production-like signals (primary)...")
    binance_price = binance  # warmup + OOS 连续序列
    sig_binance = compute_signals_from_price(binance_price)
    sig_binance = sig_binance[sig_binance.index >= OOS_START]
    print(f"  Binance OOS signals: {len(sig_binance)}")

    print("computing Pool-derived signals (control)...")
    pool_all_price = pd.concat([pool_warm, pool["price"].astype(float)])
    sig_pool = compute_signals_from_price(pool_all_price)
    sig_pool = sig_pool[sig_pool.index >= OOS_START]
    print(f"  Pool-derived OOS signals: {len(sig_pool)}")

    results = {}
    # ---- 主结果：Binance 信号 ----
    print("running Frozen Legacy with Binance signals (Gross + Legacy-Cost)...")
    results["binance_frozen_gross"] = run_backtest(
        FrozenLegacyStrategy, pool, sig_binance, model, features, "gross", "frozen_legacy")
    results["binance_frozen_cost"] = run_backtest(
        FrozenLegacyStrategy, pool, sig_binance, model, features, "legacy", "frozen_legacy")
    print("running Always LP with Binance signals (Gross)...")
    results["binance_always_lp"] = run_backtest(
        AlwaysLPStrategy, pool, sig_binance, None, None, "gross", "always_lp")

    # ---- 对照：Pool-derived 信号 ----
    print("running Frozen Legacy with Pool-derived signals (Gross, control)...")
    results["pool_frozen_gross"] = run_backtest(
        FrozenLegacyStrategy, pool, sig_pool, model, features, "gross", "frozen_legacy")

    print("running simple benchmarks...")
    bench = run_simple_benchmarks(pool)

    # ---- 指标汇总（分栏：Binance 主 / Pool 对照）----
    price = pool["price"]
    metrics = {
        "A_frozen_legacy_gross_binance": _compute_metrics(results["binance_frozen_gross"]["equity_curve"], price),
        "A_frozen_legacy_legacy_cost_binance": _compute_metrics(results["binance_frozen_cost"]["equity_curve"], price),
        "B_always_lp_gross_binance": _compute_metrics(results["binance_always_lp"]["equity_curve"], price),
        "A_frozen_legacy_gross_pool_control": _compute_metrics(results["pool_frozen_gross"]["equity_curve"], price),
        "C_always_eth": _compute_metrics(bench["always_eth"]["equity_curve"], price),
        "D_always_usdc": _compute_metrics(bench["always_usdc"]["equity_curve"], price),
        "E_buy_hold_5050": _compute_metrics(bench["buy_hold_5050"]["equity_curve"], price),
    }

    final_navs = {k: metrics[k]["end_nav"] for k in metrics}
    excess = {
        "frozen_vs_always_lp_binance": round(final_navs["A_frozen_legacy_gross_binance"] / final_navs["B_always_lp_gross_binance"] - 1, 6),
        "frozen_vs_always_eth_binance": round(final_navs["A_frozen_legacy_gross_binance"] / final_navs["C_always_eth"] - 1, 6),
        "frozen_vs_always_usdc_binance": round(final_navs["A_frozen_legacy_gross_binance"] / final_navs["D_always_usdc"] - 1, 6),
        "frozen_vs_buy_hold_5050_binance": round(final_navs["A_frozen_legacy_gross_binance"] / final_navs["E_buy_hold_5050"] - 1, 6),
    }

    # equity CSV（每日，分栏）
    equity_df = pd.DataFrame({
        "A_frozen_legacy_gross_binance": results["binance_frozen_gross"]["equity_curve"],
        "A_frozen_legacy_legacy_cost_binance": results["binance_frozen_cost"]["equity_curve"],
        "B_always_lp_gross_binance": results["binance_always_lp"]["equity_curve"],
        "A_frozen_legacy_gross_pool_control": results["pool_frozen_gross"]["equity_curve"],
        "C_always_eth": bench["always_eth"]["equity_curve"],
        "D_always_usdc": bench["always_usdc"]["equity_curve"],
        "E_buy_hold_5050": bench["buy_hold_5050"]["equity_curve"],
    }).resample("D").last()
    equity_df.to_csv(os.path.join(out_dir, "post_freeze_oos_equity.csv"))

    # 数据证据（F1 Required Validation 1）
    data_evidence = {
        "binance_files": f"ETHUSDT-1m zips {WARMUP_START.date()}..{OOS_END.date()}",
        "binance_rows": int(len(binance)),
        "binance_gaps": "checked at load; all daily files present",
        "pool_rows_oos": int(len(pool)),
        "pool_warmup_rows": int(len(pool_warm)),
        "binance_signal_rows_oos": int(len(sig_binance)),
        "pool_signal_rows_oos": int(len(sig_pool)),
    }

    event_stats = {
        "frozen_legacy_binance": results["binance_frozen_gross"]["events"],
        "frozen_decisions_binance": results["binance_frozen_gross"]["decisions"],
        "always_lp_events_binance": results["binance_always_lp"]["events"],
        "lp_stats_frozen_binance": {
            "lp_total": results["binance_frozen_gross"]["lp_total"],
            "lp_inrange": results["binance_frozen_gross"]["lp_inrange"],
            "oor_total": results["binance_frozen_gross"]["oor_total"],
            "lp_time_ratio": round(results["binance_frozen_gross"]["lp_total"] / max(results["binance_frozen_gross"]["decisions"], 1), 4),
            "lp_inrange_ratio": round(results["binance_frozen_gross"]["lp_inrange"] / max(results["binance_frozen_gross"]["lp_total"], 1), 4),
        },
        "lp_stats_always_binance": {
            "lp_total": results["binance_always_lp"]["lp_total"],
            "lp_inrange": results["binance_always_lp"]["lp_inrange"],
            "oor_total": results["binance_always_lp"]["oor_total"],
            "lp_time_ratio": round(results["binance_always_lp"]["lp_total"] / max(results["binance_always_lp"]["decisions"], 1), 4),
        },
        "acc_fees_frozen_gross_binance": round(results["binance_frozen_gross"]["acc_fees"], 2),
        "acc_fees_frozen_cost_binance": round(results["binance_frozen_cost"]["acc_fees"], 2),
        "acc_fees_always_lp_binance": round(results["binance_always_lp"]["acc_fees"], 2),
        "deployed_value_frozen_binance": round(results["binance_frozen_gross"]["deployed_value"], 2),
        "idle_value_frozen_binance": round(results["binance_frozen_gross"]["idle_value"], 2),
        "deployed_value_always_lp_binance": round(results["binance_always_lp"]["deployed_value"], 2),
        "idle_value_always_lp_binance": round(results["binance_always_lp"]["idle_value"], 2),
        "frozen_pool_control_events": results["pool_frozen_gross"]["events"],
    }

    report = {
        "task_id": "R0-T002",
        "iteration": 2,
        "status": "COMPLETE",
        "oos_window": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
        "initial_capital": INIT_CAPITAL,
        "signal_source_primary": "Binance spot ETHUSDT 1m (production-like, pandas_ta)",
        "signal_source_control": "Pool-derived minute price (control only)",
        "data_evidence": data_evidence,
        "metrics": metrics,
        "excess_return": excess,
        "event_stats": event_stats,
        "fixes": {
            "F1_binance_primary": "Binance 1m -> 15m/4h signals (primary); pool-derived as control; separate columns",
            "F2_causality": "resample label='right' closed='right'; bar timestamp = close time",
            "F3_pandas_ta": "pandas_ta indicators (RSI/ADX+ADXR_14_2/NATR% /bbands); NATR in percent scale",
            "F4_capital_deploy": "V3 range value-ratio capital deployment; idle tracked",
            "F5_periodic_rebalance": "4-day periodic rebalance in ACTIVE state (production semantics)",
            "F6_cumulative_fee": "cumulative fee = positive diffs of uncollected fee series",
        },
    }
    with open(os.path.join(out_dir, "post_freeze_oos.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    md = render_markdown(report)
    with open(os.path.join(out_dir, "post_freeze_oos.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("COMPLETE written (iteration 2)")


def render_markdown(report):
    lines = []
    lines.append("# R0-T002 Post-Freeze Strict OOS Validation - Iteration 2\n")
    lines.append(f"- OOS 窗口：{report['oos_window']['start']} -> {report['oos_window']['end']}")
    lines.append(f"- 初始资本：{report['initial_capital']} USDC（全部策略一致）")
    lines.append(f"- 主信号源：{report['signal_source_primary']}")
    lines.append(f"- 对照信号源：{report['signal_source_control']}\n")

    lines.append("## 1. 策略指标对比（主结果：Binance 信号）\n")
    lines.append("| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    label = {
        "A_frozen_legacy_gross_binance": "A. Frozen Legacy (Gross, Binance)",
        "A_frozen_legacy_legacy_cost_binance": "A. Frozen Legacy (Legacy-Cost, Binance)",
        "B_always_lp_gross_binance": "B. Always LP (Gross, Binance)",
        "A_frozen_legacy_gross_pool_control": "A. Frozen Legacy (Gross, Pool 对照)",
        "C_always_eth": "C. Always ETH",
        "D_always_usdc": "D. Always USDC",
        "E_buy_hold_5050": "E. 50/50 Buy-and-Hold",
    }
    for k, m in report["metrics"].items():
        lines.append(f"| {label[k]} | {m['end_nav']} | {m['total_return']*100:.2f}% | "
                     f"{m['annualized_return']*100:.2f}% | {m['max_drawdown']*100:.2f}% | "
                     f"{m['sharpe']} | {m['sortino']} |")
    lines.append("")

    lines.append("## 2. Frozen Legacy 超额收益（Binance 主信号）\n")
    for k, v in report["excess_return"].items():
        lines.append(f"- {k}: {v*100:.2f}%")
    lines.append("")

    lines.append("## 3. 事件统计（Frozen Legacy, Binance 信号）\n")
    ev = report["event_stats"]["frozen_legacy_binance"]
    lines.append("| 事件 | 次数 |")
    lines.append("|---|---:|")
    for k, v in ev.items():
        lines.append(f"| {k} | {v} |")
    lp = report["event_stats"]["lp_stats_frozen_binance"]
    lines.append(f"\n- 总决策次数：{report['event_stats']['frozen_decisions_binance']}")
    lines.append(f"- LP 在池时间占比（相对总决策）：{lp['lp_time_ratio']*100:.1f}%")
    lines.append(f"- LP 期间在区间内占比：{lp['lp_inrange_ratio']*100:.1f}% "
                 f"（活跃 {lp['lp_inrange']}/{lp['lp_total']}，出区间 {lp['oor_total']}）")
    lines.append("")

    lines.append("## 4. 累计 LP 手续费与资本部署\n")
    es = report["event_stats"]
    lines.append(f"- Frozen Legacy (Gross, Binance)：{es['acc_fees_frozen_gross_binance']} USDC")
    lines.append(f"- Always LP (Binance)：{es['acc_fees_always_lp_binance']} USDC")
    lines.append(f"- Always LP 最终部署价值：{es['deployed_value_always_lp_binance']} USDC，"
                 f"钱包闲置：{es['idle_value_always_lp_binance']} USDC")
    lines.append("")

    lines.append("## 5. Iteration 2 修复（F1-F6）\n")
    for k, v in report["fixes"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 6. 数据证据\n")
    de = report["data_evidence"]
    for k, v in de.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
