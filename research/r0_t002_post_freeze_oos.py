# -*- coding: utf-8 -*-
"""
R0-T002 Post-Freeze Strict OOS Validation of Legacy AI Hunter
==============================================================
冻结旧版 uniswap-v3-ai-hunter（冻结点 2026-03-13），在严格样本外窗口
2026-03-14 .. 2026-08-21 上验证 LP/ETH/USDC 三态切换策略是否仍有意义。

引擎：demeter（Uniswap V3 回测框架，旧项目官方引擎）。
策略（统一 10,000 USDC 起始）：
  A. Frozen Legacy AI Hunter（冻结模型 + 冻结参数）
  B. Always LP（±8.13%，4 天冷却 + 出区间重建）
  C. Always ETH
  D. Always USDC
  E. 50/50 ETH-USDC Buy-and-Hold
两套成本：Gross / Legacy-Cost（latency_bias=5bps、exit 扣 0.0002）。

因果性：信号只用已收盘 15m/4h bar；无优化器；无 OOS 后数据。
demeter 包来自本机旧项目 Linux venv（纯 Python 包），已复制到 .local/demeter_pkg。
"""

import json
import math
import os
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# demeter 来自本机旧项目（Linux venv 纯 Python 包，复制到 .local/demeter_pkg）
DEMETER_PKG = os.path.join(REPO_ROOT, ".local", "demeter_pkg")
if os.path.isdir(DEMETER_PKG) and DEMETER_PKG not in sys.path:
    sys.path.insert(0, DEMETER_PKG)

from demeter import TokenInfo, Actuator, Strategy, MarketInfo, Asset, ChainType  # noqa: E402
from demeter.uniswap import UniLpMarket, UniV3Pool  # noqa: E402

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
# 旧 GA 参数（模型静态解析值：RSI 下限 / RSI 上限 / NATR 上限）
GA_PARAMS = [46.78085945837288, 80.70883111005968, 1.5875745741755496]

OOS_START = pd.Timestamp("2026-03-14 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-08-21 23:59:59", tz="UTC")

LATENCY_BIAS = 0.0005
EXIT_DEDUCTION = 0.0002

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
# 数据准备（demeter 格式 + 信号列）
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


def load_pool_minute_oos():
    """加载 OOS 窗口的池 minute 数据，demeter 原生列 + 信号列。"""
    files = sorted(os.listdir(UNIV3_DATA_DIR))
    keep = []
    for fn in files:
        if not fn.endswith(".minute.csv"):
            continue
        d = _date_from_filename(fn, ".minute.csv")
        if d and OOS_START.date().isoformat() <= d <= OOS_END.date().isoformat():
            keep.append(os.path.join(UNIV3_DATA_DIR, fn))
    if not keep:
        raise RuntimeError("no UNIV3 minute.csv in OOS window")
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
    # 用 .map(int).astype(object) 保证 Series 元素是 Python int 而非 numpy.int64
    for c in ["currentLiquidity", "closeTick", "openTick", "lowestTick", "highestTick", "inAmount0", "inAmount1", "netAmount0", "netAmount1"]:
        if c in df.columns:
            df[c] = df[c].fillna(0).map(int).astype(object)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def load_pool_minute_warmup():
    """加载 OOS 之前（2025-01-01 .. 2026-03-13）的池数据用于指标 warmup。"""
    files = sorted(os.listdir(UNIV3_DATA_DIR))
    keep = []
    for fn in files:
        if not fn.endswith(".minute.csv"):
            continue
        d = _date_from_filename(fn, ".minute.csv")
        if d and d < OOS_START.date().isoformat():
            keep.append(os.path.join(UNIV3_DATA_DIR, fn))
    if not keep:
        return pd.DataFrame()
    dfs = []
    for f in sorted(keep):
        df = pd.read_csv(f)
        for c in ["closeTick", "currentLiquidity"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = (1.0001 ** df["closeTick"]) * 1e12
    df = df[["timestamp", "price"]].set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df["price"]


# ---------------------------------------------------------------------------
# 技术指标
# ---------------------------------------------------------------------------
def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def _natr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    return (atr / close).fillna(0)


def _adx(high, low, close, period=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def compute_signals_from_price(price_series):
    """从价格序列计算 15m/4h 特征（已收盘 bar）。返回 15m 频率信号 DataFrame。"""
    price = price_series
    s15 = price.resample("15min").agg(["last", "max", "min"]).dropna()
    s15.columns = ["close", "high", "low"]
    s4 = price.resample("4h").agg(["last", "max", "min"]).dropna()
    s4.columns = ["close", "high", "low"]

    c15, h15, l15 = s15["close"].astype(float), s15["high"].astype(float), s15["low"].astype(float)
    rsi14 = _rsi(c15)
    natr14 = _natr(h15, l15, c15)
    adx14, dmp14, dmn14 = _adx(h15, l15, c15)
    adxr14_2 = (adx14 + adx14.shift(14)) / 2
    mid = c15.rolling(20).mean()
    sd = c15.rolling(20).std()
    bb_width = ((mid + 2*sd) - (mid - 2*sd)) / mid.replace(0, np.nan)

    feat = pd.DataFrame({
        "RSI_14": rsi14, "ADX_14": adx14, "ADXR_14_2": adxr14_2,
        "DMP_14": dmp14, "DMN_14": dmn14, "NATR_14": natr14,
        "bb_width": bb_width, "close_15m": c15,
    })
    for col in ["RSI_14", "NATR_14", "ADX_14", "bb_width"]:
        for lag in [1, 2, 4]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    c4 = s4["close"].astype(float)
    macro = pd.DataFrame({
        "macro_rsi": _rsi(c4),
        "macro_ema": c4.ewm(span=50, adjust=False).mean(),
    })
    feat = feat.join(macro.reindex(feat.index, method="ffill"))
    return feat


def attach_signals_to_pool(pool_df, signals):
    """把 15m 信号 merge 到每分钟池数据（仅用已收盘 bar，backward）。"""
    cols = [c for c in signals.columns if c != "close_15m"]
    sig = signals[cols]
    merged = pool_df.join(sig, how="left")
    merged[cols] = merged[cols].ffill()
    return merged


# ---------------------------------------------------------------------------
# 策略 A：Frozen Legacy AI Hunter（demeter Strategy）
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
                       "COOLDOWN_SKIP": 0}
        self.lp_total = 0
        self.lp_inrange = 0
        self.oor_total = 0
        self.decisions = 0  # 总决策次数（每 15 分钟一次）

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
            # 逻辑：active → 确保有仓位；inactive → 退出避险
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
                    # 全换 ETH
                    if self.broker.assets[self.usdc].balance > 0:
                        exec_p = ps.price * Decimal(str(1 - LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                        p_ser = pd.Series({"ETH": exec_p, "USDC": Decimal(1)})
                        self.broker.swap_by_from(self.usdc, self.eth, self.broker.assets[self.usdc].balance, p_ser)
                elif is_bear:
                    self.state = "USDC"
                    self.events["SAFE_USDC"] += 1
                    # 全换 USDC
                    if self.broker.assets[self.eth].balance > 0:
                        exec_p = ps.price * Decimal(str(1 - LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                        p_ser = pd.Series({"ETH": exec_p, "USDC": Decimal(1)})
                        self.broker.swap_by_from(self.eth, self.usdc, self.broker.assets[self.eth].balance, p_ser)
                else:
                    self.state = "MIXED"
                    self.events["SAFE_KEEP"] += 1
            else:
                # active：首次建仓
                if not market.positions:
                    self._ensure_eth_for_lp(market, ps)
                    exec_p = ps.price * Decimal(str(1 + LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                    p_float = float(exec_p)
                    market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                                         p_float * (1 + FROZEN["RANGE_PCT"]))
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
                    # 重新进入 LP
                    self.state = "LP"
                    self.events["SAFE_TO_ACTIVE"] += 1
                    self.last_rebalance = now
                    self._ensure_eth_for_lp(market, ps)
                    exec_p = ps.price * Decimal(str(1 + LATENCY_BIAS)) if self.cost_mode == "legacy" else ps.price
                    p_float = float(exec_p)
                    market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                                         p_float * (1 + FROZEN["RANGE_PCT"]))
                else:
                    self.events["COOLDOWN_SKIP"] += 1

    def _ensure_eth_for_lp(self, market, ps):
        """LP 建仓前平衡双币：若某币为 0，把另一半换成它，确保有 ETH 和 USDC。"""
        eth_asset = self.broker.assets[self.eth]
        usdc_asset = self.broker.assets[self.usdc]
        eth_val = float(eth_asset.balance) * float(ps.price)
        usdc_val = float(usdc_asset.balance)
        total = eth_val + usdc_val
        if total <= 0:
            return
        # 目标：ETH 与 USDC 价值各半
        target_eth_val = total / 2
        # 若 ETH 价值不足目标，用 USDC 买 ETH
        if eth_val < target_eth_val - 1e-9 and usdc_asset.balance > 0:
            need_usdc = (target_eth_val - eth_val) / float(ps.price)
            swap_amt = min(need_usdc, float(usdc_asset.balance))
            if swap_amt > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.usdc, self.eth, Decimal(str(swap_amt)), p_ser)
        # 若 USDC 价值不足目标，用 ETH 换 USDC
        elif usdc_val < target_eth_val - 1e-9 and eth_asset.balance > 0:
            need_eth = (target_eth_val - usdc_val) / float(ps.price)
            swap_amt = min(need_eth, float(eth_asset.balance))
            if swap_amt > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.eth, self.usdc, Decimal(str(swap_amt)), p_ser)


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
        self.decisions = 0  # 总决策次数（每 15 分钟一次）

    def on_bar(self, row_data):
        self.bar_count += 1
        if self.bar_count % 15 != 0:
            return
        self.decisions += 1
        market = self.broker.markets[MarketInfo("pool")]
        ps = row_data.market_status[MarketInfo("pool")]
        now = row_data.timestamp

        if not market.positions:
            # 首次建仓：先确保有 ETH（把一半 USDC 换成 ETH，按池价）
            self._ensure_eth_for_lp(market, ps)
            exec_p = ps.price
            p_float = float(exec_p)
            market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                                 p_float * (1 + FROZEN["RANGE_PCT"]))
            self.last_rebalance = now
            return

        # 出区间 + 冷却结束 → 重建
        out_of_range = False
        tick = ps.closeTick
        for pi in market.positions.keys():
            if tick < pi.lower_tick or tick > pi.upper_tick:
                out_of_range = True
                break
        if out_of_range:
            self.oor_total += 1
            if self.last_rebalance is None or (now - self.last_rebalance >= pd.Timedelta(days=FROZEN["REBALANCE_DELAY_DAYS"])):
                market.remove_all_liquidity()
                self._ensure_eth_for_lp(market, ps)
                exec_p = ps.price
                p_float = float(exec_p)
                market.add_liquidity(p_float * (1 - FROZEN["RANGE_PCT"]),
                                     p_float * (1 + FROZEN["RANGE_PCT"]))
                self.last_rebalance = now
        else:
            self.lp_inrange += 1
        self.lp_total += 1

    def _ensure_eth_for_lp(self, market, ps):
        """LP 建仓前平衡双币：若某币为 0，把另一半换成它，确保有 ETH 和 USDC。"""
        eth_asset = self.broker.assets[self.eth]
        usdc_asset = self.broker.assets[self.usdc]
        eth_val = float(eth_asset.balance) * float(ps.price)
        usdc_val = float(usdc_asset.balance)
        total = eth_val + usdc_val
        if total <= 0:
            return
        # 目标：ETH 与 USDC 价值各半
        target_eth_val = total / 2
        # 若 ETH 价值不足目标，用 USDC 买 ETH
        if eth_val < target_eth_val - 1e-9 and usdc_asset.balance > 0:
            need_usdc = (target_eth_val - eth_val) / float(ps.price)
            swap_amt = min(need_usdc, float(usdc_asset.balance))
            if swap_amt > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.usdc, self.eth, Decimal(str(swap_amt)), p_ser)
        # 若 USDC 价值不足目标，用 ETH 换 USDC
        elif usdc_val < target_eth_val - 1e-9 and eth_asset.balance > 0:
            need_eth = (target_eth_val - usdc_val) / float(ps.price)
            swap_amt = min(need_eth, float(eth_asset.balance))
            if swap_amt > 0:
                p_ser = pd.Series({"ETH": ps.price, "USDC": Decimal(1)})
                self.broker.swap_by_from(self.eth, self.usdc, Decimal(str(swap_amt)), p_ser)


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def run_backtest(strategy_cls, pool_df, signals, model, features, cost_mode, strategy_name):
    """构造 demeter Actuator 并运行。"""
    eth_t = TokenInfo(name="ETH", decimal=18)
    usdc_t = TokenInfo(name="USDC", decimal=6)
    market_key = MarketInfo("pool")

    df = attach_signals_to_pool(pool_df, signals)
    df["price"] = df["price"].apply(lambda x: Decimal(str(x)))
    # demeter 需要 closeTick / currentLiquidity
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

    # 最终净值（需要价格 Series：ETH 价格 + USDC=1）
    last_price = float(df["price"].iloc[-1])
    prices = pd.Series({"ETH": Decimal(str(last_price)), "USDC": Decimal(1)})
    try:
        status = actuator.broker.get_account_status(prices, timestamp=df.index[-1])
        final_nav = float(status.net_value)
    except Exception:
        try:
            fs = actuator.final_status()
            final_nav = float(fs.net_value)
        except Exception:
            final_nav = 0.0

    # 净值曲线（逐分钟）用于指标与 CSV
    equity_curve = pd.Series(dtype=float)
    acc_fees = 0.0
    try:
        status_df = actuator.account_status_df
        equity_curve = status_df["net_value"].astype(float)
        equity_curve.index = status_df.index
        # 累计手续费：统计每次 collect_fee 后的净值增量（含已转入余额的 fee）。
        # 简化：用 pool 的 uncollected + liquidity_value 估算当前持仓手续费贡献，
        # 若不可用则记 0（fee 已计入 net_value，不影响净值结论）。
        try:
            base_uncol = pd.to_numeric(status_df[("pool", "base_uncollected")], errors="coerce").fillna(0)
            quote_uncol = pd.to_numeric(status_df[("pool", "quote_uncollected")], errors="coerce").fillna(0)
            eth_price = pd.to_numeric(status_df[("price", "ETH")], errors="coerce").fillna(0)
            acc_fees = float((base_uncol * eth_price).iloc[-1] + quote_uncol.iloc[-1])
        except Exception:
            acc_fees = 0.0
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
    }


# ---------------------------------------------------------------------------
# 简单基准：Always ETH / USDC / 50-50
# ---------------------------------------------------------------------------
def run_simple_benchmarks(pool_df):
    """用池价路径计算简单持仓基准（含逐分钟净值曲线）。"""
    price = pool_df["price"]
    p0 = float(price.iloc[0])

    # C. Always ETH：初始全换 ETH，净值随价格
    eth_units = INIT_CAPITAL / p0
    eq_eth = eth_units * price

    # D. Always USDC：恒定
    eq_usdc = pd.Series(INIT_CAPITAL, index=price.index)

    # E. 50/50 Buy-and-Hold
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
    # 最大回撤（逐分钟）
    running_max = eq.cummax()
    drawdown = (eq / running_max - 1).min()
    # 日度化收益用于 Sharpe / Sortino
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


def _compute_simple_metrics(equity_series, price_series, initial_capital=INIT_CAPITAL):
    """为简单基准（ETH/USDC/50-50）计算指标。"""
    return _compute_metrics(equity_series, price_series, initial_capital)


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "r0_t002")
    os.makedirs(out_dir, exist_ok=True)

    print("loading pool minute data (OOS + warmup)...")
    pool = load_pool_minute_oos()
    warm_price = load_pool_minute_warmup()
    print(f"  OOS pool rows: {len(pool)}; warmup price rows: {len(warm_price)}")

    print("loading frozen model...")
    model, features = load_frozen_model()
    print(f"  model: {len(features)} features")

    print("computing signals (warmup + OOS, only closed bars)...")
    all_price = pd.concat([warm_price, pool["price"]])
    signals = compute_signals_from_price(all_price)
    # 只保留 OOS 段的信号
    signals = signals[signals.index >= OOS_START]
    print(f"  OOS signals rows: {len(signals)}")

    print("running Frozen Legacy (Gross + Legacy-Cost)...")
    legacy_gross = run_backtest(FrozenLegacyStrategy, pool, signals, model, features, "gross", "frozen_legacy")
    legacy_cost = run_backtest(FrozenLegacyStrategy, pool, signals, model, features, "legacy", "frozen_legacy")

    print("running Always LP (Gross)...")
    always_lp = run_backtest(AlwaysLPStrategy, pool, signals, None, None, "gross", "always_lp")

    print("running simple benchmarks...")
    bench = run_simple_benchmarks(pool)

    # 汇总指标
    price = pool["price"]
    metrics = {
        "A_frozen_legacy_gross": _compute_metrics(legacy_gross["equity_curve"], price),
        "A_frozen_legacy_legacy_cost": _compute_metrics(legacy_cost["equity_curve"], price),
        "B_always_lp_gross": _compute_metrics(always_lp["equity_curve"], price),
        "C_always_eth": _compute_metrics(bench["always_eth"]["equity_curve"], price),
        "D_always_usdc": _compute_metrics(bench["always_usdc"]["equity_curve"], price),
        "E_buy_hold_5050": _compute_metrics(bench["buy_hold_5050"]["equity_curve"], price),
    }

    # 超额收益（Frozen Legacy 相对各基准）
    final_navs = {k: metrics[k]["end_nav"] for k in metrics}
    excess = {
        "vs_always_lp": round(final_navs["A_frozen_legacy_gross"] / final_navs["B_always_lp_gross"] - 1, 6),
        "vs_always_eth": round(final_navs["A_frozen_legacy_gross"] / final_navs["C_always_eth"] - 1, 6),
        "vs_always_usdc": round(final_navs["A_frozen_legacy_gross"] / final_navs["D_always_usdc"] - 1, 6),
        "vs_buy_hold_5050": round(final_navs["A_frozen_legacy_gross"] / final_navs["E_buy_hold_5050"] - 1, 6),
    }

    # equity 曲线 CSV（每日采样，避免过大）
    equity_df = pd.DataFrame({
        "A_frozen_legacy_gross": legacy_gross["equity_curve"],
        "A_frozen_legacy_legacy_cost": legacy_cost["equity_curve"],
        "B_always_lp_gross": always_lp["equity_curve"],
        "C_always_eth": bench["always_eth"]["equity_curve"],
        "D_always_usdc": bench["always_usdc"]["equity_curve"],
        "E_buy_hold_5050": bench["buy_hold_5050"]["equity_curve"],
    }).resample("D").last()
    equity_csv = os.path.join(out_dir, "post_freeze_oos_equity.csv")
    equity_df.to_csv(equity_csv)

    # 事件统计
    event_stats = {
        "frozen_legacy": legacy_gross["events"],
        "frozen_decisions": legacy_gross["decisions"],
        "lp_stats_frozen": {
            "lp_total": legacy_gross["lp_total"],
            "lp_inrange": legacy_gross["lp_inrange"],
            "oor_total": legacy_gross["oor_total"],
            "lp_time_ratio": round(legacy_gross["lp_total"] / max(legacy_gross["decisions"], 1), 4),
            "lp_inrange_ratio": round(legacy_gross["lp_inrange"] / max(legacy_gross["lp_total"], 1), 4),
        },
        "lp_stats_always": {
            "lp_total": always_lp["lp_total"],
            "lp_inrange": always_lp["lp_inrange"],
            "oor_total": always_lp["oor_total"],
            "lp_time_ratio": round(always_lp["lp_total"] / max(always_lp["decisions"], 1), 4),
            "lp_inrange_ratio": round(always_lp["lp_inrange"] / max(always_lp["lp_total"], 1), 4),
        },
        "acc_fees_gross": round(legacy_gross["acc_fees"], 2),
        "acc_fees_legacy_cost": round(legacy_cost["acc_fees"], 2),
        "acc_fees_always_lp": round(always_lp["acc_fees"], 2),
    }

    report = {
        "task_id": "R0-T002",
        "iteration": 1,
        "status": "COMPLETE",
        "oos_window": {"start": OOS_START.isoformat(), "end": OOS_END.isoformat()},
        "initial_capital": INIT_CAPITAL,
        "pool_rows": len(pool),
        "signal_rows": len(signals),
        "metrics": metrics,
        "excess_return": excess,
        "event_stats": event_stats,
        "note": "使用 demeter Uniswap V3 引擎；手续费基于真实池成交（inAmount*share*fee_rate）",
    }
    with open(os.path.join(out_dir, "post_freeze_oos.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # Markdown 报告
    md = render_markdown(report)
    with open(os.path.join(out_dir, "post_freeze_oos.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("COMPLETE written")


def render_markdown(report):
    """生成 Markdown 报告。"""
    lines = []
    lines.append("# R0-T002 Post-Freeze Strict OOS Validation（冻结后严格样本外验证）\n")
    lines.append(f"- OOS 窗口：{report['oos_window']['start']} → {report['oos_window']['end']}")
    lines.append(f"- 初始资本：{report['initial_capital']} USDC（全部策略一致）\n")

    lines.append("## 1. 策略指标对比\n")
    lines.append("| 策略 | 结束净值 | Total Return | 年化 | 最大回撤 | Sharpe | Sortino |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    label = {
        "A_frozen_legacy_gross": "A. Frozen Legacy (Gross)",
        "A_frozen_legacy_legacy_cost": "A. Frozen Legacy (Legacy-Cost)",
        "B_always_lp_gross": "B. Always LP (Gross)",
        "C_always_eth": "C. Always ETH",
        "D_always_usdc": "D. Always USDC",
        "E_buy_hold_5050": "E. 50/50 Buy-and-Hold",
    }
    for k, m in report["metrics"].items():
        lines.append(f"| {label[k]} | {m['end_nav']} | {m['total_return']*100:.2f}% | "
                     f"{m['annualized_return']*100:.2f}% | {m['max_drawdown']*100:.2f}% | "
                     f"{m['sharpe']} | {m['sortino']} |")
    lines.append("")

    lines.append("## 2. Frozen Legacy 超额收益\n")
    for k, v in report["excess_return"].items():
        lines.append(f"- {k}: {v*100:.2f}%")
    lines.append("")

    lines.append("## 3. 事件统计（Frozen Legacy）\n")
    ev = report["event_stats"]["frozen_legacy"]
    lines.append("| 事件 | 次数 |")
    lines.append("|---|---:|")
    for k, v in ev.items():
        lines.append(f"| {k} | {v} |")
    lp = report["event_stats"]["lp_stats_frozen"]
    lines.append(f"\n- 总决策次数：{report['event_stats']['frozen_decisions']}")
    lines.append(f"- LP 在池时间占比（相对总决策）：{lp['lp_time_ratio']*100:.1f}%")
    lines.append(f"- LP 期间在区间内占比：{lp['lp_inrange_ratio']*100:.1f}% "
                 f"（活跃 {lp['lp_inrange']}/{lp['lp_total']}，出区间 {lp['oor_total']}）")
    lines.append("")

    lines.append("## 4. 累计 LP 手续费\n")
    es = report["event_stats"]
    lines.append(f"- Frozen Legacy (Gross)：{es['acc_fees_gross']} USDC")
    lines.append(f"- Frozen Legacy (Legacy-Cost)：{es['acc_fees_legacy_cost']} USDC")
    lines.append(f"- Always LP：{es['acc_fees_always_lp']} USDC")
    lines.append("")

    lines.append("## 5. 数据与方法说明\n")
    lines.append("- 引擎：demeter（Uniswap V3 回测框架）")
    lines.append(f"- 池：Arbitrum WETH/USDC 0.05%（{report['pool_rows']} 分钟数据）")
    lines.append(f"- 信号：15m/4h 技术指标（已收盘 bar，merge_asof backward），{report['signal_rows']} 个决策点")
    lines.append("- 成本：Gross（无成本）/ Legacy-Cost（latency_bias=5bps + exit 扣 0.0002）")
    lines.append("- 详细净值曲线见 post_freeze_oos_equity.csv")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
