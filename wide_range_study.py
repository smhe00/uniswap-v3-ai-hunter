import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demeter import TokenInfo, Actuator, Strategy, MarketInfo, Asset
from demeter.uniswap import UniLpMarket, UniV3Pool
import optuna
import pickle
import glob
import os
from decimal import Decimal
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 1. Load Data ---
def load_all_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.minute.csv")))
    print(f"Loading {len(files)} days of minute data for Wide-Range Optimization...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_float'] = (1.0001 ** pd.to_numeric(df['closeTick'], errors='coerce')) * 1e12
    for c in ['netAmount0', 'netAmount1', 'closeTick', 'openTick', 'lowestTick', 'highestTick', 'inAmount0', 'inAmount1', 'currentLiquidity']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['currentLiquidity'] = df['currentLiquidity'].astype(object)
    df.set_index('timestamp', inplace=True)
    return df

DATA_DIR = 'uniswap_data/UNIV3_DATA'
full_minute_df = load_all_data(DATA_DIR)

# --- 2. Multi-Timeframe Feature Engineering ---
print("Calculating MTF Signals...")
# Micro (15m)
df_15m = full_minute_df['price_float'].resample('15Min').ohlc().dropna()
df_15m.rename(columns={'close': 'close_float'}, inplace=True)
df_15m['close'] = df_15m['close_float']
df_15m.ta.rsi(length=14, append=True)
df_15m.ta.natr(length=14, append=True)
df_15m.ta.adx(length=14, append=True)
df_15m.ta.bbands(length=20, append=True)
bbl, bbm, bbu = [c for c in df_15m.columns if 'BBL_20' in c][0], [c for c in df_15m.columns if 'BBM_20' in c][0], [c for c in df_15m.columns if 'BBU_20' in c][0]
df_15m['bb_width'] = (df_15m[bbu] - df_15m[bbl]) / (df_15m[bbm] + 1e-9)
for col in ['RSI_14', 'NATR_14', 'ADX_14', 'bb_width']:
    for lag in [1, 2, 4]: df_15m[f'{col}_lag{lag}'] = df_15m[col].shift(lag)

# Macro (4h)
df_4h = full_minute_df['price_float'].resample('4H').ohlc().dropna()
df_4h.rename(columns={'close': 'close_float'}, inplace=True)
df_4h['close'] = df_4h['close_float']
df_4h.ta.rsi(length=14, append=True)
df_4h.ta.ema(length=50, append=True)
df_4h.rename(columns={'RSI_14': 'macro_rsi', 'EMA_50': 'macro_ema'}, inplace=True)

df_15m = df_15m.join(df_4h[['macro_rsi', 'macro_ema']], how='left').ffill()
df_15m.dropna(inplace=True)

with open('v3_experimental_15m_tag/models_15m.pkl', 'rb') as f:
    m_data = pickle.load(f); xgb_model, features = m_data['xgb'], m_data['features']
df_15m['xgb_prob'] = xgb_model.predict_proba(df_15m[features])[:, 1]

shared_signals = df_15m[['xgb_prob', 'NATR_14', 'RSI_14', 'macro_rsi', 'macro_ema', 'close_float']]

# --- 3. Wide Range Strategy Class ---
eth_t = TokenInfo(name="ETH", decimal=18); usdc_t = TokenInfo(name="USDC", decimal=6)
market_key = MarketInfo("pool")

class WideRangeStrategy(Strategy):
    def __init__(self, p):
        super().__init__()
        self.p = p # range, risk_thresh, m_bull, m_bear
        self.state = "POOL"
        self.last_rebalance = None
        self.bar_count = 0
        self.latency_bias = 0.0005 

    def on_bar(self, row_data):
        self.bar_count += 1
        if self.bar_count % 15 != 0: return
        
        market = self.broker.markets[market_key]
        ps = row_data.market_status[market_key]
        now = row_data.timestamp
        
        is_risk = (ps.xgb_prob > self.p['risk_thresh']) or (ps.NATR_14 > 2.0)
        is_bull = ps.macro_rsi > self.p['m_bull']
        is_bear = ps.macro_rsi < self.p['m_bear']
        
        can_rebalance = (self.last_rebalance is None) or (now - self.last_rebalance >= timedelta(days=4))

        if is_risk:
            if self.state == "POOL":
                market.remove_all_liquidity()
                self.broker.subtract_from_balance(usdc_t, self.broker.assets[usdc_t].balance * Decimal("0.0002"))
            
            exec_p = ps.price * Decimal(str(1 - self.latency_bias))
            p_ser = pd.Series({'ETH': exec_p, 'USDC': Decimal(1)})
            
            if is_bull:
                if self.state != "ETH":
                    u_bal = self.broker.assets[usdc_t].balance
                    if u_bal > 0: self.broker.swap_by_from(usdc_t, eth_t, u_bal, p_ser)
                    self.state = "ETH"
            elif is_bear:
                if self.state != "USDC":
                    e_bal = self.broker.assets[eth_t].balance
                    if e_bal > 0: self.broker.swap_by_from(eth_t, usdc_t, e_bal, p_ser)
                    self.state = "USDC"
            else:
                self.state = "MIXED"
        else:
            oor = False
            if self.state == "POOL" and market.positions:
                tick = ps.closeTick
                for pi in market.positions.keys():
                    if tick < pi.lower_tick or tick > pi.upper_tick: oor = True; break
            
            if self.state != "POOL" or (oor and can_rebalance):
                if self.state == "POOL": market.remove_all_liquidity()
                exec_p = ps.price * Decimal(str(1 + self.latency_bias))
                market.add_liquidity(float(exec_p)*(1-self.p['range']), float(exec_p)*(1+self.p['range']))
                self.state = "POOL"
                self.last_rebalance = now

# --- 4. Optimization ---
def objective(trial):
    params = {
        'range': trial.suggest_float('range', 0.08, 0.12),
        'risk_thresh': trial.suggest_float('risk_thresh', 0.40, 0.70),
        'm_bull': trial.suggest_int('m_bull', 50, 65),
        'm_bear': trial.suggest_int('m_bear', 35, 50)
    }
    
    run_df = full_minute_df.join(shared_signals, how='inner').ffill()
    run_df['price'] = run_df['price_float'].apply(lambda x: Decimal(str(x)))
    
    actuator = Actuator()
    actuator.set_assets([Asset(eth_t, Decimal(5)), Asset(usdc_t, Decimal(10000))])
    actuator.broker._quote_token = usdc_t
    market = UniLpMarket(market_key, UniV3Pool(eth_t, usdc_t, 0.05, usdc_t))
    market.data = run_df
    actuator.broker.add_market(market)
    actuator.strategy = WideRangeStrategy(params)
    actuator.run()
    
    final_nav = float(actuator.broker.get_account_status(pd.Series({'ETH': run_df.iloc[-1].price, 'USDC': Decimal(1)})).net_value)
    return final_nav

print("--- STARTING WIDE-RANGE SEARCH (8-12%) ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_p = study.best_params
print(f"\nBest Wide Params: {best_p}")

# --- 5. PEAK START STRESS TEST ---
print("\n--- PEAK START STRESS TEST (Starting 2025-08-24) ---")
peak_date = datetime(2025, 8, 24)
peak_df = full_minute_df[full_minute_df.index >= peak_date].join(shared_signals, how='inner').ffill()
peak_df['price'] = peak_df['price_float'].apply(lambda x: Decimal(str(x)))

act = Actuator(); act.set_assets([Asset(eth_t, Decimal(5)), Asset(usdc_t, Decimal(10000))])
act.broker._quote_token = usdc_t
mkt = UniLpMarket(market_key, UniV3Pool(eth_t, usdc_t, 0.05, usdc_t))
mkt.data = peak_df; act.broker.add_market(mkt)
act.strategy = WideRangeStrategy(best_p); act.run()
final_peak = float(act.broker.get_account_status(pd.Series({'ETH': peak_df.iloc[-1].price, 'USDC': Decimal(1)})).net_value)

start_p = float(peak_df.iloc[0].price); end_p = float(peak_df.iloc[-1].price)
eth_base = (5 * end_p + 10000)
print(f"Peak Start Final Value: ${final_peak:.2f}")
print(f"ETH Baseline Value:   ${eth_base:.2f}")
print(f"Alpha ROI: {(final_peak/eth_base - 1)*100:.2f}%")

with open('wide_golden_params.pkl', 'wb') as f:
    pickle.dump(best_p, f)
