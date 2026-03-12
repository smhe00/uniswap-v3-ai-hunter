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

# --- 1. Data Preparation (Full Year) ---
def load_all_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.minute.csv")))
    print(f"Loading {len(files)} days of minute data for Dual-Engine Optimization...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_float'] = (1.0001 ** pd.to_numeric(df['closeTick'], errors='coerce')) * 1e12
    # Ensure numeric for Demeter later
    for c in ['netAmount0', 'netAmount1', 'closeTick', 'openTick', 'lowestTick', 'highestTick', 'inAmount0', 'inAmount1', 'currentLiquidity']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['currentLiquidity'] = df['currentLiquidity'].astype(object)
    df.set_index('timestamp', inplace=True)
    return df

DATA_DIR = 'uniswap_data/UNIV3_DATA'
full_minute_df = load_all_data(DATA_DIR)

# --- 2. Feature Engineering (Multi-Timeframe) ---
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

# Macro (4h) - for long-term regime detection
df_4h = full_minute_df['price_float'].resample('4H').ohlc().dropna()
df_4h.rename(columns={'close': 'close_float'}, inplace=True)
df_4h['close'] = df_4h['close_float']
df_4h.ta.ema(length=50, append=True) # Long term trend
df_4h.ta.rsi(length=14, append=True) # Long term momentum
df_4h.rename(columns={'EMA_50': 'macro_ema', 'RSI_14': 'macro_rsi'}, inplace=True)

# Join Macro to 15m
df_15m = df_15m.join(df_4h[['macro_ema', 'macro_rsi']], how='left').ffill()
df_15m.dropna(inplace=True)

# Load XGBoost Risk Model (trained on 1h/15m)
with open('v3_experimental_15m_tag/models_15m.pkl', 'rb') as f:
    m_data = pickle.load(f); xgb_model, features = m_data['xgb'], m_data['features']
df_15m['xgb_prob'] = xgb_model.predict_proba(df_15m[features])[:, 1]

# Shared indicators for optimization loop
shared_signals = df_15m[['xgb_prob', 'NATR_14', 'RSI_14', 'macro_ema', 'macro_rsi', 'close_float']]

# --- 3. Dual-Engine Strategy Class ---
class DualEngineStrategy(Strategy):
    def __init__(self, p):
        super().__init__()
        self.p = p # range, risk_thresh, macro_rsi_bull, macro_rsi_bear
        self.state = "POOL"
        self.last_rebalance = None
        self.bar_count = 0
        self.latency_bias = 0.0005

    def on_bar(self, row_data):
        self.bar_count += 1
        if self.bar_count % 15 != 0: return
        
        market = self.broker.markets[MarketInfo("pool")]
        ps = row_data.market_status[MarketInfo("pool")]
        now = row_data.timestamp
        
        # MACRO FILTER
        is_macro_bull = ps.macro_rsi > self.p['m_bull']
        is_macro_bear = ps.macro_rsi < self.p['m_bear']
        
        # MICRO DECISION
        is_risk = (ps.xgb_prob > self.p['risk_thresh']) or (ps.NATR_14 > 2.0)
        
        can_rebalance = (self.last_rebalance is None) or (now - self.last_rebalance >= timedelta(days=4))

        if is_risk:
            if self.state == "POOL":
                market.remove_all_liquidity()
                self.broker.subtract_from_balance(usdc_t, self.broker.assets[usdc_t].balance * Decimal("0.0002")) # Latency fee
            
            exec_p = ps.price * Decimal(str(1 - self.latency_bias))
            p_ser = pd.Series({'ETH': exec_p, 'USDC': Decimal(1)})
            
            if is_macro_bull: # Trust macro trend for asset hold
                if self.state != "ETH":
                    u_bal = self.broker.assets[usdc_t].balance
                    if u_bal > 0: self.broker.swap_by_from(usdc_t, eth_t, u_bal, p_ser)
                    self.state = "ETH"
            elif is_macro_bear:
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

# --- 4. Optimization Loop ---
eth_t = TokenInfo(name="ETH", decimal=18); usdc_t = TokenInfo(name="USDC", decimal=6)

def objective(trial):
    params = {
        'range': trial.suggest_float('range', 0.02, 0.05),
        'risk_thresh': trial.suggest_float('risk_thresh', 0.40, 0.65),
        'm_bull': trial.suggest_int('m_bull', 50, 65),
        'm_bear': trial.suggest_int('m_bear', 35, 50)
    }
    
    # We'll use 6 months for the "search" phase to be faster, then validate full year
    search_df = full_minute_df.iloc[-260000:].join(shared_signals, how='inner').ffill()
    search_df['price'] = search_df['price_float'].apply(lambda x: Decimal(str(x)))
    
    actuator = Actuator()
    actuator.set_assets([Asset(eth_t, Decimal(5)), Asset(usdc_t, Decimal(10000))])
    actuator.broker._quote_token = usdc_t
    market = UniLpMarket(MarketInfo("pool"), UniV3Pool(eth_t, usdc_t, 0.05, usdc_t))
    market.data = search_df
    actuator.broker.add_market(market)
    
    actuator.strategy = DualEngineStrategy(params)
    actuator.run()
    
    final_nav = float(actuator.broker.get_account_status(pd.Series({'ETH': search_df.iloc[-1].price, 'USDC': Decimal(1)})).net_value)
    return final_nav

print("Starting Dual-Engine Parameter Optimization (20 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_p = study.best_params
print(f"Best MTF Params: {best_p}")

# --- 5. Final 1-Year Validation ---
print("\n--- FINAL 1-YEAR DUAL-ENGINE BATTLE ---")
final_df = full_minute_df.join(shared_signals, how='inner').ffill()
final_df['price'] = final_df['price_float'].apply(lambda x: Decimal(str(x)))

actuator = Actuator()
actuator.set_assets([Asset(eth_t, Decimal(5)), Asset(usdc_t, Decimal(10000))])
actuator.broker._quote_token = usdc_t
market = UniLpMarket(MarketInfo("pool"), UniV3Pool(eth_t, usdc_t, 0.05, usdc_t))
market.data = final_df
actuator.broker.add_market(market)
actuator.strategy = DualEngineStrategy(best_p)
actuator.run()

status = actuator.broker.get_account_status(pd.Series({'ETH': final_df.iloc[-1].price, 'USDC': Decimal(1)}))
print(f"1-Year Final Net Value: {status.net_value} USDC")
roi = (float(status.net_value)/20863 - 1)*100
print(f"Ultimate 1-Year ROI: {roi:.2f}%")

# Save Result
with open('dual_engine_golden_params.pkl', 'wb') as f:
    pickle.dump(best_p, f)
