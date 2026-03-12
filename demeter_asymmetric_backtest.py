import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demeter import TokenInfo, Actuator, Strategy, ChainType, MarketInfo, UnitDecimal, Asset
from demeter.uniswap import UniLpMarket, UniV3Pool, UniV3PoolStatus
import pandas_ta as ta
import pickle
import os
import glob
from decimal import Decimal

# --- 1. Data Preparation ---
def prepare_demeter_data(data_dir, days=365):
    files = sorted(glob.glob(os.path.join(data_dir, "*.minute.csv")))
    dfs = [pd.read_csv(f) for f in files[-days:]]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cols = ['netAmount0', 'netAmount1', 'closeTick', 'openTick', 'lowestTick', 'highestTick', 'inAmount0', 'inAmount1', 'currentLiquidity']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['currentLiquidity'] = df['currentLiquidity'].apply(lambda x: int(Decimal(str(x))))
    df.set_index('timestamp', inplace=True)
    return df

# --- 2. Advanced Asymmetric Strategy ---
class AsymmetricSmartStrategy(Strategy):
    def __init__(self, xgb_model, ga_params, features):
        super().__init__()
        self.xgb_model = xgb_model
        self.ga_params = ga_params 
        self.features = features
        self.last_rebalance_time = None
        self.state = "POOL" # POOL, HOLD_ETH, HOLD_USDC, HOLD_RATIO
        self.bar_count = 0

    def on_bar(self, row_data: pd.Series):
        self.bar_count += 1
        if self.bar_count % 15 != 0:
            return

        market = self.broker.markets[market_key]
        pool_status = row_data.market_status[market_key]
        now = row_data.timestamp
        
        # Signals from pre-calculated columns
        is_risk = pool_status.risk_signal
        is_bull = pool_status.bull_trend
        is_bear = pool_status.bear_trend
        
        can_rebalance = (self.last_rebalance_time is None) or \
                        (now - self.last_rebalance_time >= timedelta(days=4))

        if is_risk:
            # --- EMERGENCY EXIT LOGIC ---
            if self.state == "POOL":
                market.remove_all_liquidity()
                print(f"[{now}] RISK! Exiting Pool...")
            
            prices_series = pd.Series({'ETH': pool_status.price, 'USDC': Decimal(1)})
            if is_bull:
                if self.state != "HOLD_ETH":
                    usdc_bal = self.broker.assets[usdc].balance
                    if usdc_bal > 0:
                        self.broker.swap_by_from(usdc, eth, usdc_bal, prices_series)
                    self.state = "HOLD_ETH"
                    print(f"[{now}] Bullish Risk -> HOLDING 100% ETH")
            elif is_bear:
                if self.state != "HOLD_USDC":
                    eth_bal = self.broker.assets[eth].balance
                    if eth_bal > 0:
                        self.broker.swap_by_from(eth, usdc, eth_bal, prices_series)
                    self.state = "HOLD_USDC"
                    print(f"[{now}] Bearish Risk -> HOLDING 100% USDC")
            else:
                self.state = "HOLD_RATIO" # Stay as is
                if self.state == "POOL": print(f"[{now}] Unknown Risk -> HOLDING RATIO")
        else:
            # --- SAFE TO MAKE MARKET LOGIC ---
            out_of_range = False
            if self.state == "POOL":
                if not market.positions:
                    self.state = "RE-ENTER" # Force re-entry
                else:
                    curr_tick = pool_status.closeTick
                    for pos_info, pos in market.positions.items():
                        if curr_tick < pos_info.lower_tick or curr_tick > pos_info.upper_tick:
                            out_of_range = True
                        break
            
            if (self.state != "POOL") or (out_of_range and can_rebalance):
                if self.state == "POOL":
                    market.remove_all_liquidity()
                
                # Re-center and add liquidity
                price = float(pool_status.price)
                # Ensure we have a balanced ratio if possible, or just add what we have
                # Demeter add_liquidity will use assets to match the ratio of the range
                market.add_liquidity(price * 0.96, price * 1.04)
                self.state = "POOL"
                self.last_rebalance_time = now
                print(f"[{now}] MARKET SAFE -> Deploying ±4% Pool @ {price:.2f}")

# --- 3. Main Runner ---
print("Preparing Asymmetric Demeter Backtest...")
DATA_DIR = 'uniswap_data/UNIV3_DATA'
raw_df = prepare_demeter_data(DATA_DIR, days=365)

print("Pre-calculating signals...")
raw_df['price_float'] = (1.0001 ** raw_df['closeTick']) * 1e12
df_15m = raw_df['price_float'].resample('15Min').ohlc().dropna()
df_15m.rename(columns={'close': 'close_float'}, inplace=True)
df_15m['close'] = df_15m['close_float']

# Risk
df_15m.ta.rsi(length=14, append=True)
df_15m.ta.natr(length=14, append=True)
df_15m.ta.adx(length=14, append=True)
df_15m.ta.bbands(length=20, append=True)
bbl, bbm, bbu = [c for c in df_15m.columns if 'BBL_20' in c][0], [c for c in df_15m.columns if 'BBM_20' in c][0], [c for c in df_15m.columns if 'BBU_20' in c][0]
df_15m['bb_width'] = (df_15m[bbu] - df_15m[bbl]) / (df_15m[bbm] + 1e-9)
for col in ['RSI_14', 'NATR_14', 'ADX_14', 'bb_width']:
    for lag in [1, 2, 4]: df_15m[f'{col}_lag{lag}'] = df_15m[col].shift(lag)

# Trend
df_15m.ta.ema(length=20, append=True)
df_15m.ta.ema(length=50, append=True)

df_15m.dropna(inplace=True)

with open('v3_experimental_15m_tag/models_15m.pkl', 'rb') as f:
    m_data = pickle.load(f)
    xgb_model, ga_params, features = m_data['xgb'], m_data['ga'], m_data['features']

df_15m['xgb_prob'] = xgb_model.predict_proba(df_15m[features])[:, 1]
df_15m['risk_signal'] = (df_15m['xgb_prob'] > 0.45) | (df_15m['NATR_14'] > df_15m['NATR_14'].quantile(0.90))
df_15m['bull_trend'] = (df_15m['EMA_20'] > df_15m['EMA_50']) & (df_15m['DMP_14'] > df_15m['DMN_14']) & (df_15m['RSI_14'] > 60)
df_15m['bear_trend'] = (df_15m['EMA_20'] < df_15m['EMA_50']) & (df_15m['DMN_14'] > df_15m['DMP_14']) & (df_15m['RSI_14'] < 40)

raw_df = raw_df.join(df_15m[['risk_signal', 'bull_trend', 'bear_trend']], how='left').ffill()
raw_df.dropna(subset=['risk_signal'], inplace=True)
raw_df['price'] = raw_df['price_float'].apply(lambda x: Decimal(str(x)))

# Demeter Run
eth = TokenInfo(name="eth", decimal=18)
usdc = TokenInfo(name="usdc", decimal=6)
market_key = MarketInfo("eth_usdc_pool")

actuator = Actuator()
actuator.set_assets([Asset(eth, Decimal(5)), Asset(usdc, Decimal(10000))])
actuator.broker._quote_token = usdc

market = UniLpMarket(market_key, UniV3Pool(eth, usdc, 0.05, usdc)) 
market.data = raw_df
actuator.broker.add_market(market)

actuator.strategy = AsymmetricSmartStrategy(xgb_model, ga_params, features)
print("Starting Actuator (Asymmetric 1-Year)...")
actuator.run()

print("\n--- ASYMMETRIC DEMETER FINAL STATUS ---")
# Manually calculate final net value to avoid method errors
final_st = actuator.broker.get_account_status(raw_df.iloc[-1].price)
print(f"Final Net Value: {final_st.net_value} USDC")
print(f"Final Assets: ETH={final_st.tokens[eth]}, USDC={final_st.tokens[usdc]}")
