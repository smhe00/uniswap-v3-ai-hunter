import pandas as pd
import numpy as np
import os
import glob
import pickle
import random
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- 1. Precompute Signals ---
def precompute_hunter_signals():
    print("Pre-computing dynamic AI signals...")
    DATA_DIR = 'uniswap_data/UNIV3_DATA'
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.minute.csv")))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_float'] = (1.0001 ** pd.to_numeric(df['closeTick'], errors='coerce')) * 1e12
    df.set_index('timestamp', inplace=True)
    
    df_15m = df['price_float'].resample('15Min').ohlc().dropna()
    import pandas_ta as ta
    df_15m.ta.rsi(length=14, append=True)
    df_15m.ta.natr(length=14, append=True)
    df_15m.ta.ema(length=20, append=True)
    df_15m.ta.ema(length=50, append=True)
    df_15m.ta.bbands(length=20, append=True)
    bbl, bbm, bbu = [c for c in df_15m.columns if 'BBL_20' in c][0], [c for c in df_15m.columns if 'BBM_20' in c][0], [c for c in df_15m.columns if 'BBU_20' in c][0]
    df_15m['bb_width'] = (df_15m[bbu] - df_15m[bbl]) / (df_15m[bbm] + 1e-9)
    for col in ['RSI_14', 'NATR_14', 'bb_width']:
        for lag in [1, 2, 4]: df_15m[f'{col}_lag{lag}'] = df_15m[col].shift(lag)
    
    with open('v3_experimental_15m_tag/models_15m.pkl', 'rb') as f:
        m_data = pickle.load(f); xgb_model, features = m_data['xgb'], m_data['features']
    
    for f_name in features:
        if f_name not in df_15m.columns: df_15m[f_name] = 0
        
    df_15m['xgb_prob'] = xgb_model.predict_proba(df_15m[features])[:, 1]
    return df_15m

def fast_p(data_hex):
    try:
        d = str(data_hex)[2:]
        tick_hex = d[256:320]
        tick = int(tick_hex, 16)
        if tick >= 2**255: tick -= 2**256
        return (1.0001 ** tick) * 1e12
    except:
        return None

def run_adaptive_hunter_sim(start_date, duration_days, signals_df):
    end_date = start_date + timedelta(days=duration_days)
    DATA_DIR = 'uniswap_data/UNIV3_DATA'
    
    raw_files = []
    curr = start_date
    while curr <= end_date:
        f = os.path.join(DATA_DIR, f"arbitrum-0xc6962004f452be9203591991d15f6b388e09e8d0-{curr.strftime('%Y-%m-%d')}.raw.csv")
        if os.path.exists(f): raw_files.append(f)
        curr += timedelta(days=1)
    
    if not raw_files: return None
    
    cap = 10000.0
    state = "POOL"
    p_entry, p_low, p_high, L = 0, 0, 0, 0
    acc_fees = 0
    
    for f in raw_files:
        day_raw = pd.read_csv(f)
        # Look for the signature 0xc42079f9 in any of the topics
        swaps = day_raw[day_raw['topics'].str.contains('0xc42079f9', na=False)].copy()
        if swaps.empty: continue
        
        swaps['price'] = swaps['data'].apply(fast_p)
        swaps.dropna(subset=['price'], inplace=True)
        swaps['timestamp'] = pd.to_datetime(swaps['block_timestamp'])
        swaps.sort_values('timestamp', inplace=True)
        
        merged = pd.merge_asof(swaps, signals_df, left_on='timestamp', right_index=True, direction='backward')
        if merged.empty: continue
        
        for _, row in merged.iterrows():
            p_curr = row['price']
            is_risk = (row['xgb_prob'] > 0.55)
            is_bull = (row['EMA_20'] > row['EMA_50']) and (row['RSI_14'] > 55)
            is_bear = (row['EMA_20'] < row['EMA_50']) and (row['RSI_14'] < 45)
            target_range = np.clip(float(row['NATR_14']) * 0.02, 0.015, 0.10) 

            if state == "POOL":
                if p_entry == 0: # Init
                    p_entry = p_curr
                    p_low, p_high = p_entry*(1-0.03), p_entry*(1+0.03)
                    L = 1 / (1 - np.sqrt(p_low/p_entry))
                
                out_of_bounds = (p_curr <= p_low or p_curr >= p_high)
                if is_risk or out_of_bounds:
                    def get_v3_val(p, pl, ph): return 2*np.sqrt(p) - np.sqrt(pl) - p/np.sqrt(ph)
                    val_ratio = get_v3_val(np.clip(p_curr, p_low, p_high), p_low, p_high) / get_v3_val(p_entry, p_low, p_high)
                    cap = cap * val_ratio + acc_fees - 10
                    acc_fees = 0
                    if is_risk:
                        if is_bull: state = "ETH"
                        elif is_bear: state = "USDC"
                        else: state = "MIXED"
                        p_entry = p_curr
                    else:
                        p_entry = p_curr
                        p_low, p_high = p_entry*(1-target_range), p_entry*(1+target_range)
                        L = 1 / (1 - np.sqrt(p_low/p_entry))
                else:
                    step_fee = (0.15/365/24/60/60) * L * cap
                    acc_fees += step_fee
            else:
                if state == "ETH": cap *= (p_curr / p_entry); p_entry = p_curr
                elif state == "MIXED": cap *= (1 + 0.5*(p_curr/p_entry - 1)); p_entry = p_curr
                if not is_risk:
                    state = "POOL"; cap -= 10; p_entry = p_curr
                    p_low, p_high = p_entry*(1-target_range), p_entry*(1+target_range)
                    L = 1 / (1 - np.sqrt(p_low/p_entry)); acc_fees = 0
                    
    return (cap / 10000.0) - 1

print("--- ADAPTIVE HUNTER MONTE CARLO ---")
signals = precompute_hunter_signals()
start_limit = signals.index.min()
end_limit = signals.index.max() - timedelta(days=40)
all_dates = [start_limit + timedelta(days=i) for i in range((end_limit - start_limit).days)]
random.shuffle(all_dates)

results = []
for i in range(10):
    s_date = all_dates[i]
    duration = random.randint(25, 35)
    print(f"Run {i+1}/10: {s_date.strftime('%Y-%m-%d')}...")
    roi = run_adaptive_hunter_sim(s_date, duration, signals)
    if roi is not None and not np.isnan(roi):
        results.append(roi)
        print(f"  > ROI: {roi*100:.2f}%")
    else:
        print(f"  > FAILED")

if results:
    res_df = pd.Series(results)
    print(f"\nMEAN ROI: {res_df.mean()*100:.2f}% | SUCCESS: {(res_df > 0).mean()*100:.1f}%")
