import pandas as pd
import numpy as np
import os, json, datetime, time
import ccxt
import pandas_ta as ta
import xgboost as xgb
import pickle
import warnings
from decimal import Decimal

warnings.filterwarnings('ignore')

# --- CONFIGURATION & GOLDEN DUAL-ENGINE PARAMETERS ---
MODEL_PATH = 'v3_experimental_15m_tag/models_15m.pkl'
STATE_FILE = 'agent_state.json'
PROXY = {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}

# Optimized Parameters from 1-Year Dual-Engine Battle (+40.44% ROI)
RANGE_PCT = 0.0297          # Optimal range +/- 2.97%
REBALANCE_DELAY_DAYS = 4    # Standard rebalance constraint
XGB_RISK_THRESHOLD = 0.62   # High-confidence risk threshold
MACRO_BULL_RSI = 58         # 4H RSI threshold for Bull regime
MACRO_BEAR_RSI = 48         # 4H RSI threshold for Bear regime

def get_dual_engine_signals():
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Fetching Multi-Timeframe Data (15m & 4H)...")
    try:
        exchange = ccxt.binance({
            'proxies': PROXY,
            'enableRateLimit': True,
        })
        
        # 1. Fetch 15m Data (Micro)
        ohlcv_15m = exchange.public_get_klines({'symbol': 'ETHUSDT', 'interval': '15m', 'limit': 200})
        df = pd.DataFrame(ohlcv_15m, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'close_ts', 'q_vol', 'trades', 't_b_vol', 't_q_vol', 'ignore'])
        df = df[['ts', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # 2. Fetch 4H Data (Macro)
        ohlcv_4h = exchange.public_get_klines({'symbol': 'ETHUSDT', 'interval': '4h', 'limit': 100})
        df_4h = pd.DataFrame(ohlcv_4h, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'close_ts', 'q_vol', 'trades', 't_b_vol', 't_q_vol', 'ignore'])
        df_4h = df_4h[['ts', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # --- Macro Feature Engineering ---
        df_4h.ta.rsi(length=14, append=True)
        macro_rsi = float(df_4h['RSI_14'].iloc[-1])
        
        # --- Micro Feature Engineering ---
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.adx(length=14, append=True)
        df.ta.natr(length=14, append=True)
        df.ta.bbands(length=20, append=True)
        
        bbl = [c for c in df.columns if c.startswith('BBL_20')][0]
        bbm = [c for c in df.columns if c.startswith('BBM_20')][0]
        bbu = [c for c in df.columns if c.startswith('BBU_20')][0]
        df['bb_width'] = (df[bbu] - df[bbl]) / (df[bbm] + 1e-9)
        
        for col in ['RSI_14', 'NATR_14', 'ADX_14', 'bb_width']:
            for lag in [1, 2, 4]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        df.dropna(inplace=True)
        curr = df.iloc[-1]
        
        # 3. Load 15m Brain
        with open(MODEL_PATH, 'rb') as f:
            m_data = pickle.load(f)
            xgb_model, ga_params, features = m_data['xgb'], m_data['ga'], m_data['features']
            
        # 4. Final Decision Logic
        # Micro Risk
        ga_ok = bool((curr['RSI_14'] > ga_params[0]) and (curr['RSI_14'] < ga_params[1]) and (curr['NATR_14'] < ga_params[2]))
        risk_prob = float(xgb_model.predict_proba(df[features].tail(1))[0, 1])
        xgb_ok = risk_prob < XGB_RISK_THRESHOLD
        vol_guard_ok = bool(curr['NATR_14'] < 2.0) # Aggressive vol guard
        
        # Macro Regime
        is_macro_bull = bool(macro_rsi > MACRO_BULL_RSI)
        is_macro_bear = bool(macro_rsi < MACRO_BEAR_RSI)
        
        is_active = ga_ok and xgb_ok and vol_guard_ok
        
        metrics = {
            'price': float(curr['close']),
            'macro_rsi': macro_rsi,
            'ga': ga_ok,
            'xgb_prob': risk_prob,
            'natr': float(curr['NATR_14']),
            'vol_guard': vol_guard_ok,
            'is_bull': is_macro_bull,
            'is_bear': is_macro_bear
        }
        return is_active, metrics
    except Exception as e:
        print(f"Error in signal generation: {e}")
        return False, None

def run_pulse_15m():
    is_active_signal, metrics = get_dual_engine_signals()
    if not metrics: return

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try: state = json.load(f)
            except: state = {'nav': 10000.0, 'mode': 'ACTIVE', 'last_rebalance': 0, 'history': []}
    else:
        state = {'nav': 10000.0, 'mode': 'ACTIVE', 'last_rebalance': 0, 'history': []}

    now_ts = time.time()
    current_price = metrics['price']
    days_since_reb = (now_ts - state.get('last_rebalance', 0)) / 86400
    old_mode = state.get('mode', 'ACTIVE')
    new_mode = 'ACTIVE' if is_active_signal else 'SAFE'
    action = "HOLD"
    
    if old_mode == 'ACTIVE' and new_mode == 'SAFE':
        # Asymmetric Decision based on MACRO
        if metrics['is_bull']: action = "EXIT POOL -> HOLD 100% ETH"
        elif metrics['is_bear']: action = "EXIT POOL -> HOLD 100% USDC"
        else: action = "EXIT POOL -> KEEP CURRENT RATIO"
        os.system(f"osascript -e 'display notification \"{action}\" with title \"🛡️ Dual-Engine Agent\"'")
    elif (new_mode == 'ACTIVE'):
        if old_mode != 'ACTIVE':
            action = f"RE-ENTER POOL (±{RANGE_PCT*100:.2f}% @ {current_price:.2f})"
            state['last_rebalance'] = now_ts
            os.system(f"osascript -e 'display notification \"Re-deploying liquidity.\" with title \"✅ Dual-Engine Agent\"'")
        elif days_since_reb >= REBALANCE_DELAY_DAYS:
            action = "PERIODIC REBALANCE (4-Day Rule)"
            state['last_rebalance'] = now_ts
            
    # Update State
    state['mode'] = new_mode
    state['last_price'] = current_price
    state['last_metrics'] = metrics
    state['last_update'] = datetime.datetime.now().isoformat()
    
    # History
    history_entry = {
        'ts': float(now_ts), 
        'nav': float(state['nav']), 
        'mode': str(new_mode), 
        'price': float(current_price),
        'macro_rsi': float(metrics['macro_rsi']),
        'xgb': float(metrics['xgb_prob']),
        'action': action
    }
    state['history'] = state.get('history', []) + [history_entry]
    if len(state['history']) > 500: state['history'] = state['history'][-500:]

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"--- DUAL-ENGINE PULSE COMPLETE ---")
    print(f"Price: {current_price:.2f} | Macro RSI: {metrics['macro_rsi']:.1f} | Mode: {new_mode}")
    print(f"Action: {action}")
    print(f"Signals: GA={metrics['ga']}, XGB={metrics['xgb_prob']:.3f}, BullRegime={metrics['is_bull']}, BearRegime={metrics['is_bear']}")

if __name__ == "__main__":
    run_pulse_15m()
