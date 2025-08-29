import numpy as np
import pandas as pd

# -------- Heikin Ashi --------
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    out["ha_open"], out["ha_close"] = ha_open, ha_close
    out["ha_high"], out["ha_low"]   = ha_high, ha_low
    out["ha_color"] = np.where(out["ha_close"]>=out["ha_open"], 1, -1) # 1=green
    return out

def ha_run_length(series: pd.Series) -> int:
    """длина последней одноцветной серии"""
    if series.empty: return 0
    last = series.iloc[-1]
    cnt=1
    for v in series.iloc[-2::-1]:
        if v==last: cnt+=1
        else: break
    return cnt if last!=0 else 0

# -------- MACD (hist) --------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    sig  = ema(macd, signal)
    return macd - sig

def streak_len_sign(series: pd.Series) -> (int, int):
    """длина последней серии по знаку (+1 / -1), (длина, знак)"""
    if series.empty: return 0,0
    sign = np.sign(series.iloc[-1])
    if sign==0: return 0,0
    cnt=1
    for v in series.iloc[-2::-1]:
        if np.sign(v)==sign and np.sign(v)!=0: cnt+=1
        else: break
    return cnt, int(sign)

def slowdown_last_n(series: pd.Series, n:int=3) -> bool:
    """последние n столбиков по модулю убывают (усталость)"""
    if len(series)<n+1: return False
    tail = series.iloc[-n:]
    return all(abs(tail.iloc[i]) <= abs(tail.iloc[i-1]) for i in range(1, n))

# -------- RSI / ATR --------
def rsi_wilder(close: pd.Series, length=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn.replace(0,np.nan))
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def atr_wilder(df: pd.DataFrame, length=14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr
