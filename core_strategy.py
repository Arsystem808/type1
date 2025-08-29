# core_strategy.py
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
from dateutil.relativedelta import relativedelta

from polygon_client import fetch_daily

# === Data class ===
@dataclass
class Decision:
    stance: str            # 'BUY' | 'SELL' | 'WAIT'
    entry: Optional[Tuple[float,float]]  # (low, high) entry zone
    target1: Optional[float]
    target2: Optional[float]
    stop: Optional[float]
    horizon: str           # 'short'|'mid'|'long'
    price: float
    notes: str = ""
    debug: Dict[str, Any] = None

    def to_dict(self):
        d = asdict(self)
        return d

# === Heikin Ashi ===
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [ (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0 ]
    for i in range(1, len(df)):
        ha_open.append( (ha_open[i-1] + ha_close.iloc[i-1]) / 2.0 )
    out["ha_open"] = pd.Series(ha_open, index=df.index)
    out["ha_close"] = ha_close
    out["ha_high"] = out[["high","ha_open","ha_close"]].max(axis=1)
    out["ha_low"] = out[["low","ha_open","ha_close"]].min(axis=1)
    out["ha_color"] = np.where(out["ha_close"] >= out["ha_open"], 1, -1)  # 1=green, -1=red
    return out

# === MACD histogram ===
def macd_hist(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return hist

# === RSI (Wilder 14) ===
def rsi(series: pd.Series, length=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === ATR (Wilder 14) ===
def atr(df: pd.DataFrame, length=14) -> pd.Series:
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

# === Fibonacci pivots based on prior period H/L/C ===
def fib_pivots_from(df: pd.DataFrame, kind:str) -> Dict[str,float]:
    """
    kind: 'weekly' | 'monthly' | 'yearly'
    We take the *previous* period's OHLC (H,L,C) to compute pivots.
    """
    if kind == "weekly":
        # Use last full ISO week (Mon-Sun). Find prior week range.
        last = df.index[-1]
        prior_end = (last - pd.Timedelta(days=last.weekday()+1)).normalize()  # previous Sunday (approx)
        prior_start = prior_end - pd.Timedelta(days=6)
    elif kind == "monthly":
        last = df.index[-1]
        prior_end = (last.replace(day=1) - pd.Timedelta(days=1)).normalize()
        prior_start = prior_end.replace(day=1)
    elif kind == "yearly":
        last = df.index[-1]
        prior_end = last.replace(month=1, day=1) - pd.Timedelta(days=1)
        prior_start = prior_end.replace(month=1, day=1)
    else:
        raise ValueError("unknown pivot kind")
    seg = df.loc[(df.index>=prior_start) & (df.index<=prior_end)]
    if seg.empty:
        seg = df.iloc[-20:]  # fallback
    H = float(seg["high"].max())
    L = float(seg["low"].min())
    C = float(seg["close"].iloc[-1])
    P = (H+L+C)/3.0
    D = H - L
    # Fibonacci pivots
    R1 = P + 0.382*D; S1 = P - 0.382*D
    R2 = P + 0.618*D; S2 = P - 0.618*D
    R3 = P + 1.000*D; S3 = P - 1.000*D
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3, "H":H, "L":L, "C":C}

def _last_streak(sign_series: pd.Series) -> int:
    # counts last consecutive non-zero same-sign values
    s = sign_series.replace(0, np.nan).dropna()
    if s.empty:
        return 0
    last_sign = np.sign(s.iloc[-1])
    cnt = 0
    for v in reversed(sign_series.values):
        if v == 0: 
            break
        if np.sign(v) == last_sign:
            cnt += 1
        else:
            break
    return cnt

def _is_slowing(hist: pd.Series, lookback=4) -> bool:
    h = hist.dropna().tail(lookback)
    if len(h) < lookback: 
        return False
    # slowing = shrinking absolute bars for 3 steps
    diffs = np.diff(np.abs(h.values))
    return np.all(diffs <= 0)

def analyze_ticker(ticker: str, horizon: str="mid") -> Decision:
    """
    horizon: 'short' (1-5d, weekly pivots), 'mid' (1-4w, monthly pivots), 'long' (1-6m, yearly pivots)
    Returns Decision.
    """
    ticker = ticker.upper().strip()
    df = fetch_daily(ticker, days=800)
    price = float(df["close"].iloc[-1])

    # Indicators
    df_ha = heikin_ashi(df)
    hist = macd_hist(df)
    hist_sign = np.sign(hist)

    # Streak thresholds
    HA_STREAK = {"short":4, "mid":5, "long":6}[horizon]
    MACD_STREAK = {"short":4, "mid":6, "long":8}[horizon]

    ha_streak = _last_streak(pd.Series(np.where(df_ha["ha_color"]>0,1,-1), index=df.index))
    macd_streak = _last_streak(hist_sign)

    slowing = _is_slowing(hist, lookback=4)

    # RSI/ATR
    rsi_series = rsi(df["close"])
    atr_series = atr(df)
    last_rsi = float(rsi_series.iloc[-1])
    last_atr = float(atr_series.iloc[-1])

    # Pivots by horizon
    pivot_kind = {"short":"weekly","mid":"monthly","long":"yearly"}[horizon]
    piv = fib_pivots_from(df, pivot_kind)

    # Tolerances
    tol = {"short":0.006, "mid":0.009, "long":0.012}[horizon]  # ~0.6%/0.9%/1.2%
    near = lambda level: abs(price-level) <= level*tol
    above = lambda level: price >= level*(1- tol)
    below = lambda level: price <= level*(1+ tol)

    # Composite signals
    debug = dict(price=price, rsi=last_rsi, atr=last_atr, ha_streak=ha_streak, macd_streak=macd_streak,
                 slowing=slowing, piv=piv, horizon=horizon)

    stance = "WAIT"; entry=None; t1=None; t2=None; stop=None; notes=[]

    # Overheated at roof
    roof = (above(piv["R2"]) and (ha_streak>=HA_STREAK) and (macd_streak>=MACD_STREAK or slowing) and (last_rsi>55))
    # Oversold at floor
    floor = (below(piv["S2"]) and (ha_streak>=HA_STREAK) and (macd_streak>=MACD_STREAK or slowing) and (last_rsi<45))

    if roof:
        # Default WAIT, aggressive SELL option
        stance = "WAIT"
        # Aggressive short entry:
        if above(piv["R3"]* (1- tol)):
            entry = (piv["R3"]* (1-0.002), piv["R3"]* (1+0.002))
            t1, t2 = piv["R2"], piv["P"]
            stop = piv["R3"] + 1.2*last_atr
            notes.append("Перегрев у крыши; возможен аккуратный шорт от R3 → цели R2, затем ~P.")
        elif above(piv["R2"]):
            entry = (piv["R2"]* (1-0.003), piv["R2"]* (1+0.003))
            t1, t2 = (piv["P"]+piv["S1"])/2, piv["S1"]
            stop = piv["R2"] + 1.0*last_atr
            notes.append("Перегрев у крыши; вариант шорта от R2 → к P/S1.")

    elif floor:
        stance = "BUY"
        if below(piv["S3"]*(1+tol)):
            entry = (piv["S3"]*(1-0.002), piv["S3"]*(1+0.002))
            t1, t2 = piv["S2"], piv["P"]
            stop = piv["S3"] - 1.2*last_atr
            notes.append("Перепроданность у дна; покупки от S3 → цели S2, затем ~P.")
        else:
            entry = (piv["S2"]*(1-0.003), piv["S2"]*(1+0.003))
            t1, t2 = piv["P"], piv["R1"]
            stop = piv["S2"] - 1.0*last_atr
            notes.append("Перепроданность у дна; покупки от S2 → к P/R1.")

    else:
        # Midline logic: prefer buying near P/S1, avoid buying at roof
        if price < piv["P"] and last_rsi<60:
            stance = "BUY"
            entry = (piv["P"]*0.995, piv["P"]*1.005)
            t1, t2 = piv["R1"], piv["R2"]
            stop = piv["S1"] - 0.8*last_atr
            notes.append("Рабочая покупка после перезагрузки к опорной зоне.")
        elif price > piv["P"] and last_rsi>40 and above(piv["R2"]) and (ha_streak>=HA_STREAK):
            stance = "SELL"
            entry = (piv["R2"]*0.997, piv["R2"]*1.003)
            t1, t2 = piv["P"], piv["S1"]
            stop = piv["R2"] + 1.0*last_atr
            notes.append("Под потолком предпочтительна защита/короткая игра к середине.")
        else:
            stance = "WAIT"
            entry=None; t1=None; t2=None; stop=None
            notes.append("На текущих ценах преимущество не на нашей стороне — ждём лучшую формацию.")

    dec = Decision(stance=stance, entry=entry, target1=t1, target2=t2, stop=stop,
                   horizon=horizon, price=price, notes=" ".join(notes), debug=debug)
    return dec
