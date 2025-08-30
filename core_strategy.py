# core_strategy.py
# Capintel-Q: анализ тикера без раскрытия формул в основном тексте.
# Python 3.11+. Нужен POLYGON_API_KEY в окружении.

from __future__ import annotations
import os, math, datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import requests
import pandas as pd
import numpy as np

# ==========================
# Dataclass результата
# ==========================
@dataclass
class Decision:
    ticker: str
    horizon: str               # 'short' | 'mid' | 'long'
    price: float
    stance: str                # 'BUY' | 'SHORT' | 'WAIT'
    entry: Optional[Tuple[float, float]]  # (low, high) входная зона
    target1: Optional[float]
    target2: Optional[float]
    stop: Optional[float]
    comment: str               # human-style текст
    meta: Dict                 # внутренняя диагностика (уровни, флаги)

# ==========================
# Сервис: Polygon.io
# ==========================
POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def _poly(url: str, params: dict) -> dict:
    if not POLY_KEY:
        raise RuntimeError("POLYGON_API_KEY не задан.")
    p = dict(params or {})
    p["apiKey"] = POLY_KEY
    r = requests.get(url, params=p, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_daily(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Дневные свечи [start, end] включительно."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    data = _poly(url, {"adjusted": "true", "limit": 50000})
    if data.get("results") is None:
        raise RuntimeError(f"Нет данных по {ticker}")
    rows = []
    for it in data["results"]:
        rows.append({
            "date": dt.datetime.utcfromtimestamp(int(it["t"])//1000).date(),
            "open": float(it["o"]),
            "high": float(it["h"]),
            "low": float(it["l"]),
            "close": float(it["c"]),
            "volume": float(it.get("v", 0.0)),
        })
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    return df

def last_price(ticker: str) -> float:
    url = f"https://api.polygon.io/v2/last/trade/{ticker}"
    try:
        j = _poly(url, {})
        return float(j["results"]["price"])
    except Exception:
        # fallback — по последней дневной
        end = dt.date.today()
        start = end - dt.timedelta(days=7)
        df = fetch_daily(ticker, start, end)
        return float(df["close"].iloc[-1])

# ==========================
# Техника
# ==========================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [ (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0 ]
    for i in range(1, len(df)):
        ha_open.append( (ha_open[i-1] + ha["ha_close"].iloc[i-1]) / 2.0 )
    ha["ha_open"] = ha_open
    ha["ha_up"] = ha["ha_close"] > ha["ha_open"]
    return ha

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal  # гистограмма

def rsi_wilder(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr_wilder(df: pd.DataFrame, period=14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# Pivot (Fibonacci) по прошлому периоду
def fib_pivot(h: float, l: float, c: float) -> Dict[str, float]:
    p = (h + l + c) / 3.0
    r1 = p + 0.382*(h-l)
    r2 = p + 0.618*(h-l)
    r3 = p + 1.000*(h-l)
    s1 = p - 0.382*(h-l)
    s2 = p - 0.618*(h-l)
    s3 = p - 1.000*(h-l)
    return {"P":p, "R1":r1, "R2":r2, "R3":r3, "S1":s1, "S2":s2, "S3":s3}

def previous_period_ohlc(df: pd.DataFrame, horizon: str) -> Dict[str, float]:
    if horizon == "short":
        # прошлая КАЛЕНДАРНАЯ неделя (пн-вс)
        today = df.index.max()
        monday = today - dt.timedelta(days=today.weekday())
        prev_week_end = monday - dt.timedelta(days=1)
        prev_week_start = prev_week_end - dt.timedelta(days=6)
        w = df.loc[(df.index>=prev_week_start) & (df.index<=prev_week_end)]
    elif horizon == "mid":
        last = df.index.max()
        first_of_this = last.replace(day=1)
        prev_month_end = first_of_this - dt.timedelta(days=1)
        prev_month_start = prev_month_end.replace(day=1)
        w = df.loc[(df.index>=prev_month_start) & (df.index<=prev_month_end)]
    else:
        # long: прошлый год (UTC)
        last = df.index.max()
        prev_year = (last.replace(month=1, day=1) - dt.timedelta(days=1)).year
        w = df.loc[df.index.year == prev_year]
    if w.empty:
        # fallback: возьмём последние N дней
        w = df.tail(20)
    H = float(w["high"].max())
    L = float(w["low"].min())
    O = float(w["open"].iloc[0])
    C = float(w["close"].iloc[-1])
    return {"H":H, "L":L, "O":O, "C":C}

# ==========================
# Логика стратегии
# ==========================
def analyze_ticker(ticker: str, horizon: str) -> Decision:
    # 1) Данные
    end = dt.date.today()
    lookback = {"short": 120, "mid": 420, "long": 1200}[horizon]  # дней
    start = end - dt.timedelta(days=lookback)
    df = fetch_daily(ticker, start, end)

    price = float(df["close"].iloc[-1])

    # 2) Техника
    ha = heikin_ashi(df)
    macdh = macd_hist(df["close"])
    rsi = rsi_wilder(df["close"])
    atr = atr_wilder(df)

    # длина последней одноцветной серии HA
    last_up = ha["ha_up"].iloc[-1]
    series_len = 1
    for i in range(2, min(60, len(ha))+1):
        if ha["ha_up"].iloc[-i] == last_up:
            series_len += 1
        else:
            break

    # длительность знака MACD-гистограммы
    sign = 1 if macdh.iloc[-1] >= 0 else -1
    macd_streak = 1
    for i in range(2, min(60, len(macdh))+1):
        s = 1 if macdh.iloc[-i] >= 0 else -1
        if s == sign:
            macd_streak += 1
        else:
            break

    # замедление гистограммы (по модулю падает 3 бара подряд)
    slow_down = False
    if len(macdh) >= 4:
        a = np.abs(macdh.iloc[-1]) <= np.abs(macdh.iloc[-2]) + 1e-9
        b = np.abs(macdh.iloc[-2]) <= np.abs(macdh.iloc[-3]) + 1e-9
        c = np.abs(macdh.iloc[-3]) <= np.abs(macdh.iloc[-4]) + 1e-9
        slow_down = a and b and c

    # Pivot на прошлом периоде
    prev = previous_period_ohlc(df, horizon)
    piv = fib_pivot(prev["H"], prev["L"], prev["C"])

    # пороги по горизонту
    if horizon == "short":
        tol = 0.007      # 0.7%
        need_ha = 4
        need_macd = 4
        atr_k_stop = 0.8
        atr_k_t1, atr_k_t2 = 0.6, 1.1
    elif horizon == "mid":
        tol = 0.010
        need_ha = 5
        need_macd = 6
        atr_k_stop = 1.0
        atr_k_t1, atr_k_t2 = 0.8, 1.4
    else:
        tol = 0.012
        need_ha = 6
        need_macd = 8
        atr_k_stop = 1.3
        atr_k_t1, atr_k_t2 = 1.0, 1.8

    # флаги «перегрева у крыши» и «перепроданности у дна»
    near_R2 = price >= piv["R2"]*(1 - tol)
    near_R3 = price >= piv["R3"]*(1 - tol)
    near_S2 = price <= piv["S2"]*(1 + tol)
    near_S3 = price <= piv["S3"]*(1 + tol)

    long_series_up = last_up and (series_len >= need_ha)
    long_series_down = (not last_up) and (series_len >= need_ha)
    macd_hot_up = (sign > 0 and macd_streak >= need_macd) or (sign>0 and slow_down)
    macd_hot_down = (sign < 0 and macd_streak >= need_macd) or (sign<0 and slow_down)

    overheat = (near_R2 or near_R3) and long_series_up and macd_hot_up
    oversold = (near_S2 or near_S3) and long_series_down and macd_hot_down

    # ==========================
    # Решение
    # ==========================
    stance = "WAIT"
    entry = None
    t1 = t2 = stop = None
    human = ""

    vol = float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else max(1.0, 0.01*price)

    if overheat:
        stance = "SHORT"
        if near_R3:
            entry = (piv["R3"]*0.996, piv["R3"]*1.004)
            t1, t2 = piv["R2"], piv["P"]
            stop = piv["R3"] + atr_k_stop*vol
        else:
            entry = (piv["R2"]*0.996, piv["R2"]*1.004)
            t1, t2 = max(piv["P"], piv["S1"]), piv["S1"]
            stop = piv["R2"] + atr_k_stop*vol
        human = "Сценарий: шорт от «крыши». Работаем аккуратно и без героизма."

    elif oversold:
        stance = "BUY"
        if near_S3:
            entry = (piv["S3"]*0.996, piv["S3"]*1.004)
            t1, t2 = piv["S2"], piv["P"]
            stop = piv["S3"] - atr_k_stop*vol
        else:
            entry = (piv["S2"]*0.996, piv["S2"]*1.004)
            t1, t2 = min(piv["P"], piv["R1"]), piv["R1"]
            stop = piv["S2"] - atr_k_stop*vol
        human = "Сценарий: откуп от «дна». Берём без суеты, при сломе — выходим."

    else:
        # базовая логика, когда явного края нет
        # для short/mid ждём откат к опоре, для long — глубокую перезагрузку
        if horizon in ("short", "mid"):
            entry = (min(piv["P"], piv["S1"])*0.996, min(piv["P"], piv["S1"])*1.004)
            t1 = price - atr_k_t1*vol if horizon=="short" else max(piv["R1"], piv["P"])
            t2 = price - atr_k_t2*vol if horizon=="short" else piv["R2"]
            stop = price + atr_k_stop*vol if horizon=="short" else max(piv["R2"], piv["R1"]) + atr_k_stop*vol*0.5
            human = "Сейчас выгоднее подождать. На текущих уровнях явного преимущества нет — ждём откат к опоре или подтверждение разворота."
        else:
            entry = (min(piv["P"], piv["S1"])*0.99, min(piv["P"], piv["S1"])*1.01)
            t1, t2 = piv["P"], piv["R1"]
            stop = min(piv["S2"], piv["S1"]) - atr_k_stop*vol
            human = "Долгосрок: работаем от опор прошлого года. Ждём перезагрузку и признаков восстановления."

    meta = dict(
        pivot=piv, prev=prev, horizon=horizon, price=price,
        ha_last_up=bool(last_up), ha_series=series_len,
        macd_streak=int(macd_streak), macd_sign=int(sign), macd_slowdown=bool(slow_down),
        flags=dict(overheat=overheat, oversold=oversold, near_R2=near_R2, near_R3=near_R3, near_S2=near_S2, near_S3=near_S3),
        thresholds=dict(need_ha=need_ha, need_macd=need_macd, tol=tol),
        atr=float(vol),
    )

    return Decision(
        ticker=ticker, horizon=horizon, price=price,
        stance=stance, entry=entry, target1=t1, target2=t2, stop=stop,
        comment=human, meta=meta
    )

# ==========================
# (опционально) бэктест — простая обвязка
# ==========================
def run_backtest(ticker: str, horizon: str, years: int = 3, start_capital: float = 100_000.0, fee_bps: float = 5.0):
    """
    Очень простой пример бэктеста: раз в день запрашиваем решение и
    работаем «вход-выход» по целям/стопу. Это каркас, чтобы UI не падал.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=years*365 + 30)
    df = fetch_daily(ticker, start, end)
    cash = start_capital
    pos = 0.0
    equity = []
    trades: List[dict] = []
    fee = fee_bps/10000.0

    for d, row in df.iterrows():
        price = float(row["close"])
        dec = analyze_ticker(ticker, horizon=horizon)

        # закрываем, если есть позиция и выполнены t/stop
        if pos != 0:
            if pos > 0:  # long
                if dec.target1 and price >= dec.target1:
                    cash *= (1 + (dec.target1/entry_px - 1) - fee)
                    trades.append(dict(date=d, side="SELL_T1", px=dec.target1))
                    pos = 0
                elif dec.stop and price <= dec.stop:
                    cash *= (1 + (dec.stop/entry_px - 1) - fee)
                    trades.append(dict(date=d, side="STOP", px=dec.stop))
                    pos = 0
            else:        # short
                if dec.target1 and price <= dec.target1:
                    cash *= (1 + (entry_px/dec.target1 - 1) - fee)
                    trades.append(dict(date=d, side="BUY_T1", px=dec.target1))
                    pos = 0
                elif dec.stop and price >= dec.stop:
                    cash *= (1 + (entry_px/dec.stop - 1) - fee)
                    trades.append(dict(date=d, side="STOP", px=dec.stop))
                    pos = 0

        # открываем новую по сигналу
        if pos == 0 and dec.entry:
            lo, hi = dec.entry
            entry_px = (lo+hi)/2
            if dec.stance == "BUY":
                pos = 1
                cash *= (1 - fee)
                trades.append(dict(date=d, side="BUY", px=entry_px))
            elif dec.stance == "SHORT":
                pos = -1
                cash *= (1 - fee)
                trades.append(dict(date=d, side="SHORT", px=entry_px))

        equity.append((d, cash))

    eq = pd.DataFrame(equity, columns=["date","equity"]).set_index("date")
    summary = dict(
        start=start_capital,
        end=float(eq["equity"].iloc[-1]),
        ret_pct=(float(eq["equity"].iloc[-1])/start_capital - 1)*100
    )
    return {"summary": summary, "equity": eq, "trades": pd.DataFrame(trades)}

