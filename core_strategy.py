# core_strategy.py
# CapinteL-Q — ядро стратегии (Polygon.io).
# Не раскрывает формулы в user-тексте; диагностика — в meta.

from __future__ import annotations
import os, math, datetime as dt
from typing import Optional, Tuple, Dict, List
import requests
import pandas as pd
import numpy as np

# ===== Универсальный результат (и dict, и объект) ===========================
class Decision(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __getattr__(self, item):
        return self.get(item)

# ===== Polygon ===============================================================
POLY_KEY = os.getenv("POLYGON_API_KEY", "").strip()

def _poly(url: str, params: dict) -> dict:
    if not POLY_KEY:
        raise RuntimeError("POLYGON_API_KEY не задан")
    p = dict(params or {})
    p["apiKey"] = POLY_KEY
    r = requests.get(url, params=p, timeout=25)
    r.raise_for_status()
    return r.json()

def fetch_daily(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    j = _poly(url, {"adjusted": "true", "limit": 50000})
    res = j.get("results") or []
    if not res:
        raise RuntimeError(f"Нет дневных данных по {ticker}")
    rows = []
    for it in res:
        rows.append({
            "date": dt.datetime.utcfromtimestamp(int(it["t"])//1000).date(),
            "open": float(it["o"]), "high": float(it["h"]),
            "low": float(it["l"]), "close": float(it["c"]),
            "volume": float(it.get("v", 0.0))
        })
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    return df

# ===== Индикаторы ============================================================
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = ha_open
    ha["ha_up"] = ha["ha_close"] > ha["ha_open"]
    return ha

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

def rsi_wilder(close: pd.Series, period=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1/period, adjust=False).mean()
    ad = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = au/(ad+1e-12)
    return 100 - (100/(1+rs))

def atr_wilder(df: pd.DataFrame, period=14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-prev).abs(),
        (df["low"]-prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ===== Пивоты (Fibo) на ПРОШЛОМ периоде =====================================
def fib_pivot(h, l, c) -> Dict[str, float]:
    p = (h + l + c) / 3.0
    r1 = p + 0.382*(h-l); r2 = p + 0.618*(h-l); r3 = p + 1.0*(h-l)
    s1 = p - 0.382*(h-l); s2 = p - 0.618*(h-l); s3 = p - 1.0*(h-l)
    return {"P":p,"R1":r1,"R2":r2,"R3":r3,"S1":s1,"S2":s2,"S3":s3}

def prev_period_ohlc(df: pd.DataFrame, horizon: str) -> Dict[str, float]:
    if horizon == "short":
        # прошлая календарная неделя (пн-вс)
        last = df.index.max()
        monday = last - dt.timedelta(days=last.weekday())
        prev_end = monday - dt.timedelta(days=1)
        prev_start = prev_end - dt.timedelta(days=6)
        w = df.loc[(df.index>=prev_start) & (df.index<=prev_end)]
    elif horizon == "mid":
        last = df.index.max()
        first_this = last.replace(day=1)
        prev_end = first_this - dt.timedelta(days=1)
        prev_start = prev_end.replace(day=1)
        w = df.loc[(df.index>=prev_start) & (df.index<=prev_end)]
    else:
        last = df.index.max()
        prev_year = (last.replace(month=1, day=1) - dt.timedelta(days=1)).year
        w = df.loc[df.index.year == prev_year]
    if w.empty:  # запасной вариант
        w = df.tail(20)
    H = float(w["high"].max()); L = float(w["low"].min())
    O = float(w["open"].iloc[0]); C = float(w["close"].iloc[-1])
    return {"H":H,"L":L,"O":O,"C":C}

# ===== ЯДРО ==================================================================
def analyze_ticker(ticker: str, horizon: str) -> Decision:
    horizon = horizon.lower().strip()
    if horizon not in ("short","mid","long"):
        raise ValueError("horizon должен быть: short | mid | long")

    end = dt.date.today()
    lookback = {"short": 180, "mid": 540, "long": 1500}[horizon]
    start = end - dt.timedelta(days=lookback)
    df = fetch_daily(ticker, start, end)

    price = float(df["close"].iloc[-1])
    ha = heikin_ashi(df)
    macdh = macd_hist(df["close"])
    rsi = rsi_wilder(df["close"])
    atr = atr_wilder(df)
    vol = float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else max(1.0, 0.01*price)

    # длина последней одноцветной серии HA
    last_up = bool(ha["ha_up"].iloc[-1])
    ha_len = 1
    for i in range(2, min(80, len(ha))+1):
        if bool(ha["ha_up"].iloc[-i]) == last_up:
            ha_len += 1
        else:
            break

    # стрик MACD hist (по знаку) + «усталость»
    sign = 1 if macdh.iloc[-1] >= 0 else -1
    macd_streak = 1
    for i in range(2, min(80, len(macdh))+1):
        s = 1 if macdh.iloc[-i] >= 0 else -1
        if s == sign:
            macd_streak += 1
        else:
            break
    slowdown = False
    if len(macdh) >= 4:
        a = abs(macdh.iloc[-1]) <= abs(macdh.iloc[-2]) + 1e-12
        b = abs(macdh.iloc[-2]) <= abs(macdh.iloc[-3]) + 1e-12
        c = abs(macdh.iloc[-3]) <= abs(macdh.iloc[-4]) + 1e-12
        slowdown = a and b and c

    # пивоты прошлого периода
    base = prev_period_ohlc(df, horizon)
    piv = fib_pivot(base["H"], base["L"], base["C"])

    # пороги по горизонту
    if horizon == "short":
        tol = 0.007; need_ha = 4; need_macd = 4
        k_stop = 0.8; k_t1, k_t2 = 0.6, 1.1
        human_wait = "Здесь решает качество точки. Ждём аккуратного отката к опоре или понятного отказа от продолжения."
    elif horizon == "mid":
        tol = 0.010; need_ha = 5; need_macd = 6
        k_stop = 1.0; k_t1, k_t2 = 0.8, 1.4
        human_wait = "Сейчас разумнее переждать. На текущих уровнях преимущество не на нашей стороне — ждём перезагрузку и ясный сигнал."
    else:
        tol = 0.012; need_ha = 6; need_macd = 8
        k_stop = 1.3; k_t1, k_t2 = 1.0, 1.8
        human_wait = "Долгосрок: работаем от зон прошлого года. Ждём перезагрузку к опорам и признаки восстановления."

    # края диапазона (без явного вывода в текст)
    near_R2 = price >= piv["R2"]*(1 - tol)
    near_R3 = price >= piv["R3"]*(1 - tol)
    near_S2 = price <= piv["S2"]*(1 + tol)
    near_S3 = price <= piv["S3"]*(1 + tol)

    long_up = last_up and (ha_len >= need_ha)
    long_dn = (not last_up) and (ha_len >= need_ha)
    macd_hot_up = (sign>0 and macd_streak>=need_macd) or (sign>0 and slowdown)
    macd_hot_dn = (sign<0 and macd_streak>=need_macd) or (sign<0 and slowdown)

    overheat = (near_R2 or near_R3) and long_up and macd_hot_up
    oversold = (near_S2 or near_S3) and long_dn and macd_hot_dn

    # базовая рекомендация
    stance = "WAIT"
    entry = None; t1=t2=stop=None
    comment = human_wait
    conf = 35  # стартовая уверенность

    if overheat:
        stance = "SHORT"
        if near_R3:
            lo, hi = piv["R3"]*0.996, piv["R3"]*1.004
            t1, t2 = piv["R2"], piv["P"]
            stop = hi + k_stop*vol
        else:
            lo, hi = piv["R2"]*0.996, piv["R2"]*1.004
            t1, t2 = max(piv["P"], piv["S1"]), piv["S1"]
            stop = hi + k_stop*vol
        entry = (float(lo), float(hi))
        comment = "Сценарий: шорт от верхней кромки диапазона. Работаем аккуратно; при сломе выходим без промедления."
        conf = 70
        if near_R3: conf += 10
        if slowdown: conf += 5

    elif oversold:
        stance = "BUY"
        if near_S3:
            lo, hi = piv["S3"]*0.996, piv["S3"]*1.004
            t1, t2 = piv["S2"], piv["P"]
            stop = lo - k_stop*vol
        else:
            lo, hi = piv["S2"]*0.996, piv["S2"]*1.004
            t1, t2 = min(piv["P"], piv["R1"]), piv["R1"]
            stop = lo - k_stop*vol
        entry = (float(lo), float(hi))
        comment = "Сценарий: откуп от нижней кромки диапазона. Берём спокойно; при сломе уровня выходим и ждём новую формацию."
        conf = 70
        if near_S3: conf += 10
        if slowdown: conf += 5

    else:
        # fallback: предлагаем «рабочую» зону
        base_ref = min(piv["P"], piv["S1"])
        entry = (base_ref*0.996, base_ref*1.004)
        if horizon == "short":
            t1 = price + (k_t1*vol if price < base_ref else -k_t1*vol)
            t2 = price + (k_t2*vol if price < base_ref else -k_t2*vol)
            stop = (entry[0] if price < base_ref else entry[1]) - np.sign(t2 - price)*k_stop*vol
        elif horizon == "mid":
            t1, t2 = max(piv["P"], piv["R1"]), piv["R2"]
            stop = min(piv["S1"], piv["S2"]) - 0.3*k_stop*vol
        else:
            t1, t2 = piv["P"], piv["R1"]
            stop = min(piv["S1"], piv["S2"]) - 0.5*k_stop*vol

    # защита от нелепостей: стоп «не по сторону»
    if entry:
        lo, hi = entry
        if stance == "BUY" and stop is not None:
            stop = min(stop, lo - 1e-6)
        if stance == "SHORT" and stop is not None:
            stop = max(stop, hi + 1e-6)

    # простая мульти-ТФ проверка (старший ТФ подтверждает — +уверенность)
    try:
        higher = {"short":"mid", "mid":"long"}.get(horizon)
        if higher:
            base2 = prev_period_ohlc(df, higher)
            piv2 = fib_pivot(base2["H"], base2["L"], base2["C"])
            if stance == "BUY" and price <= piv2["P"]:  # младший хочет вверх, а на старшем мы не выше «середины»
                conf += 10
            if stance == "SHORT" and price >= piv2["P"]:
                conf += 10
    except Exception:
        pass

    conf = int(max(10, min(conf, 95)))

    meta = dict(
        price=float(price), horizon=horizon,
        pivots=piv, prev_period=base,
        ha_last_up=bool(last_up), ha_series=int(ha_len),
        macd_sign=int(sign), macd_streak=int(macd_streak),
        macd_slowdown=bool(slowdown),
        near=dict(R2=bool(near_R2), R3=bool(near_R3), S2=bool(near_S2), S3=bool(near_S3)),
        atr=float(vol), tol=float(tol),
        thresholds=dict(ha=need_ha, macd=need_macd)
    )

    return Decision(
        ticker=ticker, horizon=horizon, price=float(price),
        stance=stance, confidence=conf,
        entry=entry, target1=float(t1) if t1 is not None else None,
        target2=float(t2) if t2 is not None else None,
        stop=float(stop) if stop is not None else None,
        comment=comment, meta=meta
    )

# ===== Простейший бэктест (каркас, чтобы UI работал) ========================
def run_backtest(ticker: str, horizon: str, years: int = 3,
                 start_capital: float = 100_000.0, fee_bps: float = 5.0):
    end = dt.date.today()
    start = end - dt.timedelta(days=years*365 + 30)
    df = fetch_daily(ticker, start, end)

    cash = start_capital
    pos = 0  # 1 long, -1 short
    entry_px = None
    fee = fee_bps/10000.0
    equity = []
    trades: List[dict] = []

    for d, row in df.iterrows():
        price = float(row["close"])
        dec = analyze_ticker(ticker, horizon)
        equity.append((d, cash))

        # выход по целям/стопу
        if pos != 0 and entry_px is not None:
            if pos > 0:
                if dec.stop and price <= dec.stop:
                    cash *= (1 + (dec.stop/entry_px - 1) - fee); pos = 0
                    trades.append(dict(date=d, action="STOP", px=dec.stop))
                elif dec.target1 and price >= dec.target1:
                    cash *= (1 + (dec.target1/entry_px - 1) - fee); pos = 0
                    trades.append(dict(date=d, action="T1", px=dec.target1))
            else:
                if dec.stop and price >= dec.stop:
                    cash *= (1 + (entry_px/dec.stop - 1) - fee); pos = 0
                    trades.append(dict(date=d, action="STOP", px=dec.stop))
                elif dec.target1 and price <= dec.target1:
                    cash *= (1 + (entry_px/dec.target1 - 1) - fee); pos = 0
                    trades.append(dict(date=d, action="T1", px=dec.target1))

        # вход по сигналу (одна позиция за раз)
        if pos == 0 and dec.entry:
            lo, hi = dec.entry
            entry_px = (lo + hi) / 2
            if dec.stance == "BUY":
                pos = 1; cash *= (1 - fee); trades.append(dict(date=d, action="BUY", px=entry_px))
            elif dec.stance == "SHORT":
                pos = -1; cash *= (1 - fee); trades.append(dict(date=d, action="SHORT", px=entry_px))

    eq = pd.DataFrame(equity, columns=["date","equity"]).set_index("date")
    summary = dict(start=start_capital, end=float(eq["equity"].iloc[-1]),
                   ret_pct=(float(eq["equity"].iloc[-1])/start_capital - 1)*100)
    return {"summary": summary, "equity": eq, "trades": pd.DataFrame(trades)}
