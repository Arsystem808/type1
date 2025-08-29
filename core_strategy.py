# core_strategy.py
# CapinteL-Q – ядро сигналов (Polygon-only, без утечек формул в текст)
# Python 3.11+

from __future__ import annotations
import os
import math
import time
import datetime as dt
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any, Literal
import requests

# --------------------------- Константы/среда ---------------------------

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
POLYGON_BASE = "https://api.polygon.io"

# для крипто тикеров принято префиксовать "X:" (например, X:BTCUSD)
# для акций/ETF — как есть: QQQ, AAPL и т.п.

# --------------------------- Утилиты данных ---------------------------

def _poly_agg_day(ticker: str, start: str, end: str, adjusted: bool = True, limit: int = 5000):
    """Сырые дневные свечи Polygon aggregates v2."""
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "limit": limit,
        "apiKey": POLYGON_API_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    results = j.get("results", []) or []
    out = []
    for x in results:
        out.append({
            "ts": int(x["t"]) // 1000,
            "o": float(x["o"]),
            "h": float(x["h"]),
            "l": float(x["l"]),
            "c": float(x["c"]),
            "v": float(x.get("v", 0)),
        })
    return out

def fetch_daily(ticker: str, lookback_days: int = 900) -> list[dict]:
    """Дневные свечи за lookback_days (с запасом). Возвращает список dict в хронологическом порядке."""
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY отсутствует в окружении")
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days + 7)
    data = _poly_agg_day(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    # сортируем по времени на всякий
    data.sort(key=lambda x: x["ts"])
    return data

def last_close(data: list[dict]) -> float:
    return float(data[-1]["c"])

# --------------------------- Индикаторные расчёты ---------------------------

def heikin_ashi(candles: list[dict]) -> list[dict]:
    """Строим HA OHLC (по классике)."""
    ha = []
    ha_o = (candles[0]["o"] + candles[0]["c"]) / 2.0
    ha_c = (candles[0]["o"] + candles[0]["h"] + candles[0]["l"] + candles[0]["c"]) / 4.0
    ha.append({"o": ha_o, "h": candles[0]["h"], "l": candles[0]["l"], "c": ha_c})
    for i in range(1, len(candles)):
        o = candles[i]["o"]; h = candles[i]["h"]; l = candles[i]["l"]; c = candles[i]["c"]
        ha_c = (o + h + l + c) / 4.0
        ha_o = (ha[-1]["o"] + ha[-1]["c"]) / 2.0
        ha_h = max(h, ha_o, ha_c)
        ha_l = min(l, ha_o, ha_c)
        ha.append({"o": ha_o, "h": ha_h, "l": ha_l, "c": ha_c})
    return ha

def macd_hist(closes: list[float], fast=12, slow=26, signal=9) -> list[float]:
    def ema(vals, period):
        k = 2 / (period + 1)
        e = []
        prev = sum(vals[:period]) / period
        e.append(prev)
        for v in vals[period:]:
            prev = v * k + prev * (1 - k)
            e.append(prev)
        # выравниваем длину (сдвиг) – добавим хвост None в начале
        pad = [None] * (len(vals) - len(e))
        return pad + e
    if len(closes) < slow + signal + 5:
        return [0.0]*len(closes)
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd = [ (a - b) if (a is not None and b is not None) else None
             for a,b in zip(ema_fast, ema_slow)]
    # уберём None в начале
    macd_clean = [x for x in macd if x is not None]
    sig = ema(macd_clean, signal)
    sig_full = [None]*(len(macd) - len(sig)) + sig
    hist = []
    for m, s in zip(macd, sig_full):
        hist.append(0.0 if (m is None or s is None) else (m - s))
    return hist

def rsi_wilder(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) < period + 2:
        return [50.0]*len(closes)
    gains = [0.0]; losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    rsis = [50.0]*period
    for i in range(period+1, len(closes)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = float('inf') if avg_loss == 0 else (avg_gain/avg_loss)
        rsi = 100 - (100 / (1 + rs))
        rsis.append(rsi)
    # выравниваем
    rsis = [50.0]*(len(closes)-len(rsis)) + rsis
    return rsis

def atr_wilder(candles: list[dict], period: int = 14) -> list[float]:
    if len(candles) < period + 2:
        return [0.0]*len(candles)
    trs = [0.0]
    for i in range(1, len(candles)):
        h = candles[i]["h"]; l = candles[i]["l"]; pc = candles[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atrs = []
    atr = sum(trs[1:period+1]) / period
    atrs = [0.0]*period + [atr]
    for i in range(period+1, len(trs)):
        atr = (atr*(period-1) + trs[i]) / period
        atrs.append(atr)
    atrs = [0.0]*(len(candles)-len(atrs)) + atrs
    return atrs

# --------------------------- Пивоты (Fibonacci) ---------------------------

def _pivot_fib(h: float, l: float, c: float) -> Dict[str, float]:
    P = (h + l + c) / 3.0
    R1 = P + 0.382*(h - l); S1 = P - 0.382*(h - l)
    R2 = P + 0.618*(h - l); S2 = P - 0.618*(h - l)
    R3 = P + 1.000*(h - l); S3 = P - 1.000*(h - l)
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def _period_bounds(ts: int, period: Literal["W","M","Y"]) -> Tuple[int,int]:
    d = dt.datetime.utcfromtimestamp(ts).date()
    if period == "W":
        # берем ПРЕДЫДУЩУЮ неделю (пн-вс)
        wd = d.weekday()
        # конец прошлой недели = понедельник текущей 00:00
        end = dt.datetime.combine(d - dt.timedelta(days=wd), dt.time())
        start = end - dt.timedelta(days=7)
    elif period == "M":
        first = dt.datetime(d.year, d.month, 1)
        end = first
        prev_month_last_day = first - dt.timedelta(days=1)
        start = dt.datetime(prev_month_last_day.year, prev_month_last_day.month, 1)
    else:  # "Y"
        end = dt.datetime(d.year, 1, 1)
        start = dt.datetime(d.year-1, 1, 1)
    return (int(start.timestamp()), int(end.timestamp()))

def pivots_from_previous_period(candles: list[dict], period: Literal["W","M","Y"]) -> Dict[str,float]:
    if not candles:
        return {"P":0,"R1":0,"R2":0,"R3":0,"S1":0,"S2":0,"S3":0}
    # берём границы пред. периода относительно последней свечи
    ts_last = candles[-1]["ts"]
    start, end = _period_bounds(ts_last, period)
    # фильтруем свечи в [start, end)
    block = [x for x in candles if start <= x["ts"] < end]
    if not block:
        # подстрахуемся: если пусто, расширим окно ещё на 7 дней назад
        start -= 7*24*3600
        block = [x for x in candles if start <= x["ts"] < end]
    h = max(x["h"] for x in block) if block else candles[-2]["h"]
    l = min(x["l"] for x in block) if block else candles[-2]["l"]
    c = block[-1]["c"] if block else candles[-2]["c"]
    return _pivot_fib(h,l,c)

# --------------------------- «Интуитивные» фильтры ---------------------------

def streak(values: list[float]) -> int:
    """Длина последней однонаправленной серии для HA_close (>=0 – зелёная, <0 – красная) или MACD hist (>0/<0)."""
    if not values:
        return 0
    last_sign = 1 if values[-1] >= 0 else -1
    cnt = 0
    for v in reversed(values):
        if (v >= 0 and last_sign>0) or (v < 0 and last_sign<0):
            cnt += 1
        else:
            break
    return cnt * last_sign  # знак = цвет

# --------------------------- Логика сигналов ---------------------------

@dataclass
class Output:
    stance: str                                  # BUY / SHORT / WAIT
    entry: Optional[Tuple[float,float]] = None   # (low, high) зоны
    target1: Optional[float] = None
    target2: Optional[float] = None
    stop: Optional[float] = None
    alt: Optional[str] = None
    comment: Optional[str] = None
    meta: Dict[str,Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str,Any]:
        return asdict(self)

def _tolerances(hz: Literal["short","mid","long"]) -> Tuple[float,float]:
    if hz == "short":  # 1–5 дней
        return (0.005, 0.008)
    if hz == "mid":    # 1–4 недели
        return (0.008, 0.010)
    return (0.010, 0.012)  # long 1–6 мес

def _horizon_to_period(hz: Literal["short","mid","long"]) -> Literal["W","M","Y"]:
    return {"short":"W", "mid":"M", "long":"Y"}[hz]

def _atr_mult(hz: Literal["short","mid","long"]) -> Tuple[float,float,float]:
    # stop_mult, tgt1_mult, tgt2_mult (в ATR)
    if hz == "short":
        return (0.8, 0.6, 1.1)
    if hz == "mid":
        return (1.0, 0.8, 1.6)  # цели всё же к опорам, множители – запас
    return (1.3, 1.2, 2.0)

def compose_decision(ticker: str, hz: Literal["short","mid","long"], candles: list[dict]) -> Output:
    price = candles[-1]["c"]
    closes = [x["c"] for x in candles]
    ha = heikin_ashi(candles)
    ha_close = [x["c"] for x in ha]
    macdh = macd_hist(closes)
    rsi = rsi_wilder(closes)
    atr = atr_wilder(candles)

    period = _horizon_to_period(hz)
    piv = pivots_from_previous_period(candles, period)
    tol_r2, tol_r3 = _tolerances(hz)
    rng = piv["R2"] - piv["P"]
    roof_r2_low  = piv["R2"] * (1 - tol_r2)
    roof_r3_low  = piv["R3"] * (1 - tol_r3)
    floor_s2_high = piv["S2"] * (1 + tol_r2)
    floor_s3_high = piv["S3"] * (1 + tol_r3)

    # стрики
    ha_st = streak([1.0 if x>=ha_close[-1] else -1.0 for x in ha_close[-5:]])  # грубо, но нам нужна только длина знака последнего
    macd_st = streak(macdh[-20:])

    # Пороговые длины по ТЗ
    need_ha = {"short":4, "mid":5, "long":6}[hz]
    need_mh = {"short":4, "mid":6, "long":8}[hz]

    # «Перегрев у крыши»
    over_roof = (
        (price >= roof_r2_low) and
        (abs(ha_st) >= need_ha and ha_close[-1] >= ha_close[-2]) and
        (abs(macd_st) >= need_mh and macdh[-1] > 0)
    )
    # «Перепроданность у дна»
    under_floor = (
        (price <= floor_s2_high) and
        (abs(ha_st) >= need_ha and ha_close[-1] <= ha_close[-2]) and
        (abs(macd_st) >= need_mh and macdh[-1] < 0)
    )

    stop_mult, t1_mult, t2_mult = _atr_mult(hz)
    last_atr = max(atr[-1], 1e-8)

    meta = {
        "price": round(price, 2),
        "piv": {k: round(v, 2) for k,v in piv.items()},
        "atr": round(last_atr, 3),
        "hz": hz,
    }

    if over_roof:
        # База по ТЗ: WAIT, но разрешаем агрессивный SHORT (опционально)
        # Дадим явный шорт с зоной "чуть выше/возле R2..R3", чтобы не шортить середину
        zone_lo = max(piv["R2"] * (1 - 0.002), price)  # не ниже текущей
        zone_hi = piv["R3"] * (1 + 0.003)
        tgt1 = piv["R2"] if price >= piv["R3"] else max(piv["P"], piv["R1"] - 0.2*rng)
        tgt2 = piv["P"] if price >= piv["R3"] else max(piv["S1"], piv["P"] - 0.5*rng)
        stop = (piv["R3"] if price >= piv["R3"] else piv["R2"]) + max(0.5*last_atr, 0.003*price)
        return Output(
            stance="SHORT",
            entry=(round(zone_lo,2), round(zone_hi,2)),
            target1=round(tgt1,2),
            target2=round(tgt2,2),
            stop=round(stop,2),
            alt="WAIT — ждём перезагрузку к P→S1",
            comment="Перегрев у крыши; играем от ослабления.",
            meta=meta
        )

    if under_floor:
        # Зеркально: LONG от дна
        zone_lo = piv["S3"] * (1 - 0.003)
        zone_hi = min(piv["S2"] * (1 + 0.002), price)  # не выше текущей
        tgt1 = piv["S2"] if price <= piv["S3"] else min(piv["P"], piv["S1"] + 0.2*rng)
        tgt2 = piv["P"] if price <= piv["S3"] else min(piv["R1"], piv["P"] + 0.5*rng)
        stop = (piv["S3"] if price <= piv["S3"] else piv["S2"]) - max(0.5*last_atr, 0.003*price)
        return Output(
            stance="BUY",
            entry=(round(zone_lo,2), round(zone_hi,2)),
            target1=round(tgt1,2),
            target2=round(tgt2,2),
            stop=round(stop,2),
            alt="WAIT — если импульс вниз слишком силён",
            comment="Перепроданность у дна; берём отскок к опорам.",
            meta=meta
        )

    # Середина диапазона — преимущество не на нашей стороне
    # Для MID/LONG предпочтение — дождаться отката к опоре
    # Для SHORT — ждать подтверждения разворота
    # Зона интереса = около P/S1 (для покупок) или R1 (для шортов) — в зависимости от расположения цены
    if price > piv["P"]:
        # Ждать отката к P…R1 (для аккуратного шорта) – но по умолчанию WAIT
        zone_lo = min(piv["R1"], piv["P"] + 0.2*rng)
        zone_hi = max(piv["R1"], piv["P"] + 0.35*rng)
        alt = f"SHORT от {round(zone_lo,2)}…{round(zone_hi,2)} при ослаблении"
    else:
        # Ждать отката к S1…P (для аккуратного лонга)
        zone_lo = min(piv["S1"], piv["P"] - 0.35*rng)
        zone_hi = max(piv["S1"], piv["P"] - 0.2*rng)
        alt = f"BUY от {round(zone_lo,2)}…{round(zone_hi,2)} при стабилизации"

    return Output(
        stance="WAIT",
        entry=None,
        target1=None,
        target2=None,
        stop=None,
        alt=alt,
        comment="Сейчас выгоднее подождать: преимущество не на нашей стороне.",
        meta=meta
    )

# --------------------------- Публичная функция ---------------------------

def analyze_ticker(ticker: str, horizon: Literal["short","mid","long"]) -> Dict[str,Any]:
    """
    Главная точка входа.
    Возвращает ПЛОСКИЙ dict: stance, entry, target1, target2, stop, alt, comment, meta
    """
    # свечей хватит с запасом, чтобы построить годовые пивоты и индикаторы
    lookback = {"short": 180, "mid": 420, "long": 900}[horizon]
    data = fetch_daily(ticker, lookback_days=lookback)
    out = compose_decision(ticker, horizon, data)
    return out.to_dict()

