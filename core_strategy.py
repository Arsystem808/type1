# core_strategy.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import time
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List, Literal

import requests


# =========================
# Конфиг и утилиты времени
# =========================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()

Horizon = Literal["short", "mid", "long"]  # 1–5д, 1–4н, 1–6м

def _utc_today() -> dt.date:
    return dt.datetime.utcnow().date()

def _date_of_ms(ms: int) -> dt.date:
    return dt.datetime.utcfromtimestamp(ms / 1000).date()

def _start_of_iso_week(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())

def _start_of_month(d: dt.date) -> dt.date:
    return d.replace(day=1)

def _start_of_year(d: dt.date) -> dt.date:
    return d.replace(month=1, day=1)

def _prev_complete_week(today: dt.date) -> Tuple[dt.date, dt.date]:
    start_cur = _start_of_iso_week(today)
    end_prev = start_cur - dt.timedelta(days=1)
    start_prev = end_prev - dt.timedelta(days=6)
    return start_prev, end_prev

def _prev_complete_month(today: dt.date) -> Tuple[dt.date, dt.date]:
    first_cur = _start_of_month(today)
    end_prev = first_cur - dt.timedelta(days=1)
    first_prev = _start_of_month(end_prev)
    return first_prev, end_prev

def _prev_complete_year(today: dt.date) -> Tuple[dt.date, dt.date]:
    first_cur = _start_of_year(today)
    end_prev = first_cur - dt.timedelta(days=1)
    first_prev = _start_of_year(end_prev)
    return first_prev, end_prev


# ================
# Polygon загрузчик
# ================

class PolygonError(RuntimeError):
    pass

def _poly_agg_day(ticker: str, start: dt.date, end: dt.date) -> List[Dict[str, Any]]:
    """
    Днёвки по тикеру [start..end] включительно; sort=asc.
    Возвращает список баров: {t(ms), o,h,l,c,v}
    """
    if not POLYGON_API_KEY:
        raise PolygonError("POLYGON_API_KEY отсутствует в окружении")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code == 429:
        # грубый backoff на всякий
        time.sleep(1.0)
        r = requests.get(url, params=params, timeout=20)
    if r.status_code >= 300:
        raise PolygonError(f"Polygon {r.status_code}: {r.text[:240]}")
    data = r.json()
    results = data.get("results") or []
    out = []
    for x in results:
        out.append({
            "t": x["t"],
            "o": float(x["o"]),
            "h": float(x["h"]),
            "l": float(x["l"]),
            "c": float(x["c"]),
            "v": float(x.get("v", 0.0)),
        })
    return out

def _poly_last_n_days(ticker: str, n: int) -> List[Dict[str, Any]]:
    end = _utc_today()
    start = end - dt.timedelta(days=max(n, 10))
    return _poly_agg_day(ticker, start, end)


# ==================
# Технич. расчёты
# ==================

def heikin_ashi(bars: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Heikin Ashi из обычных OHLC."""
    if not bars:
        return []
    out = []
    ha_open = (bars[0]["o"] + bars[0]["c"]) / 2.0
    ha_close = (bars[0]["o"] + bars[0]["h"] + bars[0]["l"] + bars[0]["c"]) / 4.0
    ha_high = max(bars[0]["h"], ha_open, ha_close)
    ha_low  = min(bars[0]["l"], ha_open, ha_close)
    out.append({"o": ha_open, "h": ha_high, "l": ha_low, "c": ha_close})
    for b in bars[1:]:
        ha_close = (b["o"] + b["h"] + b["l"] + b["c"]) / 4.0
        ha_open = (out[-1]["o"] + out[-1]["c"]) / 2.0
        ha_high = max(b["h"], ha_open, ha_close)
        ha_low  = min(b["l"], ha_open, ha_close)
        out.append({"o": ha_open, "h": ha_high, "l": ha_low, "c": ha_close})
    return out

def _ema(values: List[float], period: int) -> List[Optional[float]]:
    if not values or period <= 0:
        return [None]*len(values)
    k = 2 / (period + 1)
    out: List[Optional[float]] = []
    ema_val: Optional[float] = None
    for v in values:
        if ema_val is None:
            ema_val = v
        else:
            ema_val = (v - ema_val) * k + ema_val
        out.append(ema_val)
    return out

def macd_hist(closes: List[float]) -> List[Optional[float]]:
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [ (a - b) if (a is not None and b is not None) else None for a,b in zip(ema12, ema26) ]
    sig9 = _ema([0.0 if v is None else v for v in macd], 9)
    hist = []
    for m, s in zip(macd, sig9):
        if m is None or s is None:
            hist.append(None)
        else:
            hist.append(m - s)
    return hist

def rsi_wilder(closes: List[float], period: int = 14) -> List[Optional[float]]:
    if len(closes) < period + 1:
        return [None]*len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    # начальная средняя
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out = [None]*(period)  # до первого значения
    rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
    out.append(100 - 100/(1+rs))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        out.append(100 - 100/(1+rs))
    return out if len(out) == len(closes) else out + [None]*(len(closes)-len(out))

def atr_wilder(bars: List[Dict[str,float]], period: int=14) -> List[Optional[float]]:
    if len(bars) < period+1:
        return [None]*len(bars)
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b["h"] - b["l"])
        else:
            prev_c = bars[i-1]["c"]
            tr = max(b["h"]-b["l"], abs(b["h"]-prev_c), abs(b["l"]-prev_c))
            trs.append(tr)
    # Wilder smoothing
    atr = [None]*len(bars)
    atr[period] = sum(trs[:period+1]) / (period+1)
    for i in range(period+1, len(bars)):
        atr[i] = (atr[i-1]*(period) + trs[i]) / (period+1)
    return atr


# ============================
# Пивоты (Fibonacci) прошл. ТФ
# ============================

def _pivots_fib(H: float, L: float, C: float) -> Dict[str, float]:
    P = (H + L + C) / 3.0
    rng = H - L
    R1 = P + 0.382 * rng
    R2 = P + 0.618 * rng
    R3 = P + 1.000 * rng
    S1 = P - 0.382 * rng
    S2 = P - 0.618 * rng
    S3 = P - 1.000 * rng
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}

def _period_slice_prev(bars: List[Dict[str,Any]], hz: Horizon) -> List[Dict[str,Any]]:
    """Возвращает бары прошлой недели/месяца/года (завершённого периода)."""
    if not bars:
        return []
    today = _utc_today()
    if hz == "short":
        a, b = _prev_complete_week(today)
    elif hz == "mid":
        a, b = _prev_complete_month(today)
    else:
        a, b = _prev_complete_year(today)
    out = [x for x in bars if a <= _date_of_ms(x["t"]) <= b]
    # запасной план, если рынок выходной/праздники → пусто
    if not out:
        lookback = {"short": 7, "mid": 35, "long": 400}[hz]
        out = bars[-lookback-1:-1]
    return out

def _pivots_for_horizon(bars: List[Dict[str,Any]], hz: Horizon) -> Dict[str, float]:
    prev = _period_slice_prev(bars, hz)
    H = max(x["h"] for x in prev)
    L = min(x["l"] for x in prev)
    C = prev[-1]["c"]
    return _pivots_fib(H, L, C)


# ============================
# «Серии» (стрики) по знаку
# ============================

def _last_streak_by_sign(values: List[float]) -> int:
    """Длина последней одноцветной серии (+/-). Возвращает signed length."""
    if not values:
        return 0
    last_sign = 1 if values[-1] >= 0 else -1
    n = 0
    for v in reversed(values):
        s = 1 if v >= 0 else -1
        if s == last_sign:
            n += 1
        else:
            break
    return n * last_sign


# ============================
# Результат стратегии / вывод
# ============================

@dataclass
class Decision:
    stance: Literal["BUY","SHORT","WAIT"]
    entry: Optional[Tuple[float, float]]  # диапазон входа или None
    target1: Optional[float]
    target2: Optional[float]
    stop: Optional[float]
    alt: Optional[str]
    comment: str
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _tolerances(hz: Horizon) -> Tuple[float, float]:
    # допуски к R2/R3 (и S2/S3) по горизонту
    if hz == "short":
        return 0.008, 0.012
    if hz == "mid":
        return 0.010, 0.016
    return 0.012, 0.020


def _human_comment_wait(hz: Horizon) -> str:
    return {
        "short": "Для трейда лучше не гнаться: ждём реакцию у края диапазона и подтверждение.",
        "mid":   "Сейчас выгоднее подождать: нужен откат к опоре прошлого месяца или явный отказ.",
        "long":  "Долгосрок — без суеты: ждём перезагрузку к годовым опорам и признаки разворота.",
    }[hz]


def _build_decision(bars: List[Dict[str,Any]], hz: Horizon) -> Decision:
    # текущая цена
    price = bars[-1]["c"]

    # пивоты от прошлой недели/месяца/года
    piv = _pivots_for_horizon(bars, hz)

    # тех.метры (в meta, в текст не показываем)
    ha = heikin_ashi(bars)
    ha_close = [x["c"] for x in ha]
    ha_delta = [0.0] + [ha_close[i]-ha_close[i-1] for i in range(1, len(ha_close))]
    ha_st = _last_streak_by_sign(ha_delta)

    closes = [x["c"] for x in bars]
    macdh = macd_hist(closes)
    macd_st = _last_streak_by_sign([0.0 if v is None else v for v in macdh])

    rsi = rsi_wilder(closes, 14)
    atr = atr_wilder(bars, 14)
    cur_atr = next((x for x in reversed(atr) if x is not None), None)

    # зоны около R2/R3 & S2/S3
    tol_r2, tol_r3 = _tolerances(hz)
    near_r2 = (piv["R2"] * (1 - tol_r2), piv["R2"] * (1 + tol_r2))
    near_r3 = (piv["R3"] * (1 - tol_r3), piv["R3"] * (1 + tol_r3))
    near_s2 = (piv["S2"] * (1 - tol_r2), piv["S2"] * (1 + tol_r2))
    near_s3 = (piv["S3"] * (1 - tol_r3), piv["S3"] * (1 + tol_r3))

    in_r2_zone = near_r2[0] <= price <= near_r2[1]
    in_r3_zone = near_r3[0] <= price <= near_r3[1]
    in_s2_zone = near_s2[0] <= price <= near_s2[1]
    in_s3_zone = near_s3[0] <= price <= near_s3[1]

    need_ha = {"short":4, "mid":5, "long":6}[hz]
    need_mh = {"short":4, "mid":6, "long":8}[hz]

    over_roof = (in_r3_zone or in_r2_zone) and (abs(ha_st) >= need_ha) and (abs(macd_st) >= need_mh) and (macdh[-1] is not None and macdh[-1] > 0)
    under_floor = (in_s3_zone or in_s2_zone) and (abs(ha_st) >= need_ha) and (abs(macd_st) >= need_mh) and (macdh[-1] is not None and macdh[-1] < 0)

    meta = {
        "price": price,
        "piv": {k: round(v, 2) for k, v in piv.items()},
        "rsi": rsi[-1],
        "atr": cur_atr,
        "ha_streak": ha_st,
        "macd_streak": macd_st,
        "horizon": hz,
    }

    # SHORT у крыши
    if over_roof:
        if in_r3_zone:
            entry = (min(price, near_r3[0]), near_r3[1])
            t1, t2 = piv["R2"], piv["P"]
            stop = piv["R3"] + (cur_atr or (piv["R3"] - piv["R2"]) * 0.4) * 0.8
        else:  # у R2
            entry = (near_r2[0], near_r2[1])
            # консервативная первая цель ближе к P/S1, вторая глубже при силе
            t1 = max(piv["P"] - 0.25*(piv["P"]-piv["S1"]), piv["S1"])
            t2 = piv["S1"] if hz != "long" else piv["S2"]
            stop = piv["R2"] + (cur_atr or (piv["R2"] - piv["R1"]) * 0.5) * 0.8

        comment = "Перегруженный верх. Играем от ослабления, работаем аккуратно."
        alt = None
        return Decision(
            stance="SHORT",
            entry=(round(entry[0],2), round(entry[1],2)),
            target1=round(t1,2),
            target2=round(t2,2),
            stop=round(stop,2),
            alt=alt,
            comment=comment,
            meta=meta
        )

    # LONG у дна
    if under_floor:
        if in_s3_zone:
            entry = (near_s3[0], max(price, near_s3[1]))
            t1, t2 = piv["S2"], piv["P"]
            stop = piv["S3"] - (cur_atr or (piv["S2"] - piv["S3"]) * 0.4) * 0.8
        else:  # у S2
            entry = (near_s2[0], near_s2[1])
            t1 = min(piv["P"] + 0.25*(piv["R1"]-piv["P"]), piv["R1"])
            t2 = piv["R1"] if hz != "long" else piv["R2"]
            stop = piv["S2"] - (cur_atr or (piv["S1"] - piv["S2"]) * 0.5) * 0.8

        comment = "Цена у опорного дна. Берём восстановление с контролируемым риском."
        alt = None
        return Decision(
            stance="BUY",
            entry=(round(entry[0],2), round(entry[1],2)),
            target1=round(t1,2),
            target2=round(t2,2),
            stop=round(stop,2),
            alt=alt,
            comment=comment,
            meta=meta
        )

    # База — WAIT (разный текст и альтернатива)
    if price > piv["P"]:
        zone_lo = min(piv["R1"], piv["P"] + 0.20*(piv["R2"]-piv["P"]))
        zone_hi = max(piv["R1"], piv["P"] + 0.35*(piv["R2"]-piv["P"]))
        alt = f"SHORT при ослаблении от {round(zone_lo,2)}…{round(zone_hi,2)}"
    else:
        zone_lo = min(piv["S1"], piv["P"] - 0.35*(piv["P"]-piv["S2"]))
        zone_hi = max(piv["S1"], piv["P"] - 0.20*(piv["P"]-piv["S2"]))
        alt = f"BUY после стабилизации от {round(zone_lo,2)}…{round(zone_hi,2)}"

    return Decision(
        stance="WAIT",
        entry=None, target1=None, target2=None, stop=None,
        alt=alt,
        comment=_human_comment_wait(hz),
        meta=meta
    )


# ============================
# Публичный интерфейс стратегии
# ============================

def analyze_ticker(ticker: str, horizon: Horizon = "mid") -> Decision:
    """
    Главная функция: тянем днёвки, считаем прошлый периодные пивоты, серии и выдаём решение.
    horizon: "short" (1–5д) | "mid" (1–4н) | "long" (1–6м)
    """
    if horizon not in ("short","mid","long"):
        raise ValueError("horizon must be: short|mid|long")

    # запас по истории, чтобы точно покрыть прошлый год
    bars = _poly_last_n_days(ticker, 500)
    if not bars:
        raise PolygonError("Нет данных по тикеру")

    return _build_decision(bars, horizon)


# Быстрый self-test (локально):  python -m core_strategy
if __name__ == "__main__":
    hz_list: List[Horizon] = ["short","mid","long"]
    tk = os.getenv("TEST_TICKER", "QQQ")
    for hz in hz_list:
        d = analyze_ticker(tk, hz)
        print(hz.upper(), d.stance, d.entry, d.target1, d.target2, d.stop)
        print(json.dumps(d.meta, ensure_ascii=False))
