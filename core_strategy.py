# core_strategy.py
# -*- coding: utf-8 -*-
import os
import math
import time
import enum
import dataclasses as dc
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import requests
import pandas as pd


# ========= Параметры ==========

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
POLYGON_URL = "https://api.polygon.io"

# Сколько дней подтягиваем истории (с запасом)
DEFAULT_LOOKBACK_DAYS = 900  # ~3 года для дневок

# Порог допуска к уровням (в процентах) по горизонту
TOLERANCE = {
    "short": 0.008,   # 0.8%
    "mid":   0.010,   # 1.0%
    "long":  0.012,   # 1.2%
}

# Порог «длины серии» Heikin-Ashi (кол-во баров) по горизонту
HA_SERIES_MIN = {
    "short": 4,
    "mid":   5,
    "long":  6,
}

# Порог «стриков» MACD-гистограммы (кол-во баров) по горизонту
MACD_STREAK_MIN = {
    "short": 4,
    "mid":   6,
    "long":  8,
}

# ATR множители (для приблизительных стоп/целей)
ATR_MULT = {
    "short": {"stop": 0.8, "t1": 0.6, "t2": 1.1},
    "mid":   {"stop": 1.0, "t1": 1.0, "t2": 2.0},  # mid/long — цели больше завязаны на опорные зоны
    "long":  {"stop": 1.3, "t1": 1.5, "t2": 3.0},
}


# ========= Хелперы данных / Polygon ==========

def _http_get(url: str, params: Dict) -> Dict:
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY отсутствует в окружении")
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_daily(ticker: str, days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Грузим дневные свечи через /v2/aggs/ticker/{ticker}/range/1/day
    Берём последние 'days' баров.
    """
    # polygon принимает limit до 50000; берем с запасом
    url = f"{POLYGON_URL}/v2/aggs/ticker/{ticker.upper()}/range/1/day/1970-01-01/2100-01-01"
    data = _http_get(url, {"adjusted": "true", "sort": "desc", "limit": min(days, 50000)})
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"Нет данных по {ticker}")
    # сортируем по времени (возрастающе)
    rows = [{
        "ts": pd.to_datetime(x["t"], unit="ms"),
        "o": float(x["o"]), "h": float(x["h"]), "l": float(x["l"]), "c": float(x["c"]), "v": float(x.get("v", 0))
    } for x in results]
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    return df


# ========= Индикаторные расчёты (без раскрытия в текстах) ==========

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Формулы HA
    out["ha_close"] = (out["o"] + out["h"] + out["l"] + out["c"]) / 4.0
    ha_open = [out["o"].iloc[0]]
    for i in range(1, len(out)):
        ha_open.append((ha_open[i - 1] + out["ha_close"].iloc[i - 1]) / 2.0)
    out["ha_open"] = pd.Series(ha_open, index=out.index)
    out["ha_up"] = out["ha_close"] >= out["ha_open"]
    return out

def macd_hist(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = df["c"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["c"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return hist

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    high, low, close = df["h"], df["l"], df["c"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()  # Wilder
    return atr

def _streak(series_up: pd.Series) -> int:
    """длина последней серии True/False (по знаку серии выбираем конец)"""
    if len(series_up) == 0:
        return 0
    cnt = 1
    for i in range(len(series_up)-2, -1, -1):
        if series_up.iloc[i] == series_up.iloc[-1]:
            cnt += 1
        else:
            break
    return cnt


# ========= Уровни Pivot (Fibonacci) по старшему периоду ==========

def _pivot_fib(H: float, L: float, C: float) -> Dict[str, float]:
    P = (H + L + C) / 3.0
    R1 = P + 0.382 * (H - L)
    R2 = P + 0.618 * (H - L)
    R3 = P + 1.000 * (H - L)
    S1 = P - 0.382 * (H - L)
    S2 = P - 0.618 * (H - L)
    S3 = P - 1.000 * (H - L)
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}

def _prev_period_HLC(df: pd.DataFrame, horizon: str) -> Tuple[float, float, float]:
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["week"] = d["date"].dt.isocalendar().week.astype(int)

    if horizon == "short":
        # прошлую неделю
        grp = d.groupby(["year", "week"])
    elif horizon == "mid":
        # прошлый месяц
        grp = d.groupby(["year", "month"])
    else:
        # прошлый год
        grp = d.groupby(["year"])

    agg = grp.agg(H=("h", "max"), L=("l", "min"), C=("c", "last")).reset_index()
    if len(agg) < 2:
        # мало истории — берём последний доступный период
        row = agg.iloc[-1]
        return float(row.H), float(row.L), float(row.C)
    row = agg.iloc[-2]
    return float(row.H), float(row.L), float(row.C)


# ========= Решение и «человеческий» текст ==========

class Stance(str, enum.Enum):
    BUY = "BUY"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    WAIT = "WAIT"

@dataclass
class Decision:
    ticker: str
    horizon: str                # short / mid / long
    stance: Stance
    entry: Optional[Tuple[float, float]]  # (lo, hi) или None
    target1: Optional[float]
    target2: Optional[float]
    stop: Optional[float]
    comment: str
    price: float
    # Для «диагностики» внутри приложения; наружу не проговариваем
    meta: Dict[str, float]

    def as_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "horizon": self.horizon,
            "stance": self.stance.value,
            "entry": self.entry,
            "target1": self.target1,
            "target2": self.target2,
            "stop": self.stop,
            "comment": self.comment,
            "price": self.price,
            "meta": self.meta,
        }


def _human_text(dec: Decision) -> str:
    """Короткое живое резюме без раскрытия математики."""
    hz = {"short": "трейд", "mid": "среднесрок", "long": "долгосрок"}[dec.horizon]
    if dec.stance == Stance.WAIT:
        return f"Сейчас выгоднее подождать ({hz}). На текущих уровнях явного преимущества нет — ждём откат к опоре или подтверждение разворота."
    if dec.stance == Stance.BUY:
        a, b = dec.entry or (None, None)
        return f"Покупка ({hz}). Вход: {a:.2f}…{b:.2f}. Цели: {dec.target1:.2f} / {dec.target2:.2f}. Защита: {dec.stop:.2f}. Действуем аккуратно."
    if dec.stance == Stance.SHORT:
        a, b = dec.entry or (None, None)
        return f"Шорт ({hz}). Вход: {a:.2f}…{b:.2f}. Цели: {dec.target1:.2f} / {dec.target2:.2f}. Защита: {dec.stop:.2f}. Работаем от ослабления импульса."
    # CLOSE
    return f"Фиксация позиции ({hz}). Закрываем и ждём новую формацию."


# ========= Ядро принятия решения ==========

def analyze_ticker(ticker: str, horizon: str = "mid") -> Decision:
    """
    Возвращает Decision (объект, а не dict!)
    horizon: 'short' | 'mid' | 'long'
    """
    horizon = horizon.lower().strip()
    assert horizon in ("short", "mid", "long")

    # 1) Данные
    df = fetch_daily(ticker, days=DEFAULT_LOOKBACK_DAYS)
    px = float(df["c"].iloc[-1])

    # 2) Индикаторы (в мету; наружу не рассказываем)
    df_ha = heikin_ashi(df)
    ha_series = _streak(df_ha["ha_up"])
    hist = macd_hist(df)
    macd_up = hist.iloc[-1] >= 0
    hist_sign_series = _streak(hist >= 0)
    _atr = atr(df).iloc[-1]

    # 3) Опорные уровни (по прошлому периоду)
    H, L, C = _prev_period_HLC(df, horizon)
    piv = _pivot_fib(H, L, C)

    tol = TOLERANCE[horizon]
    atr_m = ATR_MULT[horizon]

    # 4) Композитная логика — без «выдачи формулы» наружу
    def near(x, y):  # x около y
        return abs(x - y) <= y * tol

    # Перегрев у крыши → чаще ищем шорт
    over_roof = (px >= piv["R2"]*(1 - tol)) and (ha_series >= HA_SERIES_MIN[horizon]) and (hist_sign_series >= MACD_STREAK_MIN[horizon])

    # Перепроданность у дна → чаще ищем лонг
    under_floor = (px <= piv["S2"]*(1 + tol)) and (ha_series >= HA_SERIES_MIN[horizon]) and (hist_sign_series >= MACD_STREAK_MIN[horizon])

    entry: Optional[Tuple[float, float]] = None
    t1 = t2 = st = None
    stance = Stance.WAIT
    comment = ""

    if over_roof:
        # Агрессивный шорт: вход у R2…R3
        lo = min(piv["R2"], px)
        hi = max(piv["R3"], px)
        entry = (lo, hi)
        # Цели — к R1/P (мягче) или глубже по горизонту
        t1 = piv["R1"]
        t2 = piv["P"] if horizon != "short" else max(piv["P"], px - atr_m["t2"] * _atr)
        st = max(piv["R2"], px) + atr_m["stop"] * _atr
        stance = Stance.SHORT
        comment = "Играем от ослабления. Не гонимся за вершиной — берём откат."

    elif under_floor:
        # Лонг от дна: вход у S2…S3
        lo = min(piv["S3"], px)
        hi = max(piv["S2"], px)
        entry = (lo, hi)
        t1 = piv["S1"]
        t2 = piv["P"] if horizon != "short" else min(piv["P"], px + atr_m["t2"] * _atr)
        st = min(piv["S2"], px) - atr_m["stop"] * _atr
        stance = Stance.BUY
        comment = "Покупаем от поддержки. Если импульс ломается — фиксируемся без упрямства."

    else:
        stance = Stance.WAIT
        comment = "На текущих уровнях явного преимущества нет — ждём откат к опоре или подтверждение разворота."

    dec = Decision(
        ticker=ticker.upper(),
        horizon=horizon,
        stance=stance,
        entry=entry,
        target1=t1,
        target2=t2,
        stop=st,
        comment=comment,
        price=px,
        meta={
            "P": piv["P"], "R1": piv["R1"], "R2": piv["R2"], "R3": piv["R3"],
            "S1": piv["S1"], "S2": piv["S2"], "S3": piv["S3"],
            "ha_series": float(ha_series),
            "macd_streak": float(hist_sign_series),
            "atr": float(_atr),
            "tolerance": float(tol),
            "text": _human_text  # просто ссылка на функцию (для app не используем)
        }
    )
    return dec


# ========= БЭКТЕСТ (очень компактный, но рабочий) ==========

@dc.dataclass
class BacktestRow:
    date: str
    price: float
    signal: str    # BUY/SHORT/WAIT
    pnl: float     # накопительный PnL после закрытия сделки (если была)


def run_backtest(
    ticker: str,
    horizon: str = "mid",
    years: int = 2,
    start_capital: float = 100_000.0,
    commission_bps_roundtrip: float = 5.0,  # б.п. «туда-обратно»
) -> Dict:
    """
    Простейший бэктест по дневкам.
    Правила:
      - каждый день строим новое Decision на основе истории до этого дня;
      - если stance BUY/SHORT и Close попадает в зону entry -> входим на close;
      - выходим по достижению target1 (или target2) либо по стопу;
      - иначе выходим, если сигнал сменился на WAIT.
    Это упрощение, но стабильное и быстрое.
    """
    horizon = horizon.lower().strip()
    df_full = fetch_daily(ticker, days=int(max(365*years + 60, 400)))
    rows: List[BacktestRow] = []

    pos_side = None      # "long"/"short"/None
    pos_entry = None
    capital = start_capital
    shares = 0.0
    cum_pnl = 0.0

    def _close_position(price: float):
        nonlocal pos_side, pos_entry, shares, capital, cum_pnl
        if pos_side is None:
            return
        if pos_side == "long":
            pnl = (price - pos_entry) * shares
        else:
            pnl = (pos_entry - price) * shares
        # комиссия «туда-обратно»
        fee = abs(price) * abs(shares) * (commission_bps_roundtrip/10000.0)
        pnl -= fee
        capital += pnl
        cum_pnl += pnl
        pos_side = None
        pos_entry = None
        shares = 0.0

    start_idx = max(50, len(df_full) - years*252)  # чтобы хватило истории для индикаторов
    for i in range(start_idx, len(df_full)):
        df = df_full.iloc[: i+1].copy()
        px = float(df["c"].iloc[-1])
        date_str = str(df["date"].iloc[-1])

        # строим решение на основании истории до текущего дня
        dec = analyze_ticker(ticker, horizon=horizon)
        # важное: используем текущую цену df_full.iloc[i]["c"], а не «свежий» интернет

        sig = dec.stance.value

        # логика вход/выход
        if pos_side is None:
            if dec.entry is not None and dec.stance in (Stance.BUY, Stance.SHORT):
                lo, hi = dec.entry
                if lo <= px <= hi:
                    pos_side = "long" if dec.stance == Stance.BUY else "short"
                    pos_entry = px
                    # берём на 1x капитал (упрощение)
                    shares = capital / px
        else:
            # есть позиция — проверяем t1/stop
            stop = dec.stop
            t1 = dec.target1
            if stop is not None:
                if pos_side == "long" and px <= stop:
                    _close_position(px)
                elif pos_side == "short" and px >= stop:
                    _close_position(px)
            if pos_side is not None and t1 is not None:
                if pos_side == "long" and px >= t1:
                    _close_position(px)
                elif pos_side == "short" and px <= t1:
                    _close_position(px)
            # если сигнала на продолжение нет — закрываем
            if pos_side is not None and dec.stance == Stance.WAIT:
                _close_position(px)

        rows.append(BacktestRow(date=date_str, price=px, signal=sig, pnl=cum_pnl))

    return {
        "ticker": ticker.upper(),
        "horizon": horizon,
        "start_capital": start_capital,
        "end_capital": capital,
        "pnl": cum_pnl,
        "rows": [dc.asdict(r) for r in rows]
    }

