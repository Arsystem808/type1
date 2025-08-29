import numpy as np
import pandas as pd
from indicators import heikin_ashi, ha_run_length, macd_hist, streak_len_sign, slowdown_last_n, rsi_wilder, atr_wilder
from pivots import previous_period_pivots
from narrator import Decision

# пороги по ТФ
HA_THRESH = {"short":4, "mid":5, "long":6}
MACD_STREAK = {"short":4, "mid":6, "long":8}
# допуск к уровням (доля)
TOL = {"short":0.008, "mid":0.010, "long":0.012}

def _near(price, level, tol)->bool:
    return abs(price-level) <= level*tol

def analyze_ticker(df_daily: pd.DataFrame, ticker:str, horizon:str)->Decision:
    """
    df_daily: колонки ['date','open','high','low','close','volume']
    horizon: 'short'|'mid'|'long'
    """
    d = df_daily.copy().reset_index(drop=True)
    if len(d)<120:
        raise RuntimeError("Мало данных.")

    piv = previous_period_pivots(d, horizon)
    price = float(d.iloc[-1]["close"])

    # индикаторы (на дневках)
    ind = heikin_ashi(d)
    ha_len = ha_run_length(ind["ha_color"])

    macdh = macd_hist(d["close"])
    macd_len, macd_sign = streak_len_sign(macdh)
    macd_tired = slowdown_last_n(macdh, n=3)

    rsi = rsi_wilder(d["close"])
    atr = float(atr_wilder(d).iloc[-1])

    # логика «перегрева у крыши»
    over_R2 = price >= piv["R2"]*(1- TOL[horizon])
    at_R3   = price >= piv["R3"]*(1- TOL[horizon])
    long_green = (ind["ha_color"].iloc[-1]==1) and (ha_len>=HA_THRESH[horizon])
    macd_hot   = (macd_sign==1 and macd_len>=MACD_STREAK[horizon]) or macd_tired
    rsi_high   = rsi.iloc[-1] > 60  # мягко

    if (over_R2 and long_green and macd_hot) or (at_R3 and macd_hot):
        # база = WAIT (не покупаем вершину)
        # альтернатива (для опытных) — аккуратный шорт
        entry_lo = piv["R2"]*(1-0.002); entry_hi = max(piv["R3"], price*(1+0.002))
        stop = entry_hi + 0.6*atr
        if at_R3:
            t1 = piv["R2"]
            t2 = piv["P"]
        else:
            t1 = (piv["P"] + piv["S1"])/2
            t2 = piv["S1"]
        comment = "Перегрев у крыши: длинная зелёная серия HA + горячий MACD. Играем от ослабления."
        return Decision("SHORT", (entry_lo, entry_hi), t1, t2, stop, comment,
                        meta={"price":price, "piv":piv, "atr":atr, "ha_len":ha_len, "macd_len":macd_len})

    # «перепроданность у дна»
    under_S2 = price <= piv["S2"]*(1+ TOL[horizon])
    at_S3    = price <= piv["S3"]*(1+ TOL[horizon])
    long_red = (ind["ha_color"].iloc[-1]==-1) and (ha_len>=HA_THRESH[horizon])
    macd_cold= (macd_sign==-1 and macd_len>=MACD_STREAK[horizon]) or macd_tired
    rsi_low  = rsi.iloc[-1] < 40

    if (under_S2 and long_red and macd_cold) or (at_S3 and macd_cold):
        entry_lo = min(piv["S3"], price*(1-0.002))
        entry_hi = piv["S2"]*(1+0.002)
        stop = entry_lo - 0.6*atr
        if at_S3:
            t1 = piv["S2"]
            t2 = piv["P"]
        else:
            t1 = (piv["P"] + piv["R1"])/2
            t2 = piv["R1"]
        comment = "Перепроданность у дна: длинная красная серия HA + уставший MACD. Берём восстановление."
        return Decision("LONG", (entry_lo, entry_hi), t1, t2, stop, comment,
                        meta={"price":price, "piv":piv, "atr":atr, "ha_len":ha_len, "macd_len":macd_len})

    # базовый отбор по «нормальным» покупкам/шортам от опор
    # MID/LONG: работаем после перезагрузки к P–S1 (лонг) или отбой от R2/R3 (шорт)
    # здесь — если далеко от краёв: WAIT
    comment = "На текущих уровнях явного преимущества нет — ждём откат к опоре или подтверждение разворота."
    return Decision("WAIT", None, None, None, None, comment,
                    meta={"price":price, "piv":piv, "atr":atr, "ha_len":ha_len, "macd_len":macd_len})
