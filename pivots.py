import pandas as pd

def _fib_levels(H: float, L: float, C: float):
    P = (H+L+C)/3.0
    rng = H - L
    R1 = P + 0.382*rng
    R2 = P + 0.618*rng
    R3 = P + 1.000*rng
    S1 = P - 0.382*rng
    S2 = P - 0.618*rng
    S3 = P - 1.000*rng
    return dict(P=P, R1=R1, R2=R2, R3=R3, S1=S1, S2=S2, S3=S3)

def previous_period_pivots(df: pd.DataFrame, horizon: str) -> dict:
    """
    horizon: 'short' (prev week), 'mid' (prev month), 'long' (prev year)
    df — daily OHLC
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")

    if horizon=="short":
        grp = d.resample("W-MON")   # недели (понедельник — начало)
    elif horizon=="mid":
        grp = d.resample("M")
    else:
        grp = d.resample("Y")

    # последний завершённый период = предпоследняя группа
    agg = grp.agg({"high":"max","low":"min","close":"last"})
    if len(agg)<2:
        raise RuntimeError("Недостаточно данных для расчёта пивотов.")
    H, L, C = agg.iloc[-2]["high"], agg.iloc[-2]["low"], agg.iloc[-2]["close"]
    return _fib_levels(H, L, C)
