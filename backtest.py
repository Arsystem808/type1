import datetime as dt
import pandas as pd
from polygon_client import fetch_daily
from core_strategy import analyze_ticker

def _simulate_trades(df: pd.DataFrame, horizon:str):
    """
    На каждый день принимаем решение; позиция одна за раз.
    Выходим по стопу/любой цели, иначе закрываем в конце.
    """
    balance=1.0
    trade_log=[]
    in_pos=False
    side=None; entry=None; stop=None; t1=None; t2=None

    for i in range(120, len(df)):  # первые 120 баров — «разогрев индикаторов»
        sub = df.iloc[:i+1]
        dec = analyze_ticker(sub, "TEST", horizon)

        price = float(sub.iloc[-1]["close"])

        if not in_pos:
            if dec.stance in ("LONG","SHORT") and dec.entry:
                # берём средину диапазона входа
                e = dec.entry if not isinstance(dec.entry, tuple) else (dec.entry[0]+dec.entry[1])/2
                side=dec.stance; entry=e; stop=dec.stop; t1=dec.target1; t2=dec.target2
                in_pos=True
                trade_log.append({"date":sub.iloc[-1]["date"], "action":"ENTER", "side":side, "price":entry})
        else:
            hit=None
            if side=="LONG":
                if price<=stop: hit="STOP"
                elif t1 and price>=t1: hit="T1"
                elif t2 and price>=t2: hit="T2"
            else:
                if price>=stop: hit="STOP"
                elif t1 and price<=t1: hit="T1"
                elif t2 and price<=t2: hit="T2"
            if hit:
                pnl = (price-entry) if side=="LONG" else (entry-price)
                balance *= (1 + pnl/entry*0.99)  # 1% издержки на сделку
                trade_log.append({"date":sub.iloc[-1]["date"], "action":hit, "price":price, "pnl":pnl, "balance":balance})
                in_pos=False
                side=None

    return pd.DataFrame(trade_log), balance

def run_backtest(ticker:str, years:int=3, horizon:str="mid"):
    end = dt.date.today()
    start = dt.date(end.year-years, end.month, end.day)
    df = fetch_daily(ticker, start.isoformat(), end.isoformat())
    if df.empty: 
        raise RuntimeError("Нет данных для бэктеста.")
    log, bal = _simulate_trades(df, horizon)
    return log, bal
