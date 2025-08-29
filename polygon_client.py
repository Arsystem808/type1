# polygon_client.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY","").strip()

# Helper: convert Polygon's aggregate bars to DataFrame
def _bars_to_df(bars):
    if not bars:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    rows = []
    for b in bars:
        rows.append({
            "timestamp": pd.to_datetime(b["t"], unit="ms", utc=True).tz_convert("UTC"),
            "open": float(b["o"]), "high": float(b["h"]), "low": float(b["l"]),
            "close": float(b["c"]), "volume": float(b.get("v",0))
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df

def fetch_daily(ticker:str, days:int=365*3, adjusted:bool=True) -> pd.DataFrame:
    """
    Download ~N days of daily bars for ticker from Polygon.io aggregates v2 endpoint.
    Returns UTC-indexed DataFrame with columns: open, high, low, close, volume.
    """
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY отсутствует в окружении")
    ticker = ticker.upper().strip()
    # Crypto support (like X:BTCUSD)
    is_crypto = ":" in ticker
    mult = 1
    timespan = "day"
    base_url = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{timespan}/{from_}/{to_}"
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days+5)
    url = base_url.format(ticker=ticker, mult=mult, timespan=timespan, from_=start.isoformat(), to_=end.isoformat())
    params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "apiKey": POLYGON_API_KEY, "limit":50000}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    df = _bars_to_df(results)
    if df.empty:
        raise RuntimeError(f"Polygon вернул пусто для {ticker}")
    df = df.set_index("timestamp")
    return df

def fetch_last_price(ticker:str) -> float:
    """
    Last close (from daily) as a robust 'current' price proxy.
    """
    df = fetch_daily(ticker, days=10)
    return float(df["close"].iloc[-1])
