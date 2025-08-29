import os, requests, datetime as dt
import pandas as pd

API = os.getenv("POLYGON_API_KEY","").strip()
BASE = "https://api.polygon.io"

def _check_key():
    if not API:
        raise RuntimeError("POLYGON_API_KEY не найден в окружении.")

def _get(url, params=None):
    _check_key()
    p = {"apiKey": API}
    if params:
        p.update(params)
    r = requests.get(url, params=p, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_daily(ticker:str, start:str, end:str)->pd.DataFrame:
    """
    Daily aggs (1d) [adj]
    start/end: 'YYYY-MM-DD'
    """
    url = f"{BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start}/{end}"
    js = _get(url, {"adjusted":"true","sort":"asc","limit":50000})
    rows = js.get("results",[]) or []
    if not rows: 
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = pd.DataFrame(rows)
    df["date"]  = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def latest_price(ticker:str)->float:
    """Берём последнюю дневную close; если есть ‘last trade’ — можно заменить."""
    today = dt.date.today()
    start = (today - dt.timedelta(days=400)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    df = fetch_daily(ticker, start, end)
    if df.empty: 
        raise RuntimeError("Не удалось получить котировки.")
    return float(df.iloc[-1]["close"]), df
