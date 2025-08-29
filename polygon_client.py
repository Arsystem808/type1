# polygon_client.py
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

def fetch_daily(ticker: str, start: str | None = None, end: str | None = None, **kwargs) -> pd.DataFrame:
    """
    Гибкая обёртка Polygon /v2/aggs:
      - fetch_daily('QQQ', start='2022-01-01', end='2025-01-01')
      - fetch_daily('QQQ', days=400)   # альтернативный вариант
    """
    # поддержка старого вызова: days=...
    days = kwargs.pop("days", None)
    if days is not None:
        end_d = dt.date.today()
        start_d = end_d - dt.timedelta(days=int(days))
        start, end = start_d.isoformat(), end_d.isoformat()

    if not start or not end:
        # безопасный дефолт: 400 дней
        end_d = dt.date.today()
        start_d = end_d - dt.timedelta(days=400)
        start, end = start_d.isoformat(), end_d.isoformat()

    url = f"{BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start}/{end}"
    js = _get(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
    rows = js.get("results", []) or []
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)

def latest_price(ticker: str) -> tuple[float, pd.DataFrame]:
    """Последняя close и весь датасет (примерно за 400 дней)."""
    df = fetch_daily(ticker, days=400)
    if df.empty:
        raise RuntimeError("Не удалось получить котировки.")
    return float(df.iloc[-1]["close"]), df
