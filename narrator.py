# narrator.py
from typing import Optional, Tuple
from core_strategy import Decision

def humanize(dec: Decision, ticker: str) -> str:
    """Human-style memo without revealing internal math (no raw pivots in text)."""
    hru = {"short":"Трейд (1–5 дней)","mid":"Среднесрок (1–4 недели)","long":"Долгосрок (1–6 месяцев)"}[dec.horizon]
    header = f"📌 {ticker.upper()} — {hru}\n💵 Текущая цена: {dec.price:.2f}\n"
    if dec.stance == "BUY":
        body = "📈 База: BUY — работаем от спроса.\n"
    elif dec.stance == "SELL":
        body = "📉 База: SELL — работаем от предложения.\n"
    else:
        body = "⏸ База: WAIT — вход на текущих не даёт преимущества.\n"
    parts = [header, body]

    if dec.entry:
        parts.append(f"🎯 Зона входа: {dec.entry[0]:.2f}…{dec.entry[1]:.2f}\n")
    if dec.target1:
        parts.append(f"🎯 Цель 1: {dec.target1:.2f}\n")
    if dec.target2:
        parts.append(f"🎯 Цель 2: {dec.target2:.2f}\n")
    if dec.stop:
        parts.append(f"🛡️ Стоп/защита: {dec.stop:.2f}\n")

    if dec.notes:
        parts.append(f"🧭 Комментарий: {dec.notes}")

    return "".join(parts)
