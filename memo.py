# memo.py
from core_strategy import Decision

def build_invest_memo(ticker: str, dec: Decision) -> str:
    """
    Render a clean fund-style memo: action, entry, targets, stop — without exposing internals.
    """
    lines = []
    lines.append(f"# {ticker.upper()} — Инвест-идея\n")
    stance = {"BUY":"Покупка","SELL":"Продажа","WAIT":"Наблюдать"}[dec.stance]
    hru = {"short":"Трейд (1–5 дней)","mid":"Среднесрок (1–4 недели)","long":"Долгосрок (1–6 месяцев)"}[dec.horizon]
    lines.append(f"**Горизонт:** {hru}  \n**Базовый сценарий:** {stance}\n")
    lines.append(f"**Текущая цена:** {dec.price:.2f}\n")
    if dec.entry:
        lines.append(f"**Зона входа:** {dec.entry[0]:.2f} — {dec.entry[1]:.2f}\n")
    if dec.target1:
        lines.append(f"**Цель 1:** {dec.target1:.2f}\n")
    if dec.target2:
        lines.append(f"**Цель 2:** {dec.target2:.2f}\n")
    if dec.stop:
        lines.append(f"**Стоп/защита:** {dec.stop:.2f}\n")
    if dec.notes:
        lines.append(f"\n_Комментарий:_ {dec.notes}\n")
    lines.append("\n_Внутренние расчёты и уровни скрыты намеренно._\n")
    return "".join(lines)
