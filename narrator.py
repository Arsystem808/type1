# narrator.py
from dataclasses import asdict, is_dataclass

def _as_dict(dec):
    if isinstance(dec, dict):
        return dec
    if is_dataclass(dec):
        return asdict(dec)
    # на всякий случай поддержим "объект-под-словарь"
    try:
        return dict(dec)
    except Exception:
        return {"stance": "WAIT"}

def _fmt_range(r):
    if not r or r[0] is None or r[1] is None:
        return "—"
    a, b = float(r[0]), float(r[1])
    if abs(a-b) < 1e-6:  # почти точка
        return f"{a:,.2f}"
    return f"{a:,.2f}…{b:,.2f}"

def _fmt_num(x):
    return "—" if x is None else f"{float(x):,.2f}"

def humanize(ticker: str, decision) -> str:
    d = _as_dict(decision)
    stance = (d.get("stance") or "").upper()
    entry = d.get("entry") or d.get("entry_range")
    t1 = d.get("target1") or (d.get("targets") or (None, None))[0]
    t2 = d.get("target2") or (d.get("targets") or (None, None))[1]
    stop = d.get("stop")
    alt  = d.get("alt")   # допустим, альтернативный сценарий внутрь словаря

    # Без раскрытия индикаторов: только действия.
    if stance == "WAIT":
        return (f"🧠 {ticker} — сейчас выгоднее подождать. "
                f"На текущих уровнях явного преимущества нет — ждём откат к опоре "
                f"или подтверждение разворота.")

    side = "LONG" if stance == "BUY" else "SHORT"
    lines = []
    lines.append(f"🎯 {ticker} — сценарий: {side}. "
                 f"Вход: {_fmt_range(entry)} | Цели: {_fmt_num(t1)} / {_fmt_num(t2)} | "
                 f"Защита: {_fmt_num(stop)}.")

    if alt:
        # альтернатива должна быть уже “обезличена” в core_strategy
        lines.append(f"🤝 Альтернатива: {alt}")

    # финальная оговорка без технических терминов
    lines.append("Работаем аккуратно: если сценарий ломается — выходим и ждём новую формацию.")
    return " ".join(lines)
