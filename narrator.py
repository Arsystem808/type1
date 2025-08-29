from dataclasses import dataclass

@dataclass
class Decision:
    stance: str        # 'LONG' / 'SHORT' / 'WAIT'
    entry: tuple|float|None
    target1: float|None
    target2: float|None
    stop: float|None
    comment: str
    meta: dict

def humanize(dec: Decision) -> str:
    if dec.stance == "WAIT":
        return f"🧠 Сейчас выгоднее подождать. {dec.comment}"
    side = "ЛОНГ" if dec.stance=="LONG" else "ШОРТ"
    if isinstance(dec.entry, tuple):
        e = f"{dec.entry[0]:.2f}…{dec.entry[1]:.2f}"
    elif dec.entry is None:
        e = "—"
    else:
        e = f"{float(dec.entry):.2f}"
    t1 = f"{dec.target1:.2f}" if dec.target1 else "—"
    t2 = f"{dec.target2:.2f}" if dec.target2 else "—"
    st = f"{dec.stop:.2f}" if dec.stop else "—"
    return (
        f"🧭 Сценарий: {side}\n"
        f"🎯 Вход: {e} | Цели: {t1} / {t2} | Защита: {st}\n"
        f"💬 Комментарий: {dec.comment}"
    )
