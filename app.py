# app.py
# Streamlit UI для Capintel-Q (Polygon). Совместим с Python 3.11–3.12.
# Предполагается, что core_strategy.py экспортирует:
#   from narrator import Decision, Stance
#   def analyze_ticker(ticker: str, horizon: str) -> Decision

import os
import traceback
import streamlit as st

from core_strategy import analyze_ticker  # возвращает Decision-объект

# -----------------------------
# Настройки страницы и UI
# -----------------------------
st.set_page_config(
    page_title="Capintel-Q — Анализ рынков (Polygon)",
    page_icon="📊",
    layout="centered",
)

st.markdown(
    "<h1 style='margin-bottom:0'>CapintelL-Q — Анализ рынков (Polygon)</h1>"
    "<div style='opacity:.7'>Источник данных: Polygon.io · В тексте — только действия (вход/цели/стоп/альтернатива). "
    "Внутренние правила и расчёты скрыты.</div>",
    unsafe_allow_html=True,
)

# Проверим ключ Polygon (не обязателен для запуска UI, но предупредим)
if not os.getenv("POLYGON_API_KEY"):
    st.warning(
        "Не найден POLYGON_API_KEY в окружении. Данные могут не загрузиться. "
        "Добавь секрет в Streamlit → Settings → Secrets."
    )

# -----------------------------
# Ввод пользователя
# -----------------------------
ticker = st.text_input(
    "Тикер (например, QQQ, AAPL, X:BTCUSD)",
    value="QQQ",
).strip().upper()

horizon_label = st.selectbox(
    "Горизонт:",
    ["Трейд (1–5 дней)", "Среднесрок (1–4 недели)", "Долгосрок (1–6 месяцев)"],
    index=1,
)

HMAP = {
    "Трейд (1–5 дней)": "short",
    "Среднесрок (1–4 недели)": "mid",
    "Долгосрок (1–6 месяцев)": "long",
}
horizon = HMAP[horizon_label]

run = st.button("Проанализировать", type="primary")

# -----------------------------
# Отрисовка результата
# -----------------------------
def render_decision(res):
    """Красиво вывести Decision-объект без раскрытия внутренней математики."""
    # Заголовок с текущей ценой
    st.subheader(f"{res.ticker} — текущая цена: ${res.price:,.2f}")

    # Основной блок: сценарий и текст-комментарий
    st.markdown("### 🧠 Результат:")
    # Комментарий уже 'human-style' из стратегии
    st.write(res.comment)

    # Лента действий (если что-то отсутствует — не показываем)
    info_lines = []
    if res.entry and isinstance(res.entry, (tuple, list)) and len(res.entry) == 2:
        info_lines.append(f"🎯 **Вход:** {res.entry[0]:.2f} … {res.entry[1]:.2f}")
    if res.target1:
        info_lines.append(f"🎯 **Цель 1:** {res.target1:.2f}")
    if res.target2:
        info_lines.append(f"🎯 **Цель 2:** {res.target2:.2f}")
    if res.stop:
        info_lines.append(f"🛡️ **Стоп/защита:** {res.stop:.2f}")
    if info_lines:
        st.markdown("\n\n".join(info_lines))

    # Диагностика — опционально (мета-данные/уровни), скрыто по умолчанию
    with st.expander("🔧 Диагностика (уровни и мета)"):
        # res.meta — это словарь с любыми тех. полями (не показываем по умолчанию)
        try:
            st.json(res.meta)
        except Exception:
            st.write(res.meta)

if run:
    if not ticker:
        st.error("Укажи тикер.")
    else:
        try:
            res = analyze_ticker(ticker, horizon=horizon)  # <-- ВОЗВРАЩАЕТ Decision
            # внутри app.py после analyze_ticker(...)
d = analyze_ticker(ticker, horizon_key)

st.subheader(f"{d.ticker} — текущая цена: ${d.price:,.2f}")
st.markdown("### 🧠 Результат:")
st.write(f"**Сценарий:** {d.stance} · **Уверенность:** {d.confidence}%")
st.write(d.comment)

if d.entry:
    st.write(f"🎯 **Вход:** {d.entry[0]:.2f} … {d.entry[1]:.2f}")
if d.target1: st.write(f"🎯 **Цель 1:** {d.target1:.2f}")
if d.target2: st.write(f"🎯 **Цель 2:** {d.target2:.2f}")
if d.stop:    st.write(f"🛡️ **Стоп/защита:** {d.stop:.2f}")

with st.expander("🔧 Диагностика (внутренняя)"):
    st.json(d.meta)

                st.code("".join(traceback.format_exc()))

# -----------------------------
# (необязательный) блок бэктеста — заглушка
# -----------------------------
st.markdown("---")
st.caption("📈 Бэктест (экспериментально) — будет работать только если в проекте есть run_backtest().")

try:
    from core_strategy import run_backtest  # опционально

    years = st.slider("Период, лет:", 1, 5, 3)
    start_capital = st.number_input("Стартовый капитал, $", value=100000.0, min_value=1000.0, step=1000.0, format="%.2f")
    fee_bps = st.number_input("Комиссия, б.п. (в обе стороны)", value=5.0, min_value=0.0, step=0.5, format="%.2f")

    if st.button("Запустить бэктест"):
        try:
            bt = run_backtest(
                ticker=ticker or "QQQ",
                horizon=horizon,
                years=years,
                start_capital=start_capital,
                fee_bps=fee_bps,
            )
            # Ожидаем, что bt — dict с полями summary/curve/trades и т.п.
            st.success("Бэктест завершён.")
            if isinstance(bt, dict):
                if "summary" in bt:
                    st.subheader("Итоги")
                    st.json(bt["summary"])
                if "trades" in bt:
                    st.subheader("Сделки")
                    st.dataframe(bt["trades"])
                if "equity" in bt:
                    st.subheader("Equity (синтетика)")
                    st.line_chart(bt["equity"])
            else:
                st.write(bt)
        except Exception as e:
            st.error(f"Ошибка бэктеста: {e}")
            with st.expander("Стек ошибки (бэктест)"):
                st.code("".join(traceback.format_exc()))
except Exception:
    # Если run_backtest отсутствует — просто ничего не делаем
    pass
