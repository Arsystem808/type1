# app.py
# CapinteL-Q — Streamlit UI (Polygon edition)
import os
import streamlit as st
from core_strategy import analyze_ticker, run_backtest

st.set_page_config(
    page_title="CapinteL-Q — Анализ рынков (Polygon)",
    page_icon="📈",
    layout="centered"
)

st.markdown(
    "### CapinteL-Q — Анализ рынков (Polygon)\n"
    "Источник данных: **Polygon.io**. В тексте — только действия (вход/цели/стоп/альтернатива). "
    "Внутренние правила и расчёты скрыты."
)

# --- Ввод
ticker = st.text_input(
    "Тикер (например, QQQ, AAPL, X:BTCUSD)",
    value="QQQ",
    placeholder="QQQ"
).strip().upper()

h_map = {
    "Трейд (1–5 дней)": "short",
    "Среднесрок (1–4 недели)": "mid",
    "Долгосрок (1–6 месяцев)": "long",
}
h_label = st.selectbox("Горизонт:", list(h_map.keys()), index=1)
horizon = h_map[h_label]

# --- Кнопка анализа
if st.button("Проанализировать", use_container_width=True):
    try:
        d = analyze_ticker(ticker, horizon)

        # Заголовок с ценой
        st.markdown(f"#### {d.ticker} — текущая цена: ${d.price:,.2f}")

        # Результат
        st.markdown("### 🧠 Результат:")
        st.write(f"**Сценарий:** {d.stance} · **Уверенность:** {int(d.confidence)}%")
        if getattr(d, "comment", None):
            st.write(d.comment)

        # Уровни действий
        if getattr(d, "entry", None):
            st.write(f"🎯 **Вход:** {d.entry[0]:.2f} … {d.entry[1]:.2f}")
        if getattr(d, "target1", None):
            st.write(f"🎯 **Цель 1:** {d.target1:.2f}")
        if getattr(d, "target2", None):
            st.write(f"🎯 **Цель 2:** {d.target2:.2f}")
        if getattr(d, "stop", None):
            st.write(f"🛡️ **Стоп/защита:** {d.stop:.2f}")

        # Диагностика (без пивотов)
        meta = dict(getattr(d, "meta", {}) or {})
        if "pivots" in meta:
            meta.pop("pivots", None)  # не показываем уровни пивотов
        with st.expander("🔧 Диагностика (уровни и мета)", expanded=False):
            if meta:
                st.json(meta)
            else:
                st.caption("Мета-данные недоступны.")

    except Exception as e:
        st.error(f"Ошибка: {e}")

# --- Бэктест (экспериментально)
st.markdown("---")
st.markdown("### 🧪 Бэктест (экспериментально) — будет работать только если в проекте есть `run_backtest()`.")
years = st.slider("Период, лет:", 1, 5, 3)
start_capital = st.number_input("Стартовый капитал, $", value=100_000.0, min_value=1_000.0, step=1_000.0, format="%.2f")
fee_bps = st.number_input("Комиссия, б.п. (в обе стороны)", value=5.0, min_value=0.0, step=0.5, format="%.2f")

if st.button("Запустить бэктест", use_container_width=True):
    try:
        res = run_backtest(ticker=ticker, horizon=horizon, years=years,
                           start_capital=start_capital, fee_bps=fee_bps)
        summary = res.get("summary", {})
        st.success(
            f"Доходность: **{summary.get('ret_pct', 0):.2f}%** · "
            f"Старт: ${summary.get('start', 0):,.2f} → Конец: ${summary.get('end', 0):,.2f}"
        )

        eq = res.get("equity")
        if eq is not None and not eq.empty:
            st.line_chart(eq, x=eq.index, y="equity", height=220)

        trades = res.get("trades")
        if trades is not None and not trades.empty:
            st.caption(f"Количество сделок: {len(trades)}")
            st.dataframe(trades.tail(20), use_container_width=True, height=250)
        else:
            st.caption("Сделок за период не найдено (или они отфильтрованы логикой вход/выход).")
    except Exception as e:
        st.error(f"Ошибка бэктеста: {e}")

# --- Футер
st.markdown("---")
st.caption("CapinteL-Q формулирует вывод как человек и не раскрывает внутренние расчёты. "
           "Это не инвестсовет. Риски на рынке всегда есть.")
