# app.py
import os
import streamlit as st
from narrator import humanize
from core_strategy import analyze_ticker
from backtest import run_backtest

st.set_page_config(page_title="CapinteL-Q — Анализ рынков (Polygon)", layout="centered")

st.title("CapinteL-Q — Анализ рынков (Polygon)")
st.caption("Источник данных: Polygon.io • В тексте — только действия (вход/цели/стоп/альтернатива). "
           "Внутренние правила и расчёты скрыты.")

ticker = st.text_input("Тикер (например, QQQ, AAPL, X:BTCUSD)", "QQQ").upper().strip()

h_map = {
    "Трейд (1–5 дней)": "short",
    "Среднесрок (1–4 недели)": "mid",
    "Долгосрок (1–6 месяцев)": "long",
}
h_label = st.selectbox("Горизонт:", list(h_map.keys()), index=1)
horizon = h_map[h_label]

colA, _ = st.columns([1,1])
with colA:
    if st.button("Проанализировать", use_container_width=True):
        try:
            dec = analyze_ticker(ticker, horizon=horizon)
            txt = humanize(ticker, dec)   # <- без индикаторов
            st.subheader(f"{ticker} — текущая цена: {dec.get('meta',{}).get('price','—')}")
            st.markdown("### 🧠 Результат:")
            st.write(txt)
        except Exception as e:
            st.error(f"Ошибка: {e}")

with st.expander("📊 Бэктест (экспериментально)"):
    y = st.slider("Период, лет:", 1, 5, 3)
    start_cap = st.number_input("Стартовый капитал, $", 1_000.0, 10_000_000.0, 100_000.0, step=1_000.0)
    fee_bp = st.number_input("Комиссия, б.п. (в обе стороны)", 0.0, 50.0, 5.0, step=0.5)
    if st.button("Запустить бэктест"):
        with st.spinner("Считаю…"):
            res = run_backtest(ticker, horizon=horizon, years=y, start_capital=start_cap, fee_bp=fee_bp)
        if "error" in res.get("summary", {}):
            st.warning(res["summary"]["error"])
        else:
            st.markdown("#### Результаты:")
            st.json(res["summary"])
            st.dataframe(res["trades"], use_container_width=True)
