import os, datetime as dt, streamlit as st
from polygon_client import latest_price
from core_strategy import analyze_ticker
from narrator import humanize

st.set_page_config(page_title="CapinteL-Q — Анализ рынков (Polygon)", page_icon="🧭", layout="centered")

st.markdown("### CapinteL-Q — Анализ рынков (Polygon)")
st.caption("Источник данных: Polygon.io • В тексте — только действия (вход/цели/стоп/альтернатива). Внутренние правила и расчёты скрыты.")

ticker = st.text_input("Тикер (например, QQQ, AAPL, X:BTCUSD)", "QQQ")
h_map = {"Трейд (1–5 дней)":"short", "Среднесрок (1–4 недели)":"mid", "Долгосрок (1–6 месяцев)":"long"}
h_choice = st.selectbox("Горизонт", list(h_map.keys()), index=1)
horizon = h_map[h_choice]

if st.button("Проанализировать", type="primary"):
    try:
        price, df = latest_price(ticker)
        dec = analyze_ticker(df, ticker, horizon)
        st.subheader(f"{ticker.upper()} — текущая цена: ${price:,.2f}")
        st.markdown("### 🧠 Результат:")
        st.write(humanize(dec))
        with st.expander("Диагностика (уровни и мета)"):
            st.json(dec.meta)
    except Exception as e:
        st.error(f"Ошибка: {e}")
