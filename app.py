# app.py
import os
import streamlit as st
from core_strategy import analyze_ticker
from narrator import humanize
from memo import build_invest_memo

st.set_page_config(page_title="CapinteL‑Q — Market Intelligence", page_icon="📊", layout="centered")

st.markdown("## CapinteL‑Q — Анализ рынков (Polygon)")
st.caption("Источник данных: Polygon.io • без CSV, без Yahoo. В тексте — только действия (вход/цели/стоп/альтернатива). Внутренние правила и расчёты скрыты.")

with st.sidebar:
    st.markdown("### Диагностика")
    st.write("Python:", os.sys.version.split()[0])
    st.write("Polygon ключ:", "ОК" if os.getenv("POLYGON_API_KEY") else "⛔️ не найден")

ticker = st.text_input("Тикер (например, QQQ, AAPL, X:BTCUSD)", value="QQQ")
hru2code = {"Трейд (1–5 дней)":"short","Среднесрок (1–4 недели)":"mid","Долгосрок (1–6 месяцев)":"long"}
opt = st.selectbox("Горизонт:", list(hru2code.keys()), index=1)

if st.button("Проанализировать", use_container_width=True):
    with st.spinner("Собираю данные и считаю…"):
        try:
            horizon = hru2code[opt]
            dec = analyze_ticker(ticker, horizon=horizon)
            st.markdown(f"### {ticker.upper()} — текущая цена: ${dec.price:.2f}")
            st.markdown("#### 🧠 Результат:")
            st.markdown(humanize(dec, ticker))
            with st.expander("📄 Сформировать инвест‑мемо", expanded=False):
                st.markdown(build_invest_memo(ticker, dec))
        except Exception as e:
            st.error(f"Ошибка: {e}")
