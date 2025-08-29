import streamlit as st
from polygon_client import latest_price
from core_strategy import analyze_ticker
from narrator import humanize

st.set_page_config(page_title="CapinteL-Q — Investment Memo", page_icon="📝", layout="centered")
st.markdown("### CapinteL-Q — Investment Memo (без раскрытия стратегии)")
st.caption("Источник данных: Polygon.io • В тексте — только действия (вход/цели/стоп/альтернатива). Внутренние правила и расчёты скрыты.")

ticker = st.text_input("Тикер", "QQQ")
h_map = {"Трейд (1–5 дней)":"short", "Среднесрок (1–4 недели)":"mid", "Долгосрок (1–6 месяцев)":"long"}
horizon = h_map[st.selectbox("Горизонт", list(h_map.keys()), index=1)]

if st.button("Сформировать инвест-мемо"):
    try:
        price, df = latest_price(ticker)
        dec = analyze_ticker(df, ticker, horizon)
        st.markdown(f"#### 🎯 Asset: {ticker.upper()}")
        st.caption(f"Горизонт: {st.session_state.get('horizon_label', '—')}  |  Текущая цена: ${price:,.2f}")
        st.markdown("### 🧠 Core Recommendation")
        st.write(humanize(dec))
        st.info("Текст намеренно «человеческий» и не раскрывает внутреннюю математику/уровни.")
    except Exception as e:
        st.error(f"Ошибка формирования мемо: {e}")
