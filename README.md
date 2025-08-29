# CapinteL‑Q — Streamlit (Polygon Edition)

Мини‐MVP на Streamlit, который реализует стратегию:
- Многопериодные Fibonacci‑пивоты: weekly/monthly/yearly в зависимости от горизонта.
- Heikin Ashi: серия цвета и смена.
- MACD (гистограмма): стрик и «замедление».
- RSI (вспомогательный фильтр).
- ATR: масштабирование стопов/целей.
- Текст — «живой», без раскрытия внутренних уровней.

## Быстрый старт локально
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export POLYGON_API_KEY=...your key...
streamlit run app.py
```

## Деплой на Streamlit Cloud
1. Загрузите файлы репозитория.
2. Укажите переменную **POLYGON_API_KEY** в Secrets.
3. Entry‑point: `app.py`.
