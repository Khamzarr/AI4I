# AI4I Predictive Maintenance Platform

Industrial Analytics Platform для предиктивного обслуживания оборудования.

## Модель
- Algorithm: CatBoost with Feature Engineering
- ROC-AUC: 0.9832
- Recall: 85% (work mode) / 76% (strict mode)
- Precision: 88% (strict mode)

## Структура
- `ai4i2020.csv` — датасет
- `api_flask.py` — REST API (Flask)
- `dashboard.py` — Streamlit дашборд
- `ai4i_catboost_fe_pipeline.pkl` — сохранённая модель

## Запуск

```bash
# Terminal 1: API
python api_flask.py

# Terminal 2: Dashboard
streamlit run dashboard.py