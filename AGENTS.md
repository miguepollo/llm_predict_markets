# AGENTS.md

## Contexto

App local de predicción de precios: **Kronos** (foundation model open-source de
K-lines financieras, vendorizado en `vendor/Kronos`) + **Yahoo Finance**
(yfinance) + **Streamlit** + **Plotly**. Solo investigación, no trading real.

## Entorno y comandos

- Gestor de paquetes: **uv** con **Python 3.11** (el sistema tiene 3.14; torch
  va por índice CPU-only definido en `pyproject.toml`).
- Instalar: `uv sync --python 3.11`
- Ejecutar app: `uv run streamlit run app.py`
- Tests: `uv run pytest` (unitarios, sin red ni modelo; ~1s)

## Decisiones de diseño

- **Kronos se vendoriza** en `vendor/Kronos` (no hay paquete oficial en PyPI;
  el paquete `kronos` de PyPI es un cron de Django sin relación). Commit
  pineado: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Se importa vía
  `sys.path` en `src/models.py`.
- **Modelos** (`src/models.py`, `MODEL_REGISTRY`): `mini` (4.1M, ctx 2048,
  tokenizer `Kronos-Tokenizer-2k`), `small` (24.7M, ctx 512, default) y
  `base` (102.3M, ctx 512) con tokenizers `NeoQuasar/*` de HuggingFace.
  `Kronos-large` no está publicado.
- **Sin GPU**: todo corre en CPU. `base` es lento; ofrecerlo pero avisar.
- **Formato de datos**: `src/data.py` normaliza yfinance a
  `timestamps, open, high, low, close, volume, amount` (tz-naive UTC, sin
  NaN). `amount = volume * precio medio` si falta. Kronos exige columnas
  lowercase OHLC + rellena volume/amount si faltan.
- **Timestamps futuros**: frecuencia fija por intervalo (`INTERVAL_FREQ` en
  `src/data.py`). Aproximación documentada para mercados con cierres.
- **Cachés de Streamlit**: `@st.cache_resource` para el predictor (pesos HF),
  `@st.cache_data(ttl=900)` para las descargas.

## Estilo

- Código y comentarios en español, simple y modular (src/ por capas).
- No añadir dependencias pesadas sin motivo; matplotlib no se usa (Plotly).
- Al cambiar interfaz de `src/`, actualizar `app.py`, tests y este archivo.
