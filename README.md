# Kronos Price Predictor

App local de predicción de precios (velas OHLCV) usando el foundation model
[**Kronos**](https://github.com/shiyu-coder/Kronos) (MIT) y datos de
**Yahoo Finance**. Solo con fines de investigación — **no es consejo financiero**.

## Características

- Descarga de series OHLCV de Yahoo Finance por **ticker** y **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Modelo configurable**: Kronos `mini` (4.1M, ctx 2048), `small` (24.7M, ctx 512)
  o `base` (102M, ctx 512). Los pesos se descargan de HuggingFace en la primera ejecución.
- Modo **Forecast**: predice las próximas N velas a partir del contexto reciente.
- Modo **Backtest**: predice una ventana histórica conocida y compara con la realidad
  (MAE, RMSE, MAPE, acierto direccional, gráfico real vs predicho).
- Gráficos de velas interactivos (Plotly) y exportación a CSV.
- Funciona en **CPU** (no requiere GPU).

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) (gestiona Python 3.11 automáticamente):

```bash
# 1. Vendorizar Kronos (no existe paquete oficial en PyPI)
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
# Versión verificada de este proyecto:
git -C vendor/Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a

# 2. Instalar dependencias (torch CPU-only incluido)
uv sync --python 3.11
```

## Uso

```bash
uv run streamlit run app.py
```

Abre http://localhost:8501, elige ticker/timeframe/modelo y pulsa **Predecir**.

## Tests

```bash
uv run pytest
```

## Estructura

```
app.py              # UI Streamlit
src/
  data.py           # yfinance -> DataFrame OHLCV normalizado (formato Kronos)
  models.py         # registro de modelos (mini/small/base) + carga cacheada
  predict.py        # forecast y backtest sobre KronosPredictor
  backtest.py       # métricas: MAE/RMSE/MAPE/acierto direccional
  plotting.py       # gráficos de velas Plotly
tests/              # tests unitarios (sin red ni modelo)
vendor/Kronos/      # repo de Kronos (commit pineado, ver arriba)
```

## Notas y limitaciones

- **Ningún modelo predice el mercado de forma fiable**; los backtests sirven para
  evaluar la calidad del forecast en cada ticker/timeframe concreto.
- Los timestamps futuros usan frecuencia fija: exacto para cripto (24/7),
  aproximado para acciones (noches/fines de semana).
- yfinance es una API no oficial: historia intradía limitada (p. ej. 1m ≈ 7 días,
  1h ≈ 2 años) y posibles rate-limits.
- En CPU: `mini` y `small` responden en segundos; `base` puede tardar minutos
  según `pred_len`.
