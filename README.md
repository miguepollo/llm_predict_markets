# TimesFM Price Predictor

Local price prediction app (OHLCV candles) using Google's
[**TimesFM**](https://github.com/google-research/timesfm) time-series
foundation model (Apache-2.0) and **Yahoo Finance** data. For research purposes
only — **not financial advice**.

## Features

- Downloads OHLCV series from Yahoo Finance by **ticker** and **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Configurable model**: TimesFM 2.5 (`200M` params, up to 16k context). Weights
  are downloaded from HuggingFace on first run.
- **Forecast mode**: predicts the next N candles from recent context.
- **Backtest mode**: predicts a known historical window and compares it against
  reality (MAE, RMSE, MAPE, directional accuracy, actual vs predicted chart).
- Interactive candlestick charts (Plotly) and CSV export.
- Runs on **CPU** by default; **NVIDIA GPU (CUDA)** supported if torch is built
  with CUDA. Selectable in the sidebar ("Compute device").

## How prediction works

TimesFM is a **univariate point-forecast** foundation model. Its `forecast()`
method predicts several series in a single batched call, so this app forecasts
the five OHLCV series (**open, high, low, close, volume**) independently in one
forward pass. Each predicted candle is then reconciled so the geometry is
consistent: `high = max(hi, open, close)` and `low = min(lo, open, close)`, and
volume is clamped at `>= 0`. `amount = volume * mean price`.

## Installation

Requires [uv](https://docs.astral.sh/uv/) (manages Python 3.11 automatically):

```bash
uv sync --python 3.11   # installs CPU-only torch + timesfm from PyPI
```

## Usage

```bash
uv run streamlit run app.py
```

Open http://localhost:8501, pick ticker/timeframe/model and press **Predict**.

Startup and model-loading logs (including the selected inference device) are
printed to the console where Streamlit runs, e.g.:

```
INFO src.models: torch 2.13.0+cpu | inference device: cpu -> CPU (4 torch threads)
INFO src.models: TimesFM-2.5 ready on cpu in 7.9s (max_context=1024, max_horizon=256)
```

## Parameters

All parameters are set in the sidebar:

### Data

| Parameter | What it does |
|---|---|
| **Ticker** | Yahoo Finance symbol. Stocks (`AAPL`, `TSLA`), indices (`^GSPC`), forex (`EURUSD=X`), crypto (`BTC-USD`)… |
| **Timeframe** | Candle interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk. Determines the frequency of both the history and the predicted candles. |
| **Historical period** | How much history to download (1d … max). Limited by the timeframe: Yahoo only keeps ~7 days of 1m data, ~1 month of 5m–30m, ~2 years of 1h. |

### Model

| Parameter | What it does |
|---|---|
| **TimesFM model** | Which pre-trained model to use. `2.5` is the current release (200M, context 1024 configured for CPU-friendly speed). |
| **Compute device** | `cpu` or `cuda`. TimesFM 2.5 torch only accelerates on CUDA; XPU/MPS fall back to CPU. |
| **Mode** | `Forecast`: predicts the next N candles into the future. `Backtest`: predicts a known historical window and compares it against reality with metrics (MAE, RMSE, MAPE, directional accuracy) — use this to judge whether the model works for your ticker/timeframe before trusting a forecast. |
| **Lookback** | Number of past candles fed to the model as context. TimesFM patches the series in windows of 32, so the slider uses multiples of 32 (64–1024). More context = more information, but slower. |
| **Candles to predict / Backtest candles** | Prediction horizon (`pred_len`, max 240): how many candles the model generates. Longer horizons are slower and less reliable. |

## Hardware acceleration

The app picks the compute device from the sidebar ("Compute device"). The list
is auto-detected from torch and limited to `cpu` and `cuda`, because TimesFM's
torch inference only runs natively on those.

- To use CUDA, install a CUDA-enabled torch build instead of the default CPU one
  (e.g. change the `pytorch-cpu` index in `pyproject.toml` to a CUDA wheel) and
  re-run `uv sync`.
- Intel GPU (XPU), MPS and Intel NPU are **not** used by TimesFM in this app.

## Tests

```bash
uv run pytest
```

## Project layout

```
app.py              # Streamlit UI
src/
  data.py           # yfinance -> normalized OHLCV DataFrame
  models.py         # TimesFM registry (2.5) + cached loading + TimesFMPredictor
  predict.py        # forecast and backtest on top of a predictor
  backtest.py       # metrics: MAE/RMSE/MAPE/directional accuracy
  plotting.py       # Plotly candlestick charts
tests/              # unit tests (no network, no model)
```

## Notes and limitations

- **No model predicts the market reliably**; backtests are meant to assess
  forecast quality for each specific ticker/timeframe.
- TimesFM predicts **open, high, low, close** and volume independently (each in
  its own univariate forecast within one batched call); the candle geometry is
  then reconciled so `high`/`low` always enclose the `open`/`close` body.
- Future timestamps use a fixed frequency: exact for crypto (24/7),
  approximate for stocks (nights/weekends).
- yfinance is an unofficial API: limited intraday history (e.g. 1m ≈ 7 days,
  1h ≈ 2 years) and possible rate limits.
- On CPU, forecasting with context 1024 returns in seconds-to-tens-of-seconds
  depending on `pred_len`.