# Kronos Price Predictor

Local price prediction app (OHLCV candles) using the
[**Kronos**](https://github.com/shiyu-coder/Kronos) foundation model (MIT)
and **Yahoo Finance** data. For research purposes only — **not financial advice**.

## Features

- Downloads OHLCV series from Yahoo Finance by **ticker** and **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Configurable model**: Kronos `mini` (4.1M, ctx 2048), `small` (24.7M, ctx 512)
  or `base` (102M, ctx 512). Weights are downloaded from HuggingFace on first run.
- **Forecast mode**: predicts the next N candles from recent context.
- **Backtest mode**: predicts a known historical window and compares it against
  reality (MAE, RMSE, MAPE, directional accuracy, actual vs predicted chart).
- Interactive candlestick charts (Plotly) and CSV export.
- Runs on **CPU** (no GPU required).

## Installation

Requires [uv](https://docs.astral.sh/uv/) (manages Python 3.11 automatically):

```bash
# 1. Vendor Kronos (no official PyPI package exists)
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
# Version verified by this project:
git -C vendor/Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a

# 2. Install dependencies (CPU-only torch included)
uv sync --python 3.11
```

## Usage

```bash
uv run streamlit run app.py
```

Open http://localhost:8501, pick ticker/timeframe/model and press **Predict**.

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
| **Kronos model** | Which pre-trained model to use. `mini` (4.1M params, context 2048: fastest, best for CPU), `small` (24.7M, context 512: default balance), `base` (102M, context 512: best quality, slow on CPU). |
| **Mode** | `Forecast`: predicts the next N candles into the future. `Backtest`: predicts a known historical window and compares it against reality with metrics (MAE, RMSE, MAPE, directional accuracy) — use this to judge whether the model works for your ticker/timeframe before trusting a forecast. |
| **Lookback** | Number of past candles fed to the model as context. More context = more information, but slower. Capped by the model's context length (512 for small/base, 2048 for mini). |
| **Candles to predict / Backtest candles** | Prediction horizon (`pred_len`): how many candles the model generates. Longer horizons are slower and less reliable. |

### Sampling

Kronos is a generative model: each forecast is a sample from a probability
distribution. These parameters control the sampling process:

| Parameter | What it does |
|---|---|
| **Temperature (T)** | Randomness of the sampling (0.1–2.0). Lower = more conservative, closer to the most likely path. Higher = more diverse and volatile paths. Values around 1.0 are a good default. |
| **Top-p** | Nucleus sampling threshold (0.1–1.0). Only tokens within the top cumulative probability `p` are considered. Lower = safer, less diverse predictions; 0.9 is a good default. |
| **Sample count** | Number of independent forecast paths generated and averaged into the final prediction (1–5). More samples = smoother, more stable results, at linear CPU cost. |

## Tests

```bash
uv run pytest
```

## Project layout

```
app.py              # Streamlit UI
src/
  data.py           # yfinance -> normalized OHLCV DataFrame (Kronos format)
  models.py         # model registry (mini/small/base) + cached loading
  predict.py        # forecast and backtest on top of KronosPredictor
  backtest.py       # metrics: MAE/RMSE/MAPE/directional accuracy
  plotting.py       # Plotly candlestick charts
tests/              # unit tests (no network, no model)
vendor/Kronos/      # Kronos repo (pinned commit, see above)
```

## Notes and limitations

- **No model predicts the market reliably**; backtests are meant to assess
  forecast quality for each specific ticker/timeframe.
- Future timestamps use a fixed frequency: exact for crypto (24/7),
  approximate for stocks (nights/weekends).
- yfinance is an unofficial API: limited intraday history (e.g. 1m ≈ 7 days,
  1h ≈ 2 years) and possible rate limits.
- On CPU: `mini` and `small` respond in seconds; `base` can take minutes
  depending on `pred_len`.
