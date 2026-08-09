# TimesFM & Moirai Price Predictor

Local price prediction app (OHLCV candles) using Google's
[**TimesFM**](https://github.com/google-research/timesfm) (Apache-2.0) or
Salesforce's [**Moirai**](https://github.com/SalesforceAIResearch/uni2ts)
(CC-BY-NC-4.0) plus **Yahoo Finance** data. For research purposes only —
**not financial advice**.

## Features

- Downloads OHLCV series from Yahoo Finance by **ticker** and **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Two interchangeable foundation models** (sidebar):
  - **TimesFM 2.5** — `200M` params, point forecast, Apache-2.0.
  - **Moirai 1.1-R** — `small`/`base`/`large` (14M/91M/311M), probabilistic and
    truly **multivariate** (forecasts all OHLCV variates together).
- **Forecast mode**: predicts the next N candles from recent context.
- **Backtest mode**: predicts a known historical window and compares it against
  reality (MAE, RMSE, MAPE, directional accuracy, actual vs predicted chart).
- Interactive candlestick charts (Plotly) and CSV export.
- Runs on **CPU** by default; **NVIDIA GPU (CUDA)** supported if torch is built
  with CUDA. Selectable in the sidebar ("Compute device").

## How prediction works

Both models feed the five OHLCV series (**open, high, low, close, volume**) and
return a forecast for each candle:

- **TimesFM** is a univariate point-forecast model; its `forecast()` accepts
  several series in one batched call, so the five series are forecast
  independently in a single forward pass.
- **Moirai** is a multivariate probabilistic transformer; it forecasts all five
  variates **jointly** (one series with `target_dim = 5`), so cross-series
  correlation is modeled. The median over sampled trajectories is used as the
  point forecast.

Either way each predicted candle is reconciled so the geometry is consistent:
`high = max(hi, open, close)`, `low = min(lo, open, close)`, and
`volume = max(vol, 0)`. `amount = volume * mean price`.

## Installation

Requires [uv](https://docs.astral.sh/uv/) (manages Python 3.11 automatically).
Installation also pulls in `uni2ts` (Moirai), which pins **torch <2.5**, numpy
1.26 and einops 0.7; this is shared with TimesFM.

```bash
uv sync --python 3.11   # CPU-only torch + timesfm + uni2ts from PyPI
```

## Usage

```bash
uv run streamlit run app.py
```

Open http://localhost:8501, pick ticker/timeframe/model and press **Predict**.
The first run of each model downloads its weights from HuggingFace.

Startup and model-loading logs (including the selected inference device) are
printed to the console where Streamlit runs, e.g.:

```
INFO src.models: torch 2.4.1+cpu | inference device: cpu -> CPU (8 torch threads)
INFO src.models: moirai-moirai-base ready on cpu in 7.9s (max_context=512)
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
| **Foundation model** | `TimesFM-2.5` (point forecast), `Moirai small/base/large` (probabilistic + true multivariate). Default: `moirai-base`. |
| **Compute device** | `cpu` or `cuda`. Both models only accelerate on CUDA; XPU/MPS fall back to CPU. |
| **Mode** | `Forecast`: predicts the next N candles into the future. `Backtest`: predicts a known historical window and compares it against reality with metrics (MAE, RMSE, MAPE, directional accuracy) — use this to judge whether the model works for your ticker/timeframe before trusting a forecast. |
| **Lookback** | Number of past candles fed as context (multiples of 32; up to 1024 for TimesFM, 512 for Moirai). More context = more information, but slower. |
| **Candles to predict / Backtest candles** | Prediction horizon (`pred_len`, max 240): how many candles the model generates. Longer horizons are slower and less reliable. |

## Hardware acceleration

The app picks the compute device from the sidebar ("Compute device"). The list
is auto-detected from torch and limited to `cpu` and `cuda`, because the models'
torch inference only runs natively on those.

By default `pyproject.toml` pins a **CPU-only** torch build:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

To accelerate inference on an **NVIDIA GPU**, replace that CPU wheel with a CUDA
build. Pick **one** of the two options below, then re-verify:

```bash
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

You should see your CUDA version (e.g. `cu124`) and `True`.

### Option A — persistent (recommended)

Change the `url` in `pyproject.toml` to the CUDA wheel index. Choose the CUDA
version that matches your driver (see `nvidia-smi`); `cu124` is a safe default
for recent NVIDIA drivers and works with the torch `<2.5` pin used here:

```toml
# Instead of https://download.pytorch.org/whl/cpu
url = "https://download.pytorch.org/whl/cu124"
```

Then recreate the environment (this recompiles torch):

```bash
uv sync --python 3.11 --reinstall-package torch
```

### Option B — one-off (no file changes)

Install just torch from the CUDA index with `uv pip install`:

```bash
uv pip install --python 3.11 torch --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** this only swaps the current environment; the next `uv sync` will
> restore the CPU-only build defined in `pyproject.toml` unless you also apply
> Option A (or `uv sync` again with `--reinstall-package torch`).

### Which CUDA wheel to pick

| PyTorch wheel | Min. NVIDIA driver (Windows) | Notes |
|---|---|---|
| `cu118` | ~450.80 | oldest; for old drivers |
| `cu121` | ~525.60 | |
| `cu124` | ~550.54 | recommended default |

Your concrete CUDA architecture/driver is shown by `nvidia-smi`. Both models
(TimesFM and Moirai) then run inference on the GPU and the sidebar will offer
`cuda`.

- Intel GPU (XPU), MPS and Intel NPU are **not** used by these models.

## Tests

```bash
uv run pytest
```

## Project layout

```
app.py              # Streamlit UI
src/
  data.py           # yfinance -> normalized OHLCV DataFrame
  models.py         # registry (timesfm + moirai), caching, predictors
  predict.py        # forecast and backtest on top of a predictor
  backtest.py       # metrics: MAE/RMSE/MAPE/directional accuracy
  plotting.py       # Plotly candlestick charts
tests/              # unit tests (no network, no model)
```

## Notes and limitations

- **No model predicts the market reliably**; backtests are meant to assess
  forecast quality for each specific ticker/timeframe.
- Moirai weights are licensed **CC-BY-NC-4.0** (non-commercial); TimesFM is
  Apache-2.0.
- `open/high/low/close/volume` are reconciled so `high`/`low` always enclose the
  `open`/`close` body. TimesFM forecasts each series independently; Moirai
  forecasts them jointly.
- Future timestamps use a fixed frequency: exact for crypto (24/7),
  approximate for stocks (nights/weekends).
- yfinance is an unofficial API: limited intraday history (e.g. 1m ≈ 7 days,
  1h ≈ 2 years) and possible rate limits.
- On CPU, forecasting returns in seconds-to-tens-of-seconds depending on the
  model, context and `pred_len`.