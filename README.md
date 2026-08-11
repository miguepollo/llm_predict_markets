# TimesFM · Moirai · Kronos · Chronos-2 Price Predictor

Local price prediction app (OHLCV candles) using Google's
[**TimesFM**](https://github.com/google-research/timesfm) (Apache-2.0),
Salesforce's [**Moirai**](https://github.com/SalesforceAIResearch/uni2ts)
(CC-BY-NC-4.0), [**Kronos**](https://github.com/shiyu-coder/Kronos)
(Apache-2.0) or Amazon's [**Chronos-2**](https://github.com/amazon-science/chronos-forecasting)
(Apache-2.0) plus **Yahoo Finance** data. For research purposes only —
**not financial advice**.

## Features

- Downloads OHLCV series from Yahoo Finance by **ticker** and **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Four interchangeable foundation models** (sidebar):
  - **TimesFM 2.5** — `200M` params, point forecast, Apache-2.0.
  - **Moirai 1.1-R** — `small`/`base`/`large` (14M/91M/311M), probabilistic and
    truly **multivariate** (forecasts all OHLCV variates together).
  - **Kronos** — `mini`/`small`/`base` (4.1M/24.7M/102.3M), **generative** with
    sampling controls (temperature, top-p, sample count), Apache-2.0.
  - **Chronos-2** — `120M`, **quantile-based** (Amazon), univariate and
    multivariate, Apache-2.0.
- **Forecast mode**: predicts the next N candles from recent context.
- **Backtest mode**: predicts a known historical window and compares it against
  reality (MAE, RMSE, MAPE, directional accuracy, actual vs predicted chart).
- Interactive candlestick charts (Plotly) and CSV export.
- Runs on **CPU** by default; **NVIDIA GPU (CUDA)** supported if torch is built
  for the GPU. Selectable in the sidebar ("Compute device").

## How prediction works

All four models feed the five OHLCV series (**open, high, low, close, volume**)
and return a forecast for each candle:

- **TimesFM** is a univariate point-forecast model; its `forecast()` accepts
  several series in one batched call, so the five series are forecast
  independently in a single forward pass.
- **Moirai** is a multivariate probabilistic transformer; it forecasts all five
  variates **jointly** (one series with `target_dim = 5`), so cross-series
  correlation is modeled. The median over sampled trajectories is used as the
  point forecast.
- **Kronos** is a generative token-based model (vendored under `vendor/Kronos`).
  It samples candle sequences from the next-token distribution — controlled by
  **temperature**, **top-p** and **sample count** — and averages the sampled
  paths into the point forecast.
- **Chronos-2** is Amazon's quantile-based encoder-only model. Like Moirai, it
  forecasts all five OHLCV variates **jointly**; the 0.5 quantile is used as
  the point forecast.

Either way each predicted candle is reconciled so the geometry is consistent:
`high = max(hi, open, close)`, `low = min(lo, open, close)`, and
`volume = max(vol, 0)`. `amount = volume * mean price`.

## Installation

Requires [uv](https://docs.astral.sh/uv/) (manages Python 3.11 automatically).
Installation also pulls in `uni2ts` (Moirai), which pins **torch <2.5**, numpy
1.26 and einops 0.7; this is shared with the other backends. `chronos-forecasting`
(Chronos-2) brings `transformers`, which is pinned `<5` because v5 needs
torch >=2.5.

```bash
uv sync --python 3.11   # CPU-only torch + timesfm + uni2ts + chronos-forecasting

# Kronos has no PyPI package; clone it once into vendor/ (gitignored):
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
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
| **Foundation model** | `TimesFM-2.5` (point forecast), `Moirai small/base/large` (probabilistic + true multivariate), `Kronos mini/small/base` (generative) or `Chronos-2` (quantile-based, Amazon). Default: `moirai-base`. |
| **Compute device** | `cpu` or `cuda`. `cuda` targets an NVIDIA (CUDA) GPU — TimesFM/Moirai/Chronos-2 accelerate on it; XPU/MPS fall back to CPU (Kronos can use them). |
| **Mode** | `Forecast`: predicts the next N candles into the future. `Backtest`: predicts a known historical window and compares it against reality with metrics (MAE, RMSE, MAPE, directional accuracy) — use this to judge whether the model works for your ticker/timeframe before trusting a forecast. |
| **Lookback** | Number of past candles fed as context (multiples of 32; up to 1024 for TimesFM, 512 for Moirai, 2048 for Kronos-mini/Chronos-2). More context = more information, but slower. |
| **Candles to predict / Backtest candles** | Prediction horizon (`pred_len`, max 240): how many candles the model generates. Longer horizons are slower and less reliable. |
| **Temperature (T)** | *(Kronos only)* Sampling temperature: `1.0` is neutral, lower = more conservative, higher = more varied paths. |
| **Top-p** | *(Kronos only)* Nucleus sampling threshold: probability mass kept at each token step. |
| **Sample count** | *(Kronos only)* Forecast paths generated and averaged. More samples = smoother prediction, proportionally slower. |

## Hardware acceleration

The app picks the compute device from the sidebar ("Compute device"). The list
is auto-detected from torch and limited to `cpu` and `cuda`, because the models'
torch inference only runs natively on those. `cuda` runs on an NVIDIA GPU when
torch is built with CUDA.

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

Your concrete CUDA architecture/driver is shown by `nvidia-smi`. TimesFM, Moirai
and Chronos-2 then run inference on the GPU and the sidebar will offer `cuda`
(Kronos can also use Intel GPU / Apple Silicon, if torch is built for those).

## Tests

```bash
uv run pytest
```

## Project layout

```
app.py              # Streamlit UI
src/
  data.py           # yfinance -> normalized OHLCV DataFrame
  models.py         # registry (timesfm + moirai + kronos + chronos2), caching, predictors
  predict.py        # forecast and backtest on top of a predictor
  backtest.py       # metrics: MAE/RMSE/MAPE/directional accuracy
  plotting.py       # Plotly candlestick charts
vendor/Kronos       # vendored Kronos repo (gitignored, clone manually)
tests/              # unit tests (no network, no model)
```

## Notes and limitations

- **No model predicts the market reliably**; backtests are meant to assess
  forecast quality for each specific ticker/timeframe.
- Moirai weights are licensed **CC-BY-NC-4.0** (non-commercial); TimesFM,
  Kronos and Chronos-2 are Apache-2.0.
- `open/high/low/close/volume` are reconciled so `high`/`low` always enclose the
  `open`/`close` body. TimesFM forecasts each series independently; Moirai and
  Chronos-2 forecast them jointly; Kronos samples them as a sequence.
- Future timestamps use a fixed frequency: exact for crypto (24/7),
  approximate for stocks (nights/weekends).
- yfinance is an unofficial API: limited intraday history (e.g. 1m ≈ 7 days,
  1h ≈ 2 years) and possible rate limits.
- On CPU, forecasting returns in seconds-to-tens-of-seconds depending on the
  model, context and `pred_len`.