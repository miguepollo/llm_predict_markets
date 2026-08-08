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
- Runs on **CPU** by default; **Intel GPU (XPU)** supported with a torch XPU
  build (see below). Selectable in the sidebar ("Compute device").

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

Startup and model-loading logs (including the selected inference device and
driver details) are printed to the console where Streamlit runs, e.g.:

```
INFO src.models: torch 2.13.0+cpu | inference device: cpu -> CPU (4 torch threads)
INFO src.models: Kronos-small ready on cpu in 7.9s (max_context=512)
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
| **Sample count** | Number of independent forecast paths generated and averaged into the final prediction (1–20). More samples = smoother, more stable results, at linear CPU cost (20 samples ≈ 20× the compute of a single one). |

## Hardware acceleration: Intel GPU (XPU) and NPU

The app picks the compute device from the sidebar ("Compute device"). The list
is auto-detected from torch: `cpu` always; `cuda`, `xpu` (Intel GPU) or `mps`
(Apple) appear only when available.

### Intel GPU (Arc, Iris Xe, Core Ultra iGPU) — XPU backend

PyTorch ships an XPU backend for Intel GPUs. To enable it:

1. **Install the Intel GPU compute runtime** (Ubuntu example):
   ```bash
   sudo apt install intel-opencl-icd libze1
   # For newer Arc/Core Ultra systems, follow:
   # https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/
   ```
2. **Switch torch from the CPU build to the XPU build** in `pyproject.toml`:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-xpu"
   url = "https://download.pytorch.org/whl/xpu"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-xpu" }
   ```
   Then reinstall:
   ```bash
   uv sync --python 3.11
   ```
3. **(Optional)** Install Intel Extension for PyTorch for extra optimized ops:
   ```bash
   uv pip install intel-extension-for-pytorch
   ```
4. **Verify** the GPU is visible and run the app:
   ```bash
   uv run python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
   uv run streamlit run app.py   # select "xpu" in the sidebar
   ```

Notes:
- No changes to the app code are needed: `KronosPredictor` receives the device
  explicitly and moves model + tensors with `.to(device)`.
- Some ops may fall back to CPU with a warning on older iGPUs; Arc discrete
  GPUs have the best XPU coverage.
- Kronos' own device auto-detection only knows cuda/mps/cpu, which is why this
  project always passes the device explicitly.

### Intel NPU (AI Boost, Core Ultra) — not plug-and-play

PyTorch has **no native NPU backend**; Intel NPUs are accessed through
**OpenVINO**. Kronos is a custom autoregressive model (Python sampling loop
with dynamic shapes), so it does not run on the NPU out of the box. A real
integration would require:

1. Install the NPU driver (`intel_vpu`, included in recent Linux kernels) and
   [OpenVINO](https://docs.openvino.ai/) with NPU plugin:
   ```bash
   uv pip install openvino
   ```
2. Export the Kronos tokenizer and transformer to OpenVINO IR
   (`optimum-intel` / `ovc`), or wrap single-step forwards with
   `torch.compile(model, backend="openvino")`.
3. Re-implement the autoregressive generation loop
   (`vendor/Kronos/model/kronos.py::auto_regressive_inference`) around the
   exported models.

This is a significant engineering effort and is **not currently supported** by
this app. For faster inference today, use an Intel GPU via XPU (above) or the
`mini` model on CPU.

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
