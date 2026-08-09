# AGENTS.md

## Context

Local price prediction app: **TimesFM** (Google's time-series foundation model,
from PyPI `timesfm`) + **Yahoo Finance** (yfinance) + **Streamlit** + **Plotly**.
Research only, no real trading.

## Environment and commands

- Package manager: **uv** with **Python 3.11** (the system has 3.14; torch
  comes from the CPU-only index defined in `pyproject.toml`).
- Install: `uv sync --python 3.11`
- Run the app: `uv run streamlit run app.py`
- Tests: `uv run pytest` (unit tests, no network or model needed; ~1s)

## Design decisions

- **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, 200M params) is loaded
  from HuggingFace via the `timesfm` PyPI package in `src/models.py`. It is a
  **univariate point-forecast** model, but `forecast()` accepts several series
  in one batched call: `TimesFMPredictor` forecasts **open, high, low, close,
  volume** independently in a single forward pass and reconciles each candle
  (`high >= max(open,close)`, `low <= min(open,close)`, `volume >= 0`) so the
  candlestick geometry stays consistent.
- **Models** (`src/models.py`, `MODEL_REGISTRY`): one entry, `2.5`
  (max_context=1024, max_horizon=256). Context is patched in windows of 32, so
  the lookback slider uses multiples of 32.
- **Devices**: CPU by default; CUDA if available. TimesFM 2.5's torch inference
  only accelerates on CUDA, so `available_devices()` returns only cpu/cuda
  (no XPU/MPS, unlike the old Kronos backend). `device_details()` still knows
  all names.
- **Logging**: `logging` INFO messages at startup/model init (`src/models.py`,
  `app.py`) report the torch build and the selected device.
- **Data format**: `src/data.py` normalizes yfinance to
  `timestamps, open, high, low, close, volume, amount` (tz-naive UTC, no NaN).
  `amount = volume * mean price` if missing. `close` is fed to TimesFM.
- **Future timestamps**: fixed frequency per interval (`INTERVAL_FREQ` in
  `src/data.py`). Documented approximation for markets with trading halts.
- **Streamlit caches**: `@st.cache_resource` for the predictor (HF weights),
  `@st.cache_data(ttl=900)` for downloads.

## Style

- Code and comments in English, simple and modular (src/ by layers).
- Do not add heavy dependencies without a reason; matplotlib is not used (Plotly).
- When changing the `src/` interface, update `app.py`, tests and this file.