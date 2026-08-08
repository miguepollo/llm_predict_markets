# AGENTS.md

## Context

Local price prediction app: **Kronos** (open-source foundation model for
financial K-lines, vendored in `vendor/Kronos`) + **Yahoo Finance**
(yfinance) + **Streamlit** + **Plotly**. Research only, no real trading.

## Environment and commands

- Package manager: **uv** with **Python 3.11** (the system has 3.14; torch
  comes from the CPU-only index defined in `pyproject.toml`).
- Install: `uv sync --python 3.11`
- Run the app: `uv run streamlit run app.py`
- Tests: `uv run pytest` (unit tests, no network or model needed; ~1s)

## Design decisions

- **Kronos is vendored** in `vendor/Kronos` (no official PyPI package;
  the `kronos` package on PyPI is an unrelated Django cron app). Pinned
  commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Imported via
  `sys.path` in `src/models.py`.
- **Models** (`src/models.py`, `MODEL_REGISTRY`): `mini` (4.1M, ctx 2048,
  tokenizer `Kronos-Tokenizer-2k`), `small` (24.7M, ctx 512, default) and
  `base` (102.3M, ctx 512) with `NeoQuasar/*` tokenizers from HuggingFace.
  `Kronos-large` is not published.
- **Devices**: CPU by default. The sidebar device selector is auto-detected
  via `available_devices()` in `src/models.py` (cpu/cuda/xpu/mps) and passed
  explicitly to `KronosPredictor` (its own auto-detection ignores XPU).
  Intel GPU needs the torch XPU build; Intel NPU is not supported (would
  require OpenVINO export + custom generation loop). See README
  "Hardware acceleration".
- **Logging**: `logging` INFO messages at startup/model init (`src/models.py`,
  `app.py`) report the torch build, the selected device and driver details
  (`device_details()`).
- **Data format**: `src/data.py` normalizes yfinance to
  `timestamps, open, high, low, close, volume, amount` (tz-naive UTC, no
  NaN). `amount = volume * mean price` if missing. Kronos requires lowercase
  OHLC columns and fills volume/amount if missing.
- **Future timestamps**: fixed frequency per interval (`INTERVAL_FREQ` in
  `src/data.py`). Documented approximation for markets with trading halts.
- **Streamlit caches**: `@st.cache_resource` for the predictor (HF weights),
  `@st.cache_data(ttl=900)` for downloads.

## Style

- Code and comments in English, simple and modular (src/ by layers).
- Do not add heavy dependencies without a reason; matplotlib is not used (Plotly).
- When changing the `src/` interface, update `app.py`, tests and this file.
