# AGENTS.md

## Context

Local price prediction app: two interchangeable time-series foundation models —
**TimesFM** (Google, PyPI `timesfm`) and **Moirai** (Salesforce, via the `uni2ts`
package) + **Yahoo Finance** (yfinance) + **Streamlit** + **Plotly**. Research
only, no real trading.

## Environment and commands

- Package manager: **uv** with **Python 3.11** (the system has 3.14; torch
  comes from the CPU-only index defined in `pyproject.toml`).
- Install: `uv sync --python 3.11` (installs `timesfm` + `uni2ts`; `uni2ts`
  pins torch <2.5, numpy 1.26 and einops 0.7, shared with TimesFM).
- Run the app: `uv run streamlit run app.py`
- Tests: `uv run pytest` (unit tests, no network or model needed; ~1s)

## Design decisions

- **Two backends** in `src/models.py`, selected via `MODEL_REGISTRY` and a
  `backend` field ("timesfm" | "moirai"); `load_predictor()` dispatches on it.
- **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`, 200M), Apache-2.0. It is a
  **univariate point-forecast** model, but its `forecast()` accepts several
  series in one batched call: `TimesFMPredictor` forecasts **open, high, low,
  close, volume** as five independent series in a single forward pass.
- **Moirai 1.1-R** (`Salesforce/moirai-1.1-R-{small,base,large}`), CC-BY-NC-4.0.
  A **truly multivariate probabilistic** model: `MoiraiPredictor` feeds all five
  OHLCV variates as one GluonTS series (`target_dim=5`, via
  `PandasDataset(df, target=OHLCV)`) and forecasts them jointly; the median over
  `num_samples` trajectories is the point forecast.
- **Shared candle reconciliation**: every predictor ensures
  `high >= max(open,close)`, `low <= min(open,close)`, `volume >= 0` and
  `amount = volume * mean price`, returning a DataFrame indexed by future
  timestamps with columns `open, high, low, close, volume, amount`.
- **Model registry** (`src/models.py`): `2.5` (timesfm, max_context=1024),
  `moirai-small`/`base`/`large` (max_context=512, num_samples 20/10/10). TimesFM
  patches context in windows of 32, so the lookback slider uses multiples of 32.
- **Devices**: CPU by default; CUDA if available. Both models' torch inference
  only accelerates on CUDA, so `available_devices()` returns only cpu/cuda.
  `device_details()` still knows all names.
- **Logging**: `logging` INFO messages at startup/model init (`src/models.py`,
  `app.py`) report the torch build and the selected device/model.
- **Data format**: `src/data.py` normalizes yfinance to
  `timestamps, open, high, low, close, volume, amount` (tz-naive UTC, no NaN).
  `amount = volume * mean price` if missing.
- **Future timestamps**: fixed frequency per interval (`INTERVAL_FREQ` in
  `src/data.py`). Documented approximation for markets with trading halts.
- **Streamlit caches**: `@st.cache_resource` for the predictor (HF weights),
  `@st.cache_data(ttl=900)` for downloads.

## Style

- Code and comments in English, simple and modular (src/ by layers).
- Do not add heavy dependencies without a reason; matplotlib is not used (Plotly).
- When changing the `src/` interface, update `app.py`, tests and this file.