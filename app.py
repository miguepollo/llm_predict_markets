"""App Streamlit: predicción de precios con Kronos + Yahoo Finance.

Ejecutar con:  uv run streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backtest import compute_metrics
from src.data import (
    INTERVAL_FREQ,
    MAX_PERIOD_BY_INTERVAL,
    VALID_PERIODS,
    download_ohlcv,
)
from src.models import DEFAULT_MODEL, MODEL_REGISTRY, load_predictor
from src.plotting import backtest_figure, forecast_figure
from src.predict import backtest, forecast

st.set_page_config(page_title="Kronos Price Predictor", layout="wide")
st.title("📈 Kronos Price Predictor")
st.caption(
    "Predicción de velas OHLCV con el foundation model [Kronos]"
    "(https://github.com/shiyu-coder/Kronos) y datos de Yahoo Finance. "
    "Solo con fines de investigación — no es consejo financiero."
)


@st.cache_resource(show_spinner=False)
def get_predictor(model_name: str):
    return load_predictor(model_name, device="cpu")


@st.cache_data(ttl=900, show_spinner=False)
def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    return download_ohlcv(ticker, interval, period)


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Datos")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="BTC-USD",
                           help="Ej.: BTC-USD, AAPL, ^GSPC, EURUSD=X")
    interval = st.selectbox("Timeframe", list(INTERVAL_FREQ), index=7)
    max_period = MAX_PERIOD_BY_INTERVAL.get(interval, "max")
    # VALID_PERIODS está ordenado de menor a mayor duración
    max_idx = VALID_PERIODS.index(max_period) if max_period in VALID_PERIODS else len(VALID_PERIODS) - 1
    period_options = VALID_PERIODS[: max_idx + 1]
    period = st.selectbox("Periodo histórico", period_options,
                          index=len(period_options) - 1)

    st.header("Modelo")
    model_name = st.selectbox(
        "Modelo Kronos",
        list(MODEL_REGISTRY),
        index=list(MODEL_REGISTRY).index(DEFAULT_MODEL),
        format_func=lambda n: f"{n} ({MODEL_REGISTRY[n].params})",
    )
    cfg = MODEL_REGISTRY[model_name]
    st.caption(cfg.description)

    mode = st.radio("Modo", ["Forecast", "Backtest"], horizontal=True)

    lookback_default = min(400, cfg.max_context)
    lookback = st.slider("Lookback (velas de contexto)", 64, cfg.max_context,
                         lookback_default, step=32)
    if mode == "Forecast":
        pred_len = st.slider("Velas a predecir", 8, 240, 120, step=8)
    else:
        pred_len = st.slider("Velas de backtest", 8, 240, 60, step=8)

    st.header("Muestreo")
    temperature = st.slider("Temperatura (T)", 0.1, 2.0, 1.0, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    sample_count = st.slider("Nº de muestras", 1, 5, 1)

    run = st.button("🚀 Predecir", type="primary", width="stretch")

# -------------------------------------------------------------------- main ---
try:
    with st.spinner(f"Descargando {ticker} ({interval}, {period})…"):
        df = get_data(ticker, interval, period)
except Exception as e:
    st.error(f"Error descargando datos: {e}")
    st.stop()

needed = lookback + (pred_len if mode == "Backtest" else 0)
st.info(
    f"**{ticker}** · {interval} · {len(df)} velas descargadas "
    f"({df['timestamps'].iloc[0]:%Y-%m-%d %H:%M} → {df['timestamps'].iloc[-1]:%Y-%m-%d %H:%M} UTC) · "
    f"último cierre: **{df['close'].iloc[-1]:.4g}**"
)
if len(df) < needed:
    st.warning(
        f"Se necesitan ≥{needed} velas para lookback={lookback}"
        + (" + backtest" if mode == "Backtest" else "")
        + f" y solo hay {len(df)}. Amplía el periodo o reduce el lookback."
    )
    st.stop()

if not run:
    st.plotly_chart(forecast_figure(df.tail(min(lookback, len(df))), df.tail(0),
                                    title=f"{ticker} · histórico"),
                    width="stretch")
    st.stop()

with st.spinner(f"Cargando modelo Kronos-{model_name} (primera vez descarga de HuggingFace)…"):
    try:
        predictor = get_predictor(model_name)
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        st.stop()

if mode == "Forecast":
    with st.spinner(f"Generando predicción de {pred_len} velas…"):
        try:
            pred_df = forecast(
                predictor, df, interval,
                pred_len=pred_len, lookback=lookback,
                temperature=temperature, top_p=top_p, sample_count=sample_count,
            )
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.stop()

    st.plotly_chart(
        forecast_figure(df.tail(lookback), pred_df,
                        title=f"{ticker} · {interval} · Kronos-{model_name}"),
        width="stretch",
    )
    st.subheader("Velas predichas")
    st.dataframe(pred_df.round(4), width="stretch")
    st.download_button(
        "⬇️ Descargar predicción (CSV)",
        pred_df.to_csv().encode(),
        file_name=f"{ticker}_{interval}_forecast_{model_name}.csv",
        mime="text/csv",
    )

else:  # Backtest
    with st.spinner(f"Ejecutando backtest ({pred_len} velas)…"):
        try:
            pred_df, actual_df = backtest(
                predictor, df, interval,
                pred_len=pred_len, lookback=lookback,
                temperature=temperature, top_p=top_p, sample_count=sample_count,
            )
        except Exception as e:
            st.error(f"Error en el backtest: {e}")
            st.stop()

    context_df = df.iloc[-lookback - pred_len:-pred_len]
    prev_close = float(context_df["close"].iloc[-1])
    metrics = compute_metrics(actual_df["close"], pred_df["close"], prev_close=prev_close)

    cols = st.columns(4)
    cols[0].metric("MAE", f"{metrics['mae']:.4g}")
    cols[1].metric("RMSE", f"{metrics['rmse']:.4g}")
    cols[2].metric("MAPE", f"{metrics['mape']:.2f}%")
    cols[3].metric("Acierto direccional", f"{metrics['directional_accuracy']:.1f}%")

    st.plotly_chart(
        backtest_figure(context_df, actual_df, pred_df,
                        title=f"{ticker} · {interval} · backtest Kronos-{model_name}"),
        width="stretch",
    )

    comparison = pd.DataFrame({
        "timestamp": actual_df.index,
        "close_real": actual_df["close"].to_numpy(),
        "close_predicho": pred_df["close"].to_numpy(),
    })
    comparison["error_%"] = (comparison["close_predicho"] - comparison["close_real"]) / comparison["close_real"] * 100
    st.subheader("Real vs predicho (cierre)")
    st.dataframe(comparison.round(4), width="stretch")
    st.download_button(
        "⬇️ Descargar comparación (CSV)",
        comparison.to_csv(index=False).encode(),
        file_name=f"{ticker}_{interval}_backtest_{model_name}.csv",
        mime="text/csv",
    )
