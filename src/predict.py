"""Lógica de predicción: forecast a futuro y backtest sobre datos históricos."""

from __future__ import annotations

import pandas as pd

from .data import INTERVAL_FREQ

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def future_timestamps(last_ts: pd.Timestamp, interval: str, n: int) -> pd.Series:
    """Genera n timestamps futuros con la frecuencia del intervalo.

    Nota: usa frecuencia fija de calendario. Para mercados 24/7 (cripto) es
    exacto; para acciones con cierres (noches/fines de semana) los
    timestamps son una aproximación, una limitación conocida del enfoque.
    """
    freq = INTERVAL_FREQ[interval]
    idx = pd.date_range(start=last_ts, periods=n + 1, freq=freq)[1:]
    return pd.Series(idx, name="timestamps")


def _slice_ohlcv(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df.iloc[start:end][OHLCV_COLUMNS].reset_index(drop=True)


def forecast(
    predictor,
    df: pd.DataFrame,
    interval: str,
    pred_len: int = 120,
    lookback: int = 400,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Predice las próximas ``pred_len`` velas usando las últimas ``lookback``.

    ``df`` debe venir de :func:`src.data.download_ohlcv` (columnas
    ``timestamps`` + OHLCV). Devuelve el DataFrame de predicción indexado
    por los timestamps futuros.
    """
    if lookback > predictor.max_context:
        raise ValueError(
            f"lookback ({lookback}) supera el contexto máximo del modelo "
            f"({predictor.max_context})."
        )
    if len(df) < lookback:
        raise ValueError(
            f"Datos insuficientes: {len(df)} velas disponibles, "
            f"se necesitan {lookback} para el lookback."
        )

    hist = df.iloc[-lookback:].reset_index(drop=True)
    x_df = hist[OHLCV_COLUMNS]
    x_timestamp = hist["timestamps"]
    y_timestamp = future_timestamps(hist["timestamps"].iloc[-1], interval, pred_len)

    return predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=verbose,
    )


def backtest(
    predictor,
    df: pd.DataFrame,
    interval: str,
    pred_len: int = 120,
    lookback: int = 400,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice una ventana histórica conocida para comparar con la realidad.

    Usa las velas ``[-lookback-pred_len : -pred_len]`` como contexto y
    predice las últimas ``pred_len`` velas, que se devuelven junto a las
    velas reales para evaluación.

    Devuelve ``(pred_df, actual_df)`` con índices alineados por timestamp.
    """
    if len(df) < lookback + pred_len:
        raise ValueError(
            f"Datos insuficientes para backtest: {len(df)} velas, se "
            f"necesitan lookback+pred_len = {lookback + pred_len}."
        )

    context_end = len(df) - pred_len
    context_start = context_end - lookback

    x_df = _slice_ohlcv(df, context_start, context_end)
    x_timestamp = df.iloc[context_start:context_end]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[context_end:]["timestamps"].reset_index(drop=True)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=verbose,
    )

    actual_df = df.iloc[context_end:][["timestamps"] + OHLCV_COLUMNS].reset_index(drop=True)
    pred_df = pred_df.copy()
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    actual_df.index = pd.DatetimeIndex(actual_df["timestamps"])
    return pred_df, actual_df
