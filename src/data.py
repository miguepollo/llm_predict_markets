"""Descarga de series OHLCV desde Yahoo Finance y normalización al formato Kronos.

Kronos espera un DataFrame con columnas ``open, high, low, close`` y
opcionalmente ``volume`` y ``amount``, más una Serie de timestamps.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# Intervalos soportados (yfinance) -> frecuencia pandas equivalente
INTERVAL_FREQ: dict[str, str] = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1h",
    "90m": "90min",
    "1h": "1h",
    "1d": "1D",
    "5d": "5D",
    "1wk": "7D",
}

# Periodos que ofrece yfinance
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

# Limitaciones conocidas de Yahoo: datos intradía con historia limitada
MAX_PERIOD_BY_INTERVAL = {
    "1m": "5d",
    "2m": "1mo",
    "5m": "1mo",
    "15m": "1mo",
    "30m": "1mo",
    "60m": "2y",
    "90m": "1mo",
    "1h": "2y",
}

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def download_ohlcv(ticker: str, interval: str = "1h", period: str = "1mo") -> pd.DataFrame:
    """Descarga velas de Yahoo Finance y las normaliza al formato Kronos.

    Devuelve un DataFrame con columnas
    ``timestamps, open, high, low, close, volume, amount`` ordenado por
    tiempo ascendente, sin NaNs y con timestamps tz-naive (UTC).
    """
    if interval not in INTERVAL_FREQ:
        raise ValueError(
            f"Intervalo no soportado: {interval!r}. "
            f"Usa uno de: {sorted(INTERVAL_FREQ)}"
        )

    raw = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(
            f"Yahoo Finance no devolvió datos para {ticker!r} "
            f"(interval={interval}, period={period}). "
            "Comprueba el ticker o amplía el periodo."
        )

    df = _normalize(raw)
    if len(df) < 2:
        raise ValueError(f"Datos insuficientes para {ticker!r}: {len(df)} velas.")
    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame crudo de yfinance al esquema Kronos."""
    df = raw.copy()

    # yfinance devuelve columnas MultiIndex si se descargan varios tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLC en los datos descargados: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    # amount = volumen * precio medio aproximado (Kronos lo usa como feature opcional)
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    # Timestamps: índice tz-naive en UTC para evitar problemas con .dt y torch
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df.index.name = "timestamps"

    df = df[["open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df.reset_index()
