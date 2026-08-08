import numpy as np
import pandas as pd
import pytest

from src.data import _normalize
from src.predict import future_timestamps


def _make_raw(n=10, tz_aware=True, with_volume=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    if tz_aware:
        idx = idx.tz_localize("America/New_York")
    data = {
        "Open": np.linspace(100, 110, n),
        "High": np.linspace(101, 111, n),
        "Low": np.linspace(99, 109, n),
        "Close": np.linspace(100.5, 110.5, n),
    }
    if with_volume:
        data["Volume"] = np.arange(n) * 1000.0
    return pd.DataFrame(data, index=idx)


def test_normalize_basic():
    df = _normalize(_make_raw())
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 10
    # timestamps tz-naive
    assert df["timestamps"].dt.tz is None
    # amount = volume * mean price
    expected = df["volume"] * (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    pd.testing.assert_series_equal(df["amount"], expected, check_names=False)


def test_normalize_without_volume():
    df = _normalize(_make_raw(with_volume=False))
    assert (df["volume"] == 0.0).all()
    assert (df["amount"] == 0.0).all()


def test_normalize_drops_nan_and_sorts():
    raw = _make_raw()
    raw.iloc[3, 0] = np.nan  # Open NaN
    raw = raw.iloc[::-1]     # reversed order
    df = _normalize(raw)
    assert len(df) == 9
    assert df["timestamps"].is_monotonic_increasing


def test_normalize_multiindex_columns():
    raw = _make_raw()
    raw.columns = pd.MultiIndex.from_tuples([(c, "TEST") for c in raw.columns])
    df = _normalize(raw)
    assert "open" in df.columns


def test_normalize_missing_ohlc_raises():
    raw = _make_raw().drop(columns=["High"])
    with pytest.raises(ValueError, match="Missing OHLC columns"):
        _normalize(raw)


def test_future_timestamps():
    ts = future_timestamps(pd.Timestamp("2024-01-01 00:00"), "1h", 3)
    assert len(ts) == 3
    assert list(ts) == list(pd.date_range("2024-01-01 01:00", periods=3, freq="1h"))
    ts = future_timestamps(pd.Timestamp("2024-01-01"), "1d", 2)
    assert list(ts) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
