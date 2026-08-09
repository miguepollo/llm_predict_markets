"""Unit tests for the model registry and the TimesFM/Moirai/Kronos predictors.

These tests neither download weights nor hit the network: TimesFM is exercised
through a fake model, Kronos through a fake vendored predictor, and the
registry is checked for all three backends.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_REGISTRY, KronosPredictor, TimesFMPredictor

OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _frame(n: int = 96) -> pd.DataFrame:
    """Synthetic OHLCV context (no timestamps column, like ``predict`` input)."""
    base = np.arange(n) + 100.0
    return pd.DataFrame({
        "open": base,
        "high": base + 4.0,
        "low": base - 4.0,
        "close": base + 1.0,
        "volume": np.full(n, 1000.0),
        "amount": np.full(n, 1e5),
    })


class _FakeModel:
    """Mimics ``TimesFM_2p5_200M_torch.forecast`` returning (5, horizon)."""

    def forecast(self, horizon, inputs):
        k = len(inputs)
        pred = np.stack([
            inputs[i][-1] * (1 + 0.01 * np.arange(horizon)) for i in range(k)
        ])
        return pred, np.ones((k, horizon, 10))


def test_timesfm_predictor_candles():
    p = TimesFMPredictor(_FakeModel(), max_context=1024)
    x_timestamp = pd.date_range("2024-01-01", periods=96, freq="1h")
    y_timestamp = pd.date_range("2024-01-05", periods=12, freq="1h")
    out = p.predict(_frame(96), x_timestamp, y_timestamp, pred_len=12)

    assert list(out.columns) == OHLCV_COLS
    assert len(out) == 12
    assert isinstance(out.index, pd.DatetimeIndex)
    # Candle geometry is reconciled.
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["volume"] >= 0).all()


def test_registry_has_all_backends():
    backends = {cfg.backend for cfg in MODEL_REGISTRY.values()}
    assert backends == {"timesfm", "moirai", "kronos"}
    assert "2.5" in MODEL_REGISTRY
    assert "moirai-small" in MODEL_REGISTRY
    assert "moirai-base" in MODEL_REGISTRY
    assert "kronos-mini" in MODEL_REGISTRY
    assert "kronos-small" in MODEL_REGISTRY
    assert "kronos-base" in MODEL_REGISTRY
    assert MODEL_REGISTRY["2.5"].backend == "timesfm"
    assert MODEL_REGISTRY["kronos-small"].backend == "kronos"


class _FakeKronosPredictor:
    """Mimics the vendored ``model.KronosPredictor`` (returns raw candles)."""

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_p,
                sample_count, verbose):
        # Deliberately broken geometry so the wrapper must reconcile it.
        open_ = df["open"].to_numpy()[-1] + np.arange(pred_len)
        idx = pd.DatetimeIndex(y_timestamp)
        return pd.DataFrame({
            "open": open_,
            "high": open_ - 10,          # too low  -> pushed up to open/close
            "low": open_ + 10,           # too high -> pushed down to open/close
            "close": open_,
            "volume": np.full(pred_len, -5.0),  # negative -> clamped to 0
        }, index=idx)


def test_kronos_predictor_reconciles_candles():
    p = KronosPredictor(_FakeKronosPredictor(), max_context=512)
    x_timestamp = pd.date_range("2024-01-01", periods=8, freq="1h")
    y_timestamp = pd.date_range("2024-01-02", periods=5, freq="1h")
    out = p.predict(_frame(8), x_timestamp, y_timestamp, pred_len=5,
                    temperature=1.0, top_p=0.9, sample_count=3)

    assert list(out.columns) == OHLCV_COLS
    assert len(out) == 5
    assert isinstance(out.index, pd.DatetimeIndex)
    # Candle geometry is reconciled.
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["volume"] >= 0).all()


def test_load_predictor_unknown_raises():
    from src.models import load_predictor

    with pytest.raises(ValueError):
        load_predictor("definitely-not-a-model")