import numpy as np
import pandas as pd
import pytest

from src.backtest import compute_metrics


def test_perfect_prediction():
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(actual, actual.copy(), prev_close=0.5)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0
    assert m["directional_accuracy"] == 100.0


def test_known_errors():
    actual = pd.Series([10.0, 20.0])
    pred = pd.Series([12.0, 18.0])
    m = compute_metrics(actual, pred)
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(np.sqrt(4.0))
    assert m["mape"] == pytest.approx((2 / 10 + 2 / 20) / 2 * 100)


def test_directional_accuracy():
    # Real: sube, baja, sube. Predicho: sube, sube, sube -> 2/3
    actual = pd.Series([1.0, 2.0, 1.5, 2.5])
    pred = pd.Series([1.0, 2.0, 2.2, 2.6])
    m = compute_metrics(actual, pred)
    assert m["directional_accuracy"] == pytest.approx(2 / 3 * 100)


def test_directional_with_prev_close():
    # Con prev_close el primer paso cuenta: real baja (1.0 < 2.0), pred sube
    actual = pd.Series([1.0, 2.0])
    pred = pd.Series([3.0, 4.0])
    m = compute_metrics(actual, pred, prev_close=2.0)
    assert m["directional_accuracy"] == pytest.approx(50.0)


def test_mape_ignores_zeros():
    actual = pd.Series([0.0, 10.0])
    pred = pd.Series([5.0, 11.0])
    m = compute_metrics(actual, pred)
    assert m["mape"] == pytest.approx(10.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(pd.Series([1.0]), pd.Series([1.0, 2.0]))


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
