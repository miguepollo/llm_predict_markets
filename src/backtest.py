"""Evaluation metrics for backtest mode."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    actual_close: pd.Series,
    pred_close: pd.Series,
    prev_close: float | None = None,
) -> dict[str, float]:
    """Computes error metrics and directional accuracy on the close price.

    - MAE / RMSE / MAPE: error on the close value.
    - directional_accuracy: % of steps where the sign of the predicted
      change matches the real one. If ``prev_close`` (last close of the
      context) is given, the first step is evaluated too.
    """
    actual = np.asarray(actual_close, dtype=float)
    pred = np.asarray(pred_close, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("actual and pred must have the same length.")
    if len(actual) == 0:
        raise ValueError("Empty series.")

    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")

    # Directional accuracy: sign of the change vs. the previous step
    if prev_close is not None:
        actual_with_prev = np.concatenate([[prev_close], actual])
        pred_with_prev = np.concatenate([[prev_close], pred])
    else:
        actual_with_prev = actual
        pred_with_prev = pred
    actual_dir = np.sign(np.diff(actual_with_prev))
    pred_dir = np.sign(np.diff(pred_with_prev))
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }
