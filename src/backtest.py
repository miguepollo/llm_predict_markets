"""Métricas de evaluación para el modo backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    actual_close: pd.Series,
    pred_close: pd.Series,
    prev_close: float | None = None,
) -> dict[str, float]:
    """Calcula métricas de error y acierto direccional sobre el precio de cierre.

    - MAE / RMSE / MAPE: error sobre el valor de cierre.
    - directional_accuracy: % de pasos en los que el signo del cambio
      predicho coincide con el real. Si se pasa ``prev_close`` (último
      cierre del contexto), el primer paso también se evalúa.
    """
    actual = np.asarray(actual_close, dtype=float)
    pred = np.asarray(pred_close, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("actual y pred deben tener la misma longitud.")
    if len(actual) == 0:
        raise ValueError("Series vacías.")

    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")

    # Acierto direccional: signo del cambio respecto al paso anterior
    if prev_close is not None:
        actual_with_prev = np.concatenate([[prev_close], actual])
        pred_with_prev = np.concatenate([[prev_close], pred])
    else:
        actual_with_prev = actual
        pred_with_prev = pred
    actual_dir = np.sign(np.diff(actual_with_prev))
    pred_dir = np.sign(np.diff(pred_with_prev))
    if prev_close is None:
        # Sin prev_close el primer paso no tiene dirección definida
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) else float("nan")
    else:
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
    }
