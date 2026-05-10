"""Naive seasonal baseline: y_hat[t+h] = y[t+h-S].

For hourly weather data the natural daily seasonality is S=24. Despite its
simplicity, this baseline is notoriously hard to beat on temperature series
and is the standard sanity check in forecasting benchmarks.
"""
from __future__ import annotations

import numpy as np


def naive_seasonal_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int = 24,
) -> np.ndarray:
    if len(history) < season_length:
        raise ValueError("History shorter than one seasonal cycle.")
    template = history[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(template, reps)[:horizon]


def rolling_naive_predictions(
    series: np.ndarray,
    horizon: int,
    season_length: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin naive seasonal forecast over a 1-D series.

    Returns (y_true, y_pred) flattened, where each origin produces `horizon`
    predictions starting at `season_length` (so we always have a seasonal
    template available).
    """
    n = len(series)
    starts = range(season_length, n - horizon + 1)
    y_true_list, y_pred_list = [], []
    for s in starts:
        history = series[:s]
        y_pred = naive_seasonal_forecast(history, horizon, season_length)
        y_true = series[s : s + horizon]
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)
