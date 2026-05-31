"""Error-analysis primitives.

Pure functions that slice forecast error along dimensions that matter for the
deterministic-vs-probabilistic story:

  - per forecast-horizon step (does error grow with lead time?),
  - by meteorological season and by temperature range (where does each model
    struggle?),
  - worst-window grouping under anomaly injection (which contamination hurts
    most, and for whom?), and
  - overconfident-failure analysis for probabilistic models (intervals that are
    narrow yet miss the truth -- the dangerous failure mode).

The runner (scripts/run_error_analysis.py) trains the models, builds the
timestamp bookkeeping and calls these functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import mae, rmse

# DJF / MAM / JJA / SON
_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
TEMP_EDGES = (-np.inf, 0.0, 10.0, 20.0, np.inf)
TEMP_LABELS = ("<0", "0-10", "10-20", ">20")


def per_horizon_point(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE/MAE at each horizon step. y arrays are (N, H)."""
    h = y_true.shape[1]
    return {
        "rmse": [rmse(y_true[:, k], y_pred[:, k]) for k in range(h)],
        "mae": [mae(y_true[:, k], y_pred[:, k]) for k in range(h)],
    }


def target_months(test_index: pd.DatetimeIndex, n_windows: int, lookback: int,
                  horizon: int, stride: int) -> np.ndarray:
    """Month label for every (window, horizon) target point. Shape (N, H)."""
    months = np.empty((n_windows, horizon), dtype=int)
    idx_month = test_index.month.to_numpy()
    for i in range(n_windows):
        base = i * stride + lookback
        months[i] = idx_month[base : base + horizon]
    return months


def breakdown(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict:
    """RMSE/MAE/count per label group (all arrays flattened to 1-D)."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    labels = labels.reshape(-1)
    out = {}
    for g in pd.unique(labels):
        m = labels == g
        out[str(g)] = {
            "rmse": rmse(y_true[m], y_pred[m]),
            "mae": mae(y_true[m], y_pred[m]),
            "count": int(m.sum()),
        }
    return out


def season_breakdown(y_true: np.ndarray, y_pred: np.ndarray, months: np.ndarray) -> dict:
    seasons = np.vectorize(_SEASON.get)(months)
    return breakdown(y_true, y_pred, seasons)


def temperature_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = np.asarray(TEMP_LABELS)[np.digitize(y_true.reshape(-1), TEMP_EDGES[1:-1])]
    return breakdown(y_true, y_pred, labels.reshape(y_true.shape))


def overconfident_failures(
    y_true: np.ndarray, q_preds: np.ndarray, quantiles: np.ndarray, alpha: float = 0.1
) -> dict:
    """Quantify intervals that are narrow yet miss the truth.

    A point is *missed* when the truth falls outside the central (1-alpha)
    interval. A miss is *overconfident* when its interval is also narrower than
    the median interval width across all points. Returns rates and the width
    contrast between missed and covered points.
    """
    quantiles = np.asarray(quantiles)
    lo = int(np.argmin(np.abs(quantiles - alpha / 2.0)))
    hi = int(np.argmin(np.abs(quantiles - (1.0 - alpha / 2.0))))
    y = np.asarray(y_true).reshape(-1)
    q = np.asarray(q_preds).reshape(-1, len(quantiles))
    lower, upper = q[:, lo], q[:, hi]
    width = upper - lower
    median_width = float(np.median(width))

    missed = (y < lower) | (y > upper)
    narrow = width < median_width
    overconf = missed & narrow

    return {
        "alpha": alpha,
        "miss_rate": float(missed.mean()),
        "overconfident_miss_rate": float(overconf.mean()),
        "overconfident_share_of_misses": float(overconf.sum() / max(missed.sum(), 1)),
        "median_interval_width": median_width,
        "mean_width_missed": float(width[missed].mean()) if missed.any() else 0.0,
        "mean_width_covered": float(width[~missed].mean()) if (~missed).any() else 0.0,
    }


def window_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-window RMSE over the horizon. Arrays (N, H) -> (N,)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def worst_window_summary(window_errors: np.ndarray, worst_frac: float = 0.1) -> dict:
    """Distribution stats of per-window error, emphasising the worst tail."""
    e = np.asarray(window_errors)
    k = max(1, int(len(e) * worst_frac))
    worst = np.sort(e)[-k:]
    return {
        "mean": float(e.mean()),
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
        "worst_decile_mean": float(worst.mean()),  # mean of the worst worst_frac (default 0.1 -> decile)
        "n_windows": int(len(e)),
    }
