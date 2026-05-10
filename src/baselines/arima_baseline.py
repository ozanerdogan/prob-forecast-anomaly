"""ARIMA baseline using statsmodels.

For Phase 1 we use a fixed (p, d, q) order. Order selection via auto-arima or
information criteria search is deferred to later phases. We fit on the training
split and refit on a moving window when generating rolling forecasts on test.
Refitting on every step is prohibitively slow on hundreds of windows, so we
adopt a *batch* rolling-origin scheme: refit every `refit_every` origins and
use the cached fit in between.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


@dataclass
class ArimaConfig:
    order: tuple[int, int, int] = (2, 1, 2)
    horizon: int = 24
    refit_every: int = 50  # in number of origins


def rolling_arima_predictions(
    train: np.ndarray,
    test: np.ndarray,
    config: ArimaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin ARIMA on the test split.

    For each origin t in test:
      - context = train ++ test[:t]
      - fit ARIMA on a recent window of `context`
      - forecast `horizon` steps and compare to test[t : t + horizon]

    Returns flattened (y_true, y_pred).
    """
    horizon = config.horizon
    n = len(test) - horizon + 1
    if n <= 0:
        raise ValueError("Test split shorter than one horizon.")

    full = np.concatenate([train, test])
    train_end = len(train)

    y_true_list, y_pred_list = [], []
    cached_fit = None
    cached_origin = -1

    for k, t in enumerate(range(n)):
        origin = train_end + t
        if cached_fit is None or (k % config.refit_every == 0):
            window_size = min(8000, origin)
            history = full[origin - window_size : origin]
            try:
                cached_fit = ARIMA(history, order=config.order).fit()
                cached_origin = origin
            except Exception as exc:  # numerical issues on a window
                print(f"  ARIMA refit failed at origin {origin}: {exc}")
                continue

        steps_ahead = origin - cached_origin
        forecast_len = steps_ahead + horizon
        try:
            forecast = cached_fit.forecast(steps=forecast_len)
        except Exception as exc:
            print(f"  ARIMA forecast failed at origin {origin}: {exc}")
            continue
        y_pred = np.asarray(forecast)[steps_ahead : steps_ahead + horizon]
        y_true = full[origin : origin + horizon]

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)
