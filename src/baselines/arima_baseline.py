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


@dataclass
class ArimaConfig:
    order: tuple[int, int, int] = (2, 1, 2)
    horizon: int = 24
    refit_every: int = 50  # in number of origins
    window: int = 8000  # recent history (hours) used for each fit (~11 months)


def rolling_arima_predictions(
    train: np.ndarray,
    test: np.ndarray,
    config: ArimaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin ARIMA on the test split (batched refit).

    We do NOT refit at every origin. Every ``refit_every`` origins we refit on
    the most recent window of context (train ++ test[:t]); for the origins in
    between we reuse that cached fit and read the matching slice of its
    multi-step forecast (steps_ahead .. steps_ahead + horizon). Each origin is
    scored against test[t : t + horizon].

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
            # Cap the fit history at the most recent `config.window` hours
            # (~11 months at the default): long enough to capture seasonal
            # structure, short enough to keep each ARIMA refit fast.
            window_size = min(config.window, origin)
            history = full[origin - window_size : origin]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cached_fit = ARIMA(history, order=config.order).fit()
                cached_origin = origin
            except Exception as exc:  # numerical issues on a window
                print(f"  ARIMA refit failed at origin {origin}: {exc}")
                continue

        steps_ahead = origin - cached_origin
        forecast_len = steps_ahead + horizon
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = cached_fit.forecast(steps=forecast_len)
        except Exception as exc:
            print(f"  ARIMA forecast failed at origin {origin}: {exc}")
            continue
        y_pred = np.asarray(forecast)[steps_ahead : steps_ahead + horizon]
        y_true = full[origin : origin + horizon]

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)
