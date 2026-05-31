"""SARIMA control model using statsmodels SARIMAX.

This is NOT introduced to inflate the baseline count. Its sole purpose is to act
as a *control* for the plain ARIMA baseline: ARIMA(2,1,2) scores poorly on the
hourly temperature series, and the open question is whether that is because it
lacks an explicit seasonal component. SARIMA keeps the same non-seasonal order
(2,1,2) and adds a daily seasonal order (1,0,1,24), so any improvement is
attributable to the seasonal terms alone.

We reuse the ARIMA batch rolling-origin scheme (refit every ``refit_every``
origins, cached fit in between). Because a seasonal s=24 SARIMAX fit is roughly
an order of magnitude slower than ARIMA, we fit on a shorter recent window
(default 2000 hours) and the runner evaluates a small test subset. The seasonal
period is fixed to 24 (daily cycle on hourly data).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class SarimaConfig:
    order: tuple[int, int, int] = (2, 1, 2)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24)
    horizon: int = 24
    refit_every: int = 50  # in number of origins
    window: int = 2000  # recent history used for each fit (seasonal fits are slow)


def rolling_sarima_predictions(
    train: np.ndarray,
    test: np.ndarray,
    config: SarimaConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin SARIMA on the test split.

    Mirrors ``rolling_arima_predictions``: for each origin we either refit on a
    recent window or reuse the cached fit and read the appropriate slice of its
    multi-step forecast. Returns flattened (y_true, y_pred).
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
            window_size = min(config.window, origin)
            history = full[origin - window_size : origin]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cached_fit = SARIMAX(
                        history,
                        order=config.order,
                        seasonal_order=config.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
                cached_origin = origin
            except Exception as exc:  # numerical issues on a window
                print(f"  SARIMA refit failed at origin {origin}: {exc}")
                continue

        steps_ahead = origin - cached_origin
        forecast_len = steps_ahead + horizon
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = cached_fit.forecast(steps=forecast_len)
        except Exception as exc:
            print(f"  SARIMA forecast failed at origin {origin}: {exc}")
            continue
        y_pred = np.asarray(forecast)[steps_ahead : steps_ahead + horizon]
        y_true = full[origin : origin + horizon]

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)
