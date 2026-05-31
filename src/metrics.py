"""Forecasting metrics.

Phase 1 (deterministic): RMSE, MAE, MAPE.
Phase 2 additions:
  - Relative-error: sMAPE, MASE (added alongside MAPE; MAPE is unchanged).
  - Probabilistic: CRPS (ensemble / Gaussian / quantile), pinball loss,
    PICP (prediction-interval coverage probability), MIS (mean interval score).

The Phase-1 ``report`` is intentionally left untouched so existing baseline
JSON outputs stay byte-for-byte comparable. Probabilistic models use the new
``report_probabilistic`` aggregator.

The runner pipeline scores CRPS via the quantile approximation
(``crps_from_quantiles`` = 2 * mean pinball), so DeepAR and the Transformer are
compared on equal footing; ``crps_ensemble`` (sample-based) and ``crps_gaussian``
(closed-form) are provided for reference and unit tests, not for the headline
numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    """Classic Mean Absolute Percentage Error, eps-floored to avoid blow-up.

    The denominator is ``max(|y_true|, eps)`` (the true value, not a symmetric
    average), so this is the standard MAPE with a small-denominator guard rather
    than a symmetric measure. For the symmetric variant see ``smape``.
    """
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Relative-error metrics (added for Phase 2; MAPE above is unchanged).
# --------------------------------------------------------------------------- #
def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE in percent: 100 * mean(2|y-yhat| / (|y|+|yhat|)).

    Bounded in [0, 200]; the ``eps`` guards the all-zero point.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season: int = 24,
) -> float:
    """Mean Absolute Scaled Error.

    Scales forecast MAE by the in-sample MAE of a seasonal-naive forecast on the
    training series (season=24 for hourly data with daily seasonality). A value
    < 1 means the model beats seasonal-naive in-sample scaling.
    """
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= season:
        raise ValueError("Training series shorter than one seasonal cycle.")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale <= 0:
        raise ValueError("Seasonal-naive scale is zero; cannot compute MASE.")
    return float(mae(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)) / scale)


# --------------------------------------------------------------------------- #
# Probabilistic metrics.
# --------------------------------------------------------------------------- #
def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level ``q`` in (0, 1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_q = np.asarray(y_pred_q, dtype=float)
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def mean_pinball_loss(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Average pinball loss over a set of quantile levels.

    ``q_preds`` has shape (..., Q) with the last axis aligned to ``quantiles``;
    ``y_true`` broadcasts against ``q_preds[..., 0]``.
    """
    y_true = np.asarray(y_true, dtype=float)
    q_preds = np.asarray(q_preds, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    losses = [pinball_loss(y_true, q_preds[..., i], float(quantiles[i])) for i in range(len(quantiles))]
    return float(np.mean(losses))


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> float:
    """CRPS estimated from forecast samples (energy form).

    CRPS = E|X - y| - 0.5 * E|X - X'|, averaged over all points.
    ``samples`` has shape (N, S); ``y_true`` has shape (N,).
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Expected samples of shape (N, S).")
    n_samples = samples.shape[1]
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    # Mean pairwise absolute difference via sorting: avoids the O(S^2) matrix.
    s_sorted = np.sort(samples, axis=1)
    weights = (2 * np.arange(1, n_samples + 1) - n_samples - 1)
    term2 = (2.0 / (n_samples * n_samples)) * np.sum(weights * s_sorted, axis=1)
    return float(np.mean(term1 - 0.5 * term2))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_from_quantiles(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Approximate CRPS as 2x the average pinball loss over a quantile grid.

    Exact identity: CRPS = 2 * integral_0^1 pinball_q dq. Here the integral is
    approximated by the *unweighted mean* of the pinball losses at the given
    levels, which equals the integral only for a dense, evenly spaced grid. The
    grid used in this project (QUANTILES_7 = 0.05,0.1,0.25,0.5,0.75,0.9,0.95) is
    neither dense nor evenly spaced, so the returned value is a coarse CRPS
    *proxy* rather than an accurate absolute CRPS -- but it is applied identically
    to both probabilistic models, so DeepAR-vs-Transformer comparisons stay fair.
    Lets us score quantile models that do not expose an explicit predictive
    distribution.
    """
    return 2.0 * mean_pinball_loss(y_true, q_preds, quantiles)


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability: fraction of y inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width (sharpness)."""
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def mis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Mean Interval Score for a central (1-alpha) prediction interval.

    MIS = (u - l) + (2/alpha)(l - y)1[y<l] + (2/alpha)(y - u)1[y>u].
    Lower is better; rewards both narrow intervals and correct coverage.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(width + below + above))


def report_probabilistic(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Aggregate probabilistic scores from a quantile forecast.

    ``q_preds`` shape (N, Q) aligned with ``quantiles``. The central
    (1-alpha) interval is taken from the quantile levels nearest to alpha/2 and
    1-alpha/2 (exact when those levels are present).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    q_preds = np.asarray(q_preds, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)

    lo_level, hi_level = alpha / 2.0, 1.0 - alpha / 2.0
    lo_idx = int(np.argmin(np.abs(quantiles - lo_level)))
    hi_idx = int(np.argmin(np.abs(quantiles - hi_level)))
    lower = q_preds[:, lo_idx]
    upper = q_preds[:, hi_idx]

    out = {
        "crps": crps_from_quantiles(y_true, q_preds, quantiles),
        "pinball": mean_pinball_loss(y_true, q_preds, quantiles),
        "picp": picp(y_true, lower, upper),
        "mpiw": mpiw(lower, upper),
        "mis": mis(y_true, lower, upper, alpha),
        "picp_alpha": float(1.0 - alpha),  # nominal target coverage (1 - alpha), not alpha
    }
    # Point metrics off the median (or nearest-to-0.5) quantile.
    med_idx = int(np.argmin(np.abs(quantiles - 0.5)))
    y_med = q_preds[:, med_idx]
    out["rmse"] = rmse(y_true, y_med)
    out["mae"] = mae(y_true, y_med)
    return out
