"""Post-hoc calibration for probabilistic forecasts.

Both probabilistic models tend to be mis-calibrated out of the box (DeepAR is
under-confident/over-confident depending on the run; the Transformer's pinball
quantiles need not be calibrated either). We fit a single scalar *spread
temperature* tau on the validation set that rescales the predictive spread
around the median:

    q_calibrated = median + tau * (q - median)

tau > 1 widens the intervals (cures over-confidence); tau < 1 sharpens them.
This is the quantile-space analogue of temperature scaling and applies uniformly
to both models, so the pre/post PICP comparison is apples-to-apples.

We fit tau by minimising the validation mean pinball loss over a 1-D grid (a
proper scoring rule, so this also improves sharpness-subject-to-calibration
rather than just hitting a coverage target).
"""
from __future__ import annotations

import numpy as np

from src.metrics import mean_pinball_loss, picp


def _median_index(quantiles: np.ndarray) -> int:
    return int(np.argmin(np.abs(quantiles - 0.5)))


def apply_spread_temperature(q_preds: np.ndarray, quantiles: np.ndarray, tau: float) -> np.ndarray:
    """Rescale quantile spread around the median by ``tau``. Shapes (..., Q)."""
    q_preds = np.asarray(q_preds, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    med = q_preds[..., _median_index(quantiles)][..., None]
    return med + tau * (q_preds - med)


def fit_spread_temperature(
    y_val: np.ndarray,
    q_val: np.ndarray,
    quantiles: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Return the tau in ``grid`` minimising validation mean pinball loss."""
    if grid is None:
        grid = np.linspace(0.4, 3.0, 53)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    q_val = np.asarray(q_val, dtype=float).reshape(-1, len(quantiles))
    best_tau, best_loss = 1.0, np.inf
    for tau in grid:
        loss = mean_pinball_loss(y_val, apply_spread_temperature(q_val, quantiles, tau), quantiles)
        if loss < best_loss:
            best_loss, best_tau = loss, float(tau)
    return best_tau


def coverage_at(q_preds: np.ndarray, y_true: np.ndarray, quantiles: np.ndarray, alpha: float) -> float:
    """PICP of the central (1-alpha) interval from the matching quantiles."""
    quantiles = np.asarray(quantiles)
    lo = int(np.argmin(np.abs(quantiles - alpha / 2.0)))
    hi = int(np.argmin(np.abs(quantiles - (1.0 - alpha / 2.0))))
    q = np.asarray(q_preds).reshape(-1, len(quantiles))
    return picp(np.asarray(y_true).reshape(-1), q[:, lo], q[:, hi])
