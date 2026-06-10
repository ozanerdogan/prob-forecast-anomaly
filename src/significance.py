"""Forecast-comparison significance tests on the shared window grid.

Closes the "is 2.4294 vs 2.4304 a tie?" question without any retraining:
both tests consume per-window losses computed from the frozen prediction
dumps.

  - Diebold-Mariano on per-window mean squared errors, with a Newey-West
    (Bartlett-kernel) long-run variance. The grid's windows are
    non-overlapping 24 h blocks so autocorrelation is weak, but the HAC
    correction keeps the test honest anyway.
  - Paired bootstrap over windows for the pooled-RMSE delta: resample window
    indices, recompute both RMSEs on the same draw, read off the CI and a
    two-sided p-value.

Convention: losses/deltas are (model_a - model_b); negative deltas mean
model_a is better.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def window_mse(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-window mean squared error: (N, H) -> (N,)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return np.mean((p - y) ** 2, axis=1)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, hac_lags: int = 5) -> dict:
    """Diebold-Mariano test on per-window loss differentials."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    lrv = gamma0
    for k in range(1, min(hac_lags, n - 1) + 1):
        gamma_k = float(np.mean(dc[k:] * dc[:-k]))
        lrv += 2.0 * (1.0 - k / (hac_lags + 1.0)) * gamma_k
    if lrv <= 0 or n < 2:
        # degenerate (identical losses): no evidence against equality
        return {"mean_loss_diff": dbar, "dm_stat": 0.0, "p_value": 1.0, "n": n}
    stat = dbar / np.sqrt(lrv / n)
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return {"mean_loss_diff": dbar, "dm_stat": float(stat), "p_value": float(p), "n": n}


def paired_bootstrap_rmse(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    n_boot: int = 10_000, seed: int = 42,
) -> dict:
    """Paired bootstrap (over windows) for the pooled-RMSE delta a - b."""
    y = np.asarray(y_true, dtype=float)
    se_a = (np.asarray(pred_a, dtype=float) - y) ** 2  # (N, H)
    se_b = (np.asarray(pred_b, dtype=float) - y) ** 2
    n = len(y)

    def pooled_delta(idx):
        return float(np.sqrt(se_a[idx].mean()) - np.sqrt(se_b[idx].mean()))

    obs = pooled_delta(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = pooled_delta(rng.integers(0, n, size=n))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * min(float(np.mean(boots <= 0.0)), float(np.mean(boots >= 0.0)))
    return {
        "delta_rmse": obs,
        "ci95": [float(lo), float(hi)],
        "p_value": min(1.0, p),
        "n_windows": n,
        "n_boot": n_boot,
    }
