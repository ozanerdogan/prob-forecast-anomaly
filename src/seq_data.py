"""Sequence-window builders for the probabilistic models.

Two window layouts are produced, both from a (scaled) feature matrix whose first
column is the target and whose remaining columns are covariates:

  - ``make_ar_windows``  -> full (lookback+horizon) target sequences plus aligned
    covariates, for the autoregressive DeepAR (teacher forcing + AR rollout).
  - ``make_encoder_windows`` -> a (lookback, n_features) encoder input and a
    (horizon,) target, for the encoder-style quantile Transformer.

These complement ``src/preprocessing.make_windows`` (target-only, point models);
the existing helper is left untouched so the Phase-1 baselines are unaffected.
"""
from __future__ import annotations

import numpy as np


def _check(features: np.ndarray, lookback: int, horizon: int) -> int:
    if features.ndim != 2:
        raise ValueError("Expected features of shape (T, F) with target in column 0.")
    n = len(features) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Series shorter than lookback + horizon.")
    return n


def make_ar_windows(
    features: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive windows.

    Returns:
      y_seq:   (N, L+H)      target values over the whole window
      cov_seq: (N, L+H, C)   aligned covariates (C may be 0 -> shape (N, L+H, 0))
    """
    n = _check(features, lookback, horizon)
    total = lookback + horizon
    idx = np.arange(0, n, stride)
    target = features[:, 0]
    cov = features[:, 1:]
    y_seq = np.stack([target[i : i + total] for i in idx]).astype(np.float32)
    cov_seq = np.stack([cov[i : i + total] for i in idx]).astype(np.float32)
    return y_seq, cov_seq


def freeze_future_covariates(
    cov_seq: np.ndarray,
    lookback: int,
    n_known_future: int,
) -> np.ndarray:
    """Hold the future-*unknown* covariate channels at their origin value.

    For a realistic ("past covariate") forecast the model must not see the true
    future values of channels that are unknown at prediction time (the exogenous
    weather). Over the horizon region (positions ``lookback`` onward) those
    channels are frozen to their last observed value (position ``lookback - 1``),
    i.e. a persistence extrapolation. The first ``n_known_future`` channels
    (calendar features) are genuinely known and left at their true future values.

    cov_seq: (N, L+H, C). Returns a modified copy; the input is not touched.
    Channels with index >= ``n_known_future`` are frozen; if there are none
    (``C <= n_known_future``) the array is returned unchanged.
    """
    out = cov_seq.copy()
    if out.shape[-1] > n_known_future:
        origin = out[:, lookback - 1 : lookback, n_known_future:]  # (N, 1, C_unknown)
        out[:, lookback:, n_known_future:] = origin
    return out


def make_encoder_windows(
    features: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Encoder windows.

    Returns:
      x: (N, L, F)   lookback features (target + covariates)
      y: (N, H)      target over the forecast horizon
    """
    n = _check(features, lookback, horizon)
    idx = np.arange(0, n, stride)
    target = features[:, 0]
    x = np.stack([features[i : i + lookback] for i in idx]).astype(np.float32)
    y = np.stack([target[i + lookback : i + lookback + horizon] for i in idx]).astype(np.float32)
    return x, y
