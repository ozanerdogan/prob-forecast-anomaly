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
