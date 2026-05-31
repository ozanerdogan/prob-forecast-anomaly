"""Unit tests for sequence-window helpers in src/seq_data.py.

Focus: ``freeze_future_covariates`` (the past-covariate transform used by the
leakage-free DeepAR ablation variant). Pure-NumPy, no data/model dependencies.
"""
from __future__ import annotations

import numpy as np

from src.seq_data import freeze_future_covariates


def _toy_cov(n=2, lookback=3, horizon=2, channels=3) -> np.ndarray:
    """cov_seq (N, L+H, C) with a distinct value per (n, t, c)."""
    total = lookback + horizon
    cov = np.empty((n, total, channels), dtype=float)
    for ni in range(n):
        for t in range(total):
            for c in range(channels):
                cov[ni, t, c] = 100 * ni + 10 * t + c
    return cov


def test_freeze_leaves_lookback_and_known_channel_untouched():
    lookback, horizon, n_known = 3, 2, 1  # channel 0 = "calendar" (known future)
    cov = _toy_cov(lookback=lookback, horizon=horizon)
    frozen = freeze_future_covariates(cov, lookback, n_known)

    # Lookback region is never modified.
    assert np.array_equal(frozen[:, :lookback, :], cov[:, :lookback, :])
    # Known (calendar) channel keeps its true future values over the horizon.
    assert np.array_equal(frozen[:, lookback:, :n_known], cov[:, lookback:, :n_known])


def test_freeze_holds_unknown_channels_at_origin():
    lookback, horizon, n_known = 3, 2, 1
    cov = _toy_cov(lookback=lookback, horizon=horizon)
    frozen = freeze_future_covariates(cov, lookback, n_known)

    # Future-unknown channels (>= n_known) are held at the origin (position L-1).
    origin = cov[:, lookback - 1 : lookback, n_known:]  # (N, 1, C_unknown)
    expected = np.broadcast_to(origin, frozen[:, lookback:, n_known:].shape)
    assert np.array_equal(frozen[:, lookback:, n_known:], expected)
    # And they actually changed (the toy values increase with t, so freezing differs).
    assert not np.array_equal(frozen[:, lookback:, n_known:], cov[:, lookback:, n_known:])


def test_freeze_is_noop_when_no_unknown_channels():
    lookback, horizon = 3, 2
    cov = _toy_cov(lookback=lookback, horizon=horizon, channels=1)  # only calendar
    frozen = freeze_future_covariates(cov, lookback, n_known_future=1)
    assert np.array_equal(frozen, cov)


def test_freeze_does_not_mutate_input():
    lookback, horizon = 3, 2
    cov = _toy_cov(lookback=lookback, horizon=horizon)
    before = cov.copy()
    _ = freeze_future_covariates(cov, lookback, n_known_future=1)
    assert np.array_equal(cov, before)
