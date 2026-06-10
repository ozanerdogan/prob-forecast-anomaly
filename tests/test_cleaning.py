"""Unit tests for the hampel detect-and-clean baseline (src/cleaning.py)."""
import numpy as np

from src.anomaly import inject_level_shift, inject_point_spike
from src.cleaning import hampel_clean


def _smooth_ctx(n=6, length=168):
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / 24.0)
    return np.tile(base, (n, 1)).astype(np.float32)


def test_hampel_removes_point_spike():
    ctx = _smooth_ctx()
    rng = np.random.default_rng(0)
    dirty, _ = inject_point_spike(ctx, intensity=6.0, rng=rng)
    cleaned = hampel_clean(dirty)
    # the spike positions must move back toward the clean series
    err_dirty = np.abs(dirty - ctx).max()
    err_clean = np.abs(cleaned - ctx).max()
    assert err_clean < 0.25 * err_dirty


def test_hampel_keeps_clean_series_nearly_untouched():
    ctx = _smooth_ctx()
    cleaned = hampel_clean(ctx)
    assert np.abs(cleaned - ctx).max() < 1e-5


def test_hampel_passes_level_shift_through():
    # A sustained shift moves the local median itself: the filter must NOT
    # repair it — that contrast is the point of the baseline.
    ctx = _smooth_ctx()
    rng = np.random.default_rng(1)
    dirty, mag = inject_level_shift(ctx, intensity=4.0, rng=rng)
    cleaned = hampel_clean(dirty)
    residual = np.abs(cleaned - ctx).mean()
    assert residual > 0.25 * mag  # most of the shift survives cleaning


def test_hampel_not_in_place():
    ctx = _smooth_ctx()
    rng = np.random.default_rng(2)
    dirty, _ = inject_point_spike(ctx, intensity=6.0, rng=rng)
    snapshot = dirty.copy()
    hampel_clean(dirty)
    np.testing.assert_array_equal(dirty, snapshot)
