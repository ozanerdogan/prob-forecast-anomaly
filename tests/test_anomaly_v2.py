"""Tests for the v2 sensor-fault injectors (flatline, drift, noise burst,
gap+imputation, clock skew).

Same conventions as test_anomaly.py / test_intensity_scaling.py: linearity is
checked on the quantity each family scales (magnitude for drift/noise burst,
duration/delay for flatline/gap/clock skew), plus locality and in-place
safety. All synthetic, CPU-only.
"""
import numpy as np
import pytest

from src.anomaly import (
    FAULT_TYPES_V2,
    apply_anomaly,
    inject_clock_skew,
    inject_drift,
    inject_flatline,
    inject_gap_imputation,
    inject_noise_burst,
)

L = 168


def _ctx(n=8, length=L, seed=0):
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / 24.0)
    rng = np.random.default_rng(seed)
    return (np.tile(base, (n, 1)) + 0.05 * rng.standard_normal((n, length))).astype(np.float32)


def test_dispatch_covers_v2_catalog():
    ctx = _ctx()
    for kind in FAULT_TYPES_V2:
        out, mag = apply_anomaly(ctx, kind, 2.0, np.random.default_rng(1))
        assert out.shape == ctx.shape and out.dtype == np.float32
        assert mag > 0
        assert not np.array_equal(out, ctx)


def test_v2_injectors_not_in_place():
    ctx = _ctx()
    snap = ctx.copy()
    for kind in FAULT_TYPES_V2:
        apply_anomaly(ctx, kind, 4.0, np.random.default_rng(2))
    np.testing.assert_array_equal(ctx, snap)


def test_flatline_duration_scales_and_sits_in_tail():
    ctx = _ctx()
    for intensity, k in ((1.0, 6), (2.0, 12), (4.0, 24)):
        out, _ = inject_flatline(ctx, intensity, np.random.default_rng(3))
        changed = out != ctx
        # only the last k positions may change, and they are all frozen
        assert not changed[:, : L - k].any()
        np.testing.assert_array_equal(
            out[:, L - k:], np.repeat(out[:, L - k - 1][:, None], k, axis=1)
        )


def test_drift_magnitude_linear_and_local():
    ctx = _ctx()
    mags = {}
    for intensity in (1.0, 2.0, 4.0):
        out, mag = inject_drift(ctx, intensity, np.random.default_rng(4))
        mags[intensity] = mag
        assert not (out[:, : L - 24] != ctx[:, : L - 24]).any()  # only last 24h
        # ramp: |delta| grows toward the origin
        delta = np.abs(out - ctx)[:, L - 24:]
        assert (delta[:, -1] >= delta[:, 0]).all()
    assert mags[2.0] == pytest.approx(2 * mags[1.0], rel=1e-6)
    assert mags[4.0] == pytest.approx(4 * mags[1.0], rel=1e-6)


def test_noise_burst_magnitude_exactly_linear_same_seed():
    ctx = _ctx()
    mags = {i: inject_noise_burst(ctx, i, np.random.default_rng(5))[1] for i in (1.0, 2.0, 4.0)}
    assert mags[2.0] == pytest.approx(2 * mags[1.0], rel=1e-6)
    assert mags[4.0] == pytest.approx(4 * mags[1.0], rel=1e-6)
    out, _ = inject_noise_burst(ctx, 1.0, np.random.default_rng(5))
    assert ((out != ctx).sum(axis=1) <= 12).all()  # 12h burst per window


def test_gap_imputation_duration_scales_and_interpolates():
    ctx = _ctx()
    for intensity, k in ((1.0, 6), (2.0, 12), (4.0, 24)):
        out, _ = inject_gap_imputation(ctx, intensity, np.random.default_rng(6))
        changed = (out != ctx).sum(axis=1)
        assert (changed <= k).all() and (changed >= k - 1).all()
        # the filled block is a straight line -> second difference ~ 0 inside
        for i in range(3):
            idx = np.where(out[i] != ctx[i])[0]
            if len(idx) >= 3:
                seg = out[i, idx]
                assert np.abs(np.diff(seg, 2)).max() < 1e-4


def test_clock_skew_delay_scales_exactly():
    ctx = _ctx()
    for intensity, h in ((1.0, 2), (2.0, 4), (4.0, 8)):
        out, _ = inject_clock_skew(ctx, intensity, np.random.default_rng(7))
        np.testing.assert_array_equal(out[:, h:], ctx[:, : L - h])
        np.testing.assert_allclose(out[:, :h], np.repeat(ctx[:, :1], h, axis=1))


def test_v2_reproducible_with_same_seed():
    ctx = _ctx()
    for kind in FAULT_TYPES_V2:
        a, ma = apply_anomaly(ctx, kind, 2.0, np.random.default_rng(11))
        b, mb = apply_anomaly(ctx, kind, 2.0, np.random.default_rng(11))
        np.testing.assert_array_equal(a, b)
        assert ma == mb
