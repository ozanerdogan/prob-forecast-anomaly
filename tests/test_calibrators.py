"""Unit tests for the stage-2 calibrators (src/calibrators.py).

Synthetic, CPU-only, no dataset and no model — same conventions as the rest
of the suite.
"""
import numpy as np
import pytest

from src.calibration import fit_spread_temperature
from src.calibrators import (
    ACIMargin,
    ACITau,
    CQRCalibrator,
    InputTau,
    StaticTau,
    interval_metrics,
    needed_tau,
    transform_quantiles,
)

LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
ALPHA = 0.1
RNG = np.random.default_rng(0)


def _gaussian_quantiles(mu, sigma, levels=LEVELS):
    """Quantile array (N, H, Q) of N(mu, sigma) with mu, sigma (N, H)."""
    from scipy.stats import norm

    z = norm.ppf(levels)
    return mu[..., None] + sigma[..., None] * z


def test_transform_identity_and_median_fixed_point():
    q = _gaussian_quantiles(np.zeros((5, 24)), np.ones((5, 24)))
    out = transform_quantiles(q, LEVELS, 1.0)
    np.testing.assert_allclose(out, q)
    out2 = transform_quantiles(q, LEVELS, 2.5)
    med = q[..., 3]
    np.testing.assert_allclose(out2[..., 3], med)  # median unchanged by scale
    assert (out2[..., -1] - out2[..., 0] > q[..., -1] - q[..., 0]).all()


def test_transform_per_window_scale_and_shift():
    q = _gaussian_quantiles(np.zeros((4, 6)), np.ones((4, 6)))
    scale = np.array([1.0, 2.0, 0.5, 1.0])
    shift = np.array([0.0, 0.0, 0.0, 3.0])
    out = transform_quantiles(q, LEVELS, scale, shift)
    np.testing.assert_allclose(out[0], q[0])
    width0 = q[1, :, -1] - q[1, :, 0]
    np.testing.assert_allclose(out[1, :, -1] - out[1, :, 0], 2.0 * width0)
    np.testing.assert_allclose(out[3, ..., 3], q[3, ..., 3] + 3.0)


def test_needed_tau_recovers_known_scale():
    # Median 0, q05/q95 = -/+1: a point at |y| = 0.5 needs tau = 0.5.
    n, h = 8, 24
    q = _gaussian_quantiles(np.zeros((n, h)), np.full((n, h), 1.0 / 1.6448536269514722))
    y = np.full((n, h), 0.5)
    tau = needed_tau(y, q, LEVELS, ALPHA)
    np.testing.assert_allclose(tau, 0.5, atol=0.02)
    # Two far points out of 24 may stay uncovered at the 90% target.
    y2 = y.copy()
    y2[:, :2] = 5.0
    tau2 = needed_tau(y2, q, LEVELS, ALPHA)
    np.testing.assert_allclose(tau2, 0.5, atol=0.02)
    # ... but three of 24 force the scale up to reach ceil(0.9*24)=22 covered.
    y3 = y.copy()
    y3[:, :3] = 5.0
    assert (needed_tau(y3, q, LEVELS, ALPHA) > 2.0).all()


def test_static_tau_matches_underlying_fit():
    mu = RNG.normal(size=(50, 24))
    sigma = np.full((50, 24), 0.6)  # too narrow vs unit noise -> tau > 1
    q = _gaussian_quantiles(mu, sigma)
    y = mu + RNG.normal(size=mu.shape)
    cal = StaticTau(ALPHA).fit(y, q, LEVELS)
    direct = fit_spread_temperature(y.reshape(-1), q.reshape(-1, len(LEVELS)), LEVELS)
    assert cal.tau_ == direct
    assert cal.tau_ > 1.0
    after = interval_metrics(y, cal.apply(y, q, LEVELS), LEVELS, ALPHA)
    before = interval_metrics(y, q, LEVELS, ALPHA)
    assert after["picp"] > before["picp"]


def test_cqr_reaches_nominal_coverage_on_exchangeable_data():
    mu_v = RNG.normal(size=(80, 24))
    q_v = _gaussian_quantiles(mu_v, np.full((80, 24), 0.5))  # under-dispersed
    y_v = mu_v + RNG.normal(size=mu_v.shape)
    cal = CQRCalibrator(ALPHA).fit(y_v, q_v, LEVELS)
    assert cal.margin_ > 0
    mu_t = RNG.normal(size=(80, 24))
    q_t = _gaussian_quantiles(mu_t, np.full((80, 24), 0.5))
    y_t = mu_t + RNG.normal(size=mu_t.shape)
    after = interval_metrics(y_t, cal.apply(y_t, q_t, LEVELS), LEVELS, ALPHA)
    assert after["picp"] == pytest.approx(0.9, abs=0.04)


def test_aci_adapts_to_sustained_shift():
    # First half well-specified, second half the truth shifts by 3 sigma:
    # a static interval collapses there; the online rule must recover.
    n, h = 120, 24
    mu = np.zeros((n, h))
    q = _gaussian_quantiles(mu, np.ones((n, h)))
    y = RNG.normal(size=(n, h))
    y[n // 2:] += 3.0
    cal = ACITau(ALPHA).fit(RNG.normal(size=(40, h)), _gaussian_quantiles(np.zeros((40, h)), np.ones((40, h))), LEVELS)
    out = cal.apply(y, q, LEVELS)
    before = interval_metrics(y[n // 2:], q[n // 2:], LEVELS, ALPHA)
    after = interval_metrics(y[n // 2:], out[n // 2:], LEVELS, ALPHA)
    assert after["picp"] > before["picp"] + 0.15
    # tau path must rise after the shift hits
    assert cal.last_tau_path_[-1] > cal.last_tau_path_[n // 2] - 1e-9


def test_aci_first_window_ignores_its_own_outcome():
    # Window 0 must be repaired with the warm-start tau regardless of y[0].
    h = 24
    q = _gaussian_quantiles(np.zeros((3, h)), np.ones((3, h)))
    cal = ACITau(ALPHA)
    cal.gamma_, cal.tau0_ = 0.1, 1.7
    y_easy = np.zeros((3, h))
    y_hard = np.full((3, h), 50.0)
    out_easy = cal.apply(y_easy, q, LEVELS)
    out_hard = cal.apply(y_hard, q, LEVELS)
    np.testing.assert_allclose(out_easy[0], out_hard[0])  # same first window
    assert not np.allclose(out_easy[1], out_hard[1])      # feedback from t=0 on


def test_aci_margin_adapts_and_warm_starts():
    n, h = 120, 24
    q = _gaussian_quantiles(np.zeros((n, h)), np.ones((n, h)))
    y = RNG.normal(size=(n, h))
    y[n // 2:] += 3.0
    cal = ACIMargin(ALPHA).fit(
        _gaussian_quantiles(np.zeros((40, h)), np.ones((40, h)))[..., 3] * 0 + RNG.normal(size=(40, h)),
        _gaussian_quantiles(np.zeros((40, h)), np.ones((40, h))), LEVELS)
    out = cal.apply(y, q, LEVELS)
    before = interval_metrics(y[n // 2:], q[n // 2:], LEVELS, ALPHA)
    after = interval_metrics(y[n // 2:], out[n // 2:], LEVELS, ALPHA)
    assert after["picp"] > before["picp"] + 0.15
    assert cal.m0_ >= 0.0  # warm-started from the offline CQR margin
    # only the outer pair moves; inner quantiles are untouched
    np.testing.assert_allclose(out[..., 1:-1], q[..., 1:-1])


def test_aci_margin_first_window_ignores_own_outcome():
    h = 24
    q = _gaussian_quantiles(np.zeros((3, h)), np.ones((3, h)))
    cal = ACIMargin(ALPHA)
    cal.gamma_, cal.m0_ = 0.2, 0.5
    out_easy = cal.apply(np.zeros((3, h)), q, LEVELS)
    out_hard = cal.apply(np.full((3, h), 40.0), q, LEVELS)
    np.testing.assert_allclose(out_easy[0], out_hard[0])


def test_input_tau_tracks_required_scale():
    pytest.importorskip("sklearn")
    # Contexts whose tail std signals how dispersed the truth is; the model's
    # quantiles ignore it -> required tau correlates with the tail std.
    n, h, L = 300, 24, 168
    tail_sigma = RNG.uniform(0.2, 3.0, size=n)
    ctx = RNG.normal(size=(n, L)) * 0.3
    ctx[:, -24:] = RNG.normal(size=(n, 24)) * tail_sigma[:, None]
    mu = np.zeros((n, h))
    q = _gaussian_quantiles(mu, np.ones((n, h)))
    y = RNG.normal(size=(n, h)) * tail_sigma[:, None]
    cal = InputTau(ALPHA).fit(y, q, LEVELS, context_val=ctx)
    out = cal.apply(y, q, LEVELS, context=ctx)
    assert out.shape == q.shape
    corr = np.corrcoef(cal.last_tau_, tail_sigma)[0, 1]
    assert corr > 0.8
    after = interval_metrics(y, out, LEVELS, ALPHA)
    before = interval_metrics(y, q, LEVELS, ALPHA)
    assert abs(after["picp"] - 0.9) < abs(before["picp"] - 0.9)
