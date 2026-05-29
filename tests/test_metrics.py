"""Unit tests for the Phase-2 metric additions in src/metrics.py.

Each metric gets a short test against a hand-checkable case or a known
closed-form identity. Phase-1 metrics (rmse/mae/mape) are exercised lightly to
guard against accidental regressions.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src import metrics as M


def test_rmse_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert M.mae(y, p) == pytest.approx(2.0 / 3.0)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_smape_perfect_and_symmetry():
    y = np.array([1.0, 2.0, 3.0])
    assert M.smape(y, y) == pytest.approx(0.0)
    # Equal-magnitude over/under prediction give the same sMAPE.
    over = M.smape(np.array([2.0]), np.array([3.0]))
    under = M.smape(np.array([3.0]), np.array([2.0]))
    assert over == pytest.approx(under)
    # 100% relative on a single point: y=2, yhat=6 -> 2*4/(8)=1 -> 100.
    assert M.smape(np.array([2.0]), np.array([6.0])) == pytest.approx(100.0)


def test_mase_equals_one_for_seasonal_naive():
    rng = np.random.default_rng(0)
    season = 24
    train = rng.normal(size=500)
    # A forecast whose MAE equals the seasonal-naive in-sample MAE -> MASE == 1.
    scale = np.mean(np.abs(train[season:] - train[:-season]))
    y_true = np.zeros(10)
    y_pred = np.full(10, scale)  # constant abs error == scale
    assert M.mase(y_true, y_pred, train, season=season) == pytest.approx(1.0)


def test_mase_raises_on_short_train():
    with pytest.raises(ValueError):
        M.mase(np.array([1.0]), np.array([1.0]), np.arange(10.0), season=24)


def test_pinball_loss_median_equals_half_mae():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([1.0, 1.0, 1.0, 1.0])
    # At q=0.5 pinball == 0.5 * MAE.
    assert M.pinball_loss(y, p, 0.5) == pytest.approx(0.5 * M.mae(y, p))


def test_pinball_loss_asymmetry():
    # Under-prediction penalised more at high quantiles.
    y = np.array([10.0])
    p = np.array([0.0])
    assert M.pinball_loss(y, p, 0.9) == pytest.approx(9.0)
    assert M.pinball_loss(y, p, 0.1) == pytest.approx(1.0)


def test_mean_pinball_loss_shape():
    y = np.array([0.0, 1.0])
    qs = np.array([0.1, 0.5, 0.9])
    preds = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
    val = M.mean_pinball_loss(y, preds, qs)
    manual = np.mean([M.pinball_loss(y, preds[:, i], qs[i]) for i in range(3)])
    assert val == pytest.approx(manual)


def test_crps_ensemble_matches_gaussian_closed_form():
    # Large Gaussian ensemble CRPS should approach the closed-form value.
    rng = np.random.default_rng(42)
    mu, sigma = 2.0, 1.5
    y = np.array([3.0])
    samples = rng.normal(mu, sigma, size=(1, 40000))
    approx = M.crps_ensemble(y, samples)
    exact = M.crps_gaussian(y, np.array([mu]), np.array([sigma]))
    assert approx == pytest.approx(exact, rel=0.02)


def test_crps_gaussian_zero_variance_is_mae():
    # As sigma -> 0 CRPS -> |y - mu|.
    y = np.array([5.0])
    mu = np.array([3.0])
    val = M.crps_gaussian(y, mu, np.array([1e-9]))
    assert val == pytest.approx(2.0, abs=1e-3)


def test_crps_from_quantiles_is_twice_mean_pinball():
    y = np.array([0.0, 1.0])
    qs = np.array([0.25, 0.5, 0.75])
    preds = np.array([[-1.0, 0.0, 1.0], [0.5, 1.0, 1.5]])
    assert M.crps_from_quantiles(y, preds, qs) == pytest.approx(
        2.0 * M.mean_pinball_loss(y, preds, qs)
    )


def test_picp_full_and_zero_coverage():
    y = np.array([1.0, 2.0, 3.0])
    assert M.picp(y, y - 1, y + 1) == pytest.approx(1.0)
    assert M.picp(y, y + 5, y + 10) == pytest.approx(0.0)
    # Half inside.
    y2 = np.array([0.0, 10.0])
    assert M.picp(y2, np.array([-1.0, -1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_mis_reduces_to_width_when_covered():
    y = np.array([0.0, 0.0])
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    # All covered -> MIS == mean width.
    assert M.mis(y, lower, upper, alpha=0.1) == pytest.approx(np.mean(upper - lower))


def test_mis_penalises_miss():
    y = np.array([5.0])
    lower = np.array([-1.0])
    upper = np.array([1.0])
    alpha = 0.1
    expected = (1.0 - (-1.0)) + (2.0 / alpha) * (5.0 - 1.0)
    assert M.mis(y, lower, upper, alpha) == pytest.approx(expected)


def test_report_probabilistic_keys_and_coverage():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.normal(0.0, 1.0, size=n)
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    # Oracle Gaussian quantiles -> ~90% PICP for the 0.05/0.95 interval.
    q_preds = np.stack([norm.ppf(q) * np.ones(n) for q in quantiles], axis=1)
    out = M.report_probabilistic(y, q_preds, quantiles, alpha=0.1)
    for key in ("crps", "pinball", "picp", "mpiw", "mis", "rmse", "mae"):
        assert key in out
    assert out["picp"] == pytest.approx(0.90, abs=0.03)
