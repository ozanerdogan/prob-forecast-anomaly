"""Unit tests for the Diebold-Mariano / paired-bootstrap helpers."""
import numpy as np

from src.significance import dm_test, paired_bootstrap_rmse, window_mse

RNG = np.random.default_rng(7)


def test_window_mse_shape_and_value():
    y = np.zeros((5, 24))
    p = np.full((5, 24), 2.0)
    np.testing.assert_allclose(window_mse(y, p), 4.0)


def test_dm_identical_losses_is_insignificant():
    loss = RNG.uniform(1.0, 2.0, size=200)
    res = dm_test(loss, loss.copy())
    assert res["p_value"] == 1.0
    assert res["dm_stat"] == 0.0


def test_dm_detects_clear_difference():
    base = RNG.uniform(1.0, 2.0, size=300)
    res = dm_test(base + 1.0, base)  # a consistently worse
    assert res["p_value"] < 1e-6
    assert res["dm_stat"] > 0


def test_dm_noise_only_is_usually_insignificant():
    a = RNG.normal(1.5, 0.1, size=300)
    b = a + RNG.normal(0.0, 0.001, size=300)  # tiny symmetric noise
    res = dm_test(a, b)
    assert res["p_value"] > 0.01


def test_bootstrap_ci_excludes_zero_for_clear_winner():
    y = RNG.normal(size=(150, 24))
    good = y + RNG.normal(0.0, 0.5, size=y.shape)
    bad = y + RNG.normal(0.0, 1.5, size=y.shape)
    res = paired_bootstrap_rmse(y, good, bad, n_boot=500, seed=0)
    assert res["delta_rmse"] < 0
    assert res["ci95"][1] < 0  # whole CI below zero -> a clearly better
    assert res["p_value"] < 0.05


def test_bootstrap_near_tie_keeps_zero_in_ci():
    # Two equally skilled models with independent errors: the delta is pure
    # noise, so the CI must straddle zero. (A *consistent* delta, however
    # tiny, is correctly resolved as significant by the paired test.)
    y = RNG.normal(size=(150, 24))
    a = y + RNG.normal(0.0, 1.0, size=y.shape)
    b = y + RNG.normal(0.0, 1.0, size=y.shape)
    res = paired_bootstrap_rmse(y, a, b, n_boot=500, seed=0)
    assert res["ci95"][0] <= 0.0 <= res["ci95"][1]
