"""Unit tests for post-hoc spread-temperature calibration (src/calibration.py).

WHY THESE TESTS EXIST: the same scalar calibration is applied to BOTH
probabilistic models and the pre/post-PICP comparison is a headline result. The
transform is a two-line formula, ``q_cal = median + tau * (q - median)``, where a
single index slip (wrong median column) or a sign error would silently mis-scale
every interval and quietly change the reported coverage -- with no exception to
flag it. These tests pin the formula's invariants (tau=1 is a no-op, the median
is a fixed point, tau>1 widens / tau<1 narrows) and check that the fitter actually
minimizes validation pinball, i.e. it widens demonstrably over-narrow quantiles.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src.calibration import apply_spread_temperature, coverage_at, fit_spread_temperature
from src.metrics import picp

QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])  # median at index 2


def _toy_quantiles(n=200, seed=0):
    # Monotone quantiles per row: median + symmetric offsets around the 0.5 level.
    med = np.random.default_rng(seed).normal(size=(n, 1))
    return med + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # (n, 5)


def test_tau_one_is_identity():
    q = _toy_quantiles()
    assert np.allclose(apply_spread_temperature(q, QUANTILES, 1.0), q)


def test_median_is_a_fixed_point():
    q = _toy_quantiles()
    for tau in (0.5, 1.5, 3.0):
        out = apply_spread_temperature(q, QUANTILES, tau)
        assert np.allclose(out[:, 2], q[:, 2])  # the 0.5 column never moves


def test_tau_widens_and_narrows_the_interval():
    q = _toy_quantiles()
    base = q[:, 4] - q[:, 0]
    wide = apply_spread_temperature(q, QUANTILES, 2.0)
    narrow = apply_spread_temperature(q, QUANTILES, 0.5)
    assert (wide[:, 4] - wide[:, 0] > base).all()
    assert (narrow[:, 4] - narrow[:, 0] < base).all()


def test_fit_widens_overconfident_quantiles():
    # Truth ~ N(0,1) but the model's quantiles are far too narrow (sigma 0.3):
    # the fitter should pick tau > 1 (toward the cap) to widen toward calibration.
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=4000)
    narrow_q = np.array([norm.ppf(p) * 0.3 for p in QUANTILES])
    q = np.broadcast_to(narrow_q, (len(y), len(QUANTILES)))
    assert fit_spread_temperature(y, q, QUANTILES) > 1.0


def test_coverage_at_matches_picp_on_matching_quantiles():
    q = _toy_quantiles()
    y = q[:, 2] + np.random.default_rng(9).normal(size=q.shape[0])
    got = coverage_at(q, y, QUANTILES, alpha=0.1)  # -> 0.05 / 0.95 columns
    assert got == pytest.approx(picp(y, q[:, 0], q[:, 4]))
