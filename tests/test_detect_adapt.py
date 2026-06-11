"""DetectAdaptTau: detector sanity, blend behaviour, leakage guards."""
import numpy as np
import pytest

from src.calibrators import TAU_MAX, TAU_MIN, DetectAdaptTau, interval_metrics

LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
ALPHA = 0.1
RNG = np.random.default_rng(42)


def _windows(n, length=168, shifted=False):
    """Smooth daily-cycle contexts; optionally with a level shift in the tail."""
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / 24.0)
    x = base[None, :] + 0.1 * RNG.standard_normal((n, length))
    if shifted:
        x[:, -24:] += 3.0
    return x


def _quantiles(n, h=24):
    """Synthetic gaussian quantile forecasts around a zero median."""
    from scipy.stats import norm

    med = 0.1 * RNG.standard_normal((n, h))
    q = med[:, :, None] + norm.ppf(LEVELS)[None, None, :]
    return q


@pytest.fixture(scope="module")
def fitted():
    n = 60
    ctx = np.concatenate([_windows(n), _windows(n, shifted=True)])
    lab = np.r_[np.zeros(n), np.ones(n)]
    q = _quantiles(2 * n)
    y = q[..., 3] + 0.2 * RNG.standard_normal(q.shape[:2])
    y[n:] += 2.0  # the shifted windows also miss harder (need wider tau)
    cal = DetectAdaptTau(ALPHA).fit(y, q, LEVELS, context_val=ctx, labels_val=lab)
    return cal, n


def test_detector_separates_clean_from_shifted(fitted):
    cal, n = fitted
    s_clean = cal.anomaly_score(_windows(30))
    s_anom = cal.anomaly_score(_windows(30, shifted=True))
    assert s_clean.mean() < 0.3
    assert s_anom.mean() > 0.7


def test_blend_is_monotone_in_score(fitted):
    cal, _ = fitted
    tau_anom = np.full(5, 4.0)
    taus = cal.blend(np.linspace(0, 1, 5), tau_anom)
    assert np.all(np.diff(taus) >= 0)
    assert taus[0] == pytest.approx(np.clip(cal.tau_clean_, TAU_MIN, TAU_MAX))
    assert taus[-1] == pytest.approx(4.0)


def test_apply_keeps_clean_sharp_and_widens_anomalous(fitted):
    cal, _ = fitted
    n = 40
    q = _quantiles(n)
    y = q[..., 3]
    q_clean = cal.apply(y, q, LEVELS, context=_windows(n))
    q_anom = cal.apply(y, q, LEVELS, context=_windows(n, shifted=True))
    w_raw = interval_metrics(y, q, LEVELS, ALPHA)["mpiw"]
    w_clean = interval_metrics(y, q_clean, LEVELS, ALPHA)["mpiw"]
    w_anom = interval_metrics(y, q_anom, LEVELS, ALPHA)["mpiw"]
    # clean stays close to the static regime; anomalous widens beyond it
    assert w_clean < w_anom
    assert abs(w_clean - w_raw * cal.tau_clean_) / w_raw < 0.6


def test_apply_does_not_mutate_input(fitted):
    cal, _ = fitted
    q = _quantiles(10)
    q_copy = q.copy()
    cal.apply(q[..., 3], q, LEVELS, context=_windows(10))
    np.testing.assert_array_equal(q, q_copy)


def test_fit_requires_labels_and_contexts():
    q = _quantiles(8)
    with pytest.raises(ValueError):
        DetectAdaptTau(ALPHA).fit(q[..., 3], q, LEVELS, context_val=_windows(8))
    with pytest.raises(ValueError):
        DetectAdaptTau(ALPHA).fit(q[..., 3], q, LEVELS,
                                  context_val=_windows(8),
                                  labels_val=np.zeros(8))  # single class


def test_tau_stays_clipped(fitted):
    cal, _ = fitted
    taus = cal.blend(np.array([0.0, 1.0]), np.array([100.0, 100.0]))
    assert np.all((taus >= TAU_MIN) & (taus <= TAU_MAX))
