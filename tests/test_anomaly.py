"""Unit tests for the anomaly-injection harness (src/anomaly.py).

WHY THESE TESTS EXIST: the injectors are the core of the robustness study --
every "RMSE inflation under anomaly" number flows through them, so a silent bug
here would corrupt the whole headline comparison. Two properties are both
load-bearing and easy to break without noticing:

  1. In-place safety. The eval scripts build ONE clean target context and reuse
     it across every model and intensity (``ctx_clean`` in run_anomaly_eval.py).
     If an injector mutated its input, each anomaly would silently contaminate
     every later one. The module docstring promises "they never modify the inputs
     in place" -- these tests pin that.
  2. Locality + reproducibility. point_spike must hit one position, contextual
     outlier a short contiguous burst, level_shift only the tail, and FGSM must
     stay inside its L-inf ball; all must be reproducible from the supplied RNG.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.anomaly import (
    _window_scale,
    apply_anomaly,
    inject_contextual_outlier,
    inject_level_shift,
    inject_point_spike,
    linf_fgsm,
)


def _ctx(n=4, length=48, seed=0):
    return np.random.default_rng(seed).normal(size=(n, length)).astype(np.float32)


def test_injectors_do_not_mutate_input():
    ctx = _ctx()
    before = ctx.copy()
    for fn in (inject_point_spike, inject_contextual_outlier, inject_level_shift):
        fn(ctx, 2.0, np.random.default_rng(1))
        assert np.array_equal(ctx, before), f"{fn.__name__} mutated its input"


def test_point_spike_hits_exactly_one_position_per_row():
    ctx = _ctx()
    out, mag = inject_point_spike(ctx, 3.0, np.random.default_rng(2))
    assert out.shape == ctx.shape
    changed = np.abs(out - ctx) > 0
    assert (changed.sum(axis=1) == 1).all()
    assert mag > 0 and np.isfinite(mag)


def test_contextual_outlier_changes_a_short_contiguous_burst():
    ctx = _ctx()
    out, _ = inject_contextual_outlier(ctx, 2.0, np.random.default_rng(4))  # default burst=3
    changed = np.abs(out - ctx) > 0
    assert (changed.sum(axis=1) == 3).all()
    for r in range(ctx.shape[0]):
        idx = np.where(changed[r])[0]
        assert (np.diff(idx) == 1).all()  # contiguous


def test_level_shift_only_touches_the_tail():
    ctx = _ctx()
    out, _ = inject_level_shift(ctx, 2.0, np.random.default_rng(3))
    half = ctx.shape[1] // 2  # changepoint is drawn in [length//2, length)
    assert np.array_equal(out[:, :half], ctx[:, :half])  # leading half untouched
    assert (out[:, half:] != ctx[:, half:]).any()  # something shifted in the tail


def test_apply_anomaly_matches_direct_call_with_same_rng():
    ctx = _ctx()
    a, ma = apply_anomaly(ctx, "point_spike", 2.0, np.random.default_rng(7))
    b, mb = inject_point_spike(ctx, 2.0, np.random.default_rng(7))
    assert np.allclose(a, b) and ma == pytest.approx(mb)


def test_apply_anomaly_rejects_fgsm():
    # FGSM needs a model gradient and is not dispatchable here (see the docstring).
    with pytest.raises(ValueError):
        apply_anomaly(_ctx(), "fgsm", 1.0, np.random.default_rng(0))


def test_linf_fgsm_stays_inside_the_epsilon_ball():
    ctx = _ctx()
    grad = np.random.default_rng(5).normal(size=ctx.shape)
    out, eps_mean = linf_fgsm(ctx, grad, 2.0)
    eps = 2.0 * _window_scale(ctx)  # per-row L-inf radius
    assert (np.abs(out - ctx) <= eps[:, None] + 1e-5).all()
    assert eps_mean == pytest.approx(float(np.mean(eps)))
