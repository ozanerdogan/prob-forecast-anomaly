"""Intensity-scaling tests for the anomaly injectors (src/anomaly.py).

WHY THESE TESTS EXIST: the robustness sweep reports degradation as a function of
the intensity coefficient (1.0 / 2.0 / 4.0), and the report reads that axis as
"this many local sigmas". That claim only holds if the realised perturbation
magnitude scales *exactly linearly* with the coefficient. None of the other
anomaly tests pin this: a silent re-scaling bug (e.g. applying intensity twice,
or to the std window instead of the magnitude) would shift every heatmap column
while keeping all shape/locality tests green. Linearity is exact here because a
fresh rng with the same seed reproduces the same positions and signs, so the
only difference between two calls is the coefficient itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.anomaly import (
    _window_scale,
    inject_contextual_outlier,
    inject_level_shift,
    inject_point_spike,
    linf_fgsm,
)


def _ctx(n=6, length=48, seed=11):
    # Random-walk style windows so the local rolling std is non-degenerate.
    steps = np.random.default_rng(seed).normal(0.0, 0.3, size=(n, length))
    return np.cumsum(steps, axis=1)


@pytest.mark.parametrize(
    "inject",
    [inject_point_spike, inject_contextual_outlier, inject_level_shift],
    ids=["point_spike", "contextual_outlier", "level_shift"],
)
def test_realised_magnitude_scales_linearly_with_intensity(inject):
    ctx = _ctx()
    # Same seed -> same positions/signs; only the coefficient differs.
    _, m1 = inject(ctx, 1.0, np.random.default_rng(7))
    _, m2 = inject(ctx, 2.0, np.random.default_rng(7))
    _, m4 = inject(ctx, 4.0, np.random.default_rng(7))
    assert m1 > 0
    assert m2 == pytest.approx(2.0 * m1, rel=1e-9)
    assert m4 == pytest.approx(4.0 * m1, rel=1e-9)


def test_perturbation_array_itself_scales_linearly():
    # Not just the reported scalar: the injected deltas double when intensity does.
    ctx = _ctx()
    out1, _ = inject_point_spike(ctx, 1.0, np.random.default_rng(3))
    out2, _ = inject_point_spike(ctx, 2.0, np.random.default_rng(3))
    np.testing.assert_allclose(out2 - ctx, 2.0 * (out1 - ctx), rtol=1e-9, atol=1e-12)


def test_fgsm_radius_scales_linearly_with_intensity():
    ctx = _ctx()
    grad = np.random.default_rng(5).normal(size=ctx.shape)
    grad[grad == 0] = 1.0
    _, e1 = linf_fgsm(ctx, grad, 1.0)
    _, e2 = linf_fgsm(ctx, grad, 2.0)
    assert e1 == pytest.approx(float(np.mean(_window_scale(ctx))), rel=1e-6)
    assert e2 == pytest.approx(2.0 * e1, rel=1e-6)
