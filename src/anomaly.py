"""Anomaly-injection harness.

Contaminates the *input context* (the lookback window the model conditions on)
with four anomaly families and measures how the forecast degrades. Anomaly
intensity is scaled by the local rolling standard deviation of each window, so a
given intensity coefficient means "this many local sigmas" rather than a fixed
absolute jump -- this keeps the perturbation comparable across calm and volatile
stretches of the series. The realised absolute magnitude (mean local-sigma * eps)
is returned alongside the perturbed context so it can be reported.

Anomaly families:
  - point_spike        : a single sharp spike at one random position.
  - contextual_outlier : a short burst (a few steps) offset from local context.
  - level_shift        : a permanent step from a random changepoint onward
                         (placed in the tail so it reaches the forecast origin).
  - fgsm (l-inf)       : gradient-sign perturbation bounded in an l-infinity ball
                         of radius eps * local_sigma (model-specific; see
                         ``linf_fgsm`` and the eval script).

All injectors operate on a 2-D target-context array ``ctx`` of shape (N, L) in
the model's (standardised) input space and return a perturbed copy plus the mean
realised magnitude. They never modify the inputs in place.
"""
from __future__ import annotations

import numpy as np

ANOMALY_TYPES = ("point_spike", "contextual_outlier", "level_shift", "fgsm")


def local_rolling_std(ctx: np.ndarray, window: int = 24, eps: float = 1e-6) -> np.ndarray:
    """Trailing rolling std per position, shape (N, L).

    The first ``window`` positions use the expanding std so there are no NaNs.
    """
    ctx = np.asarray(ctx, dtype=float)
    n, length = ctx.shape
    out = np.empty_like(ctx)
    for t in range(length):
        lo = max(0, t - window + 1)
        seg = ctx[:, lo : t + 1]
        out[:, t] = seg.std(axis=1)
    return np.maximum(out, eps)


def _window_scale(ctx: np.ndarray, window: int = 24) -> np.ndarray:
    """One representative local sigma per window (median of the rolling std)."""
    return np.median(local_rolling_std(ctx, window), axis=1)


def inject_point_spike(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    out = ctx.copy()
    n, length = out.shape
    scale = local_rolling_std(out, window)
    pos = rng.integers(0, length, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    rows = np.arange(n)
    mag = intensity * scale[rows, pos]
    out[rows, pos] += sign * mag
    return out, float(np.mean(np.abs(mag)))


def inject_contextual_outlier(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator,
    window: int = 24, burst: int = 3,
) -> tuple[np.ndarray, float]:
    out = ctx.copy()
    n, length = out.shape
    scale = local_rolling_std(out, window)
    start = rng.integers(0, max(1, length - burst), size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    total = 0.0
    for i in range(n):
        s = int(start[i])
        seg = slice(s, s + burst)
        mag = intensity * scale[i, seg]
        out[i, seg] += sign[i] * mag
        total += np.abs(mag).sum()
    return out, float(total / (n * burst))


def inject_level_shift(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    out = ctx.copy()
    n, length = out.shape
    scale = _window_scale(out, window)
    # changepoint in the second half so the step reaches the forecast origin
    cp = rng.integers(length // 2, length, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    mag = intensity * scale
    for i in range(n):
        out[i, int(cp[i]):] += sign[i] * mag[i]
    return out, float(np.mean(np.abs(mag)))


def linf_fgsm(
    ctx: np.ndarray, grad: np.ndarray, intensity: float, window: int = 24
) -> tuple[np.ndarray, float]:
    """l-infinity bounded gradient-sign perturbation.

    ``grad`` is d(loss)/d(ctx) supplied by the caller (model-specific). The
    per-window radius is ``intensity * local_sigma`` and the step is the full
    radius along the gradient sign (standard FGSM), so the result stays inside
    the l-inf ball of that radius.
    """
    eps = (intensity * _window_scale(ctx, window))[:, None]
    out = ctx + eps * np.sign(grad)
    return out.astype(np.float32), float(np.mean(eps))


def apply_anomaly(
    ctx: np.ndarray, kind: str, intensity: float, rng: np.random.Generator,
    window: int = 24,
) -> tuple[np.ndarray, float]:
    """Dispatch for the non-gradient anomaly families.

    FGSM is excluded here because it needs the model gradient; call
    ``linf_fgsm`` directly with a precomputed gradient.
    """
    if kind == "point_spike":
        out, mag = inject_point_spike(ctx, intensity, rng, window)
    elif kind == "contextual_outlier":
        out, mag = inject_contextual_outlier(ctx, intensity, rng, window)
    elif kind == "level_shift":
        out, mag = inject_level_shift(ctx, intensity, rng, window)
    else:
        raise ValueError(f"Unknown / non-dispatchable anomaly kind: {kind!r}")
    return out.astype(np.float32), mag
