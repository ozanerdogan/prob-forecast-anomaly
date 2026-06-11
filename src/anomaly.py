"""Anomaly-injection harness.

Contaminates the *input context* (the lookback window the model conditions on)
with four anomaly families and measures how the forecast degrades. Anomaly
intensity is scaled by the local rolling standard deviation of each window, so a
given intensity coefficient means "this many local sigmas" rather than a fixed
absolute jump -- this keeps the perturbation comparable across calm and volatile
stretches of the series. The realised absolute magnitude (mean local-sigma * eps)
is returned alongside the perturbed context so it can be reported.

The sigma reference differs by family: point_spike and contextual_outlier scale
by the *per-position* trailing rolling std (``local_rolling_std``), while
level_shift and fgsm scale by a *single per-window* sigma (``_window_scale``, the
median of the rolling std). So "1 local sigma" is position-local for the first
two and window-representative for the latter two.

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

# Sensor-fault catalog v2 (phase 2): realistic fault families. Intensity
# semantics differ by family — magnitude-scaled (drift, noise_burst) vs
# duration/position-scaled (flatline, gap_imputation, clock_skew); the
# per-family meaning is documented on each injector and the linearity tests
# check the matching quantity.
FAULT_TYPES_V2 = ("flatline", "drift", "noise_burst", "gap_imputation", "clock_skew")

# Canonical per-type index used to derive *validation-side* rng streams
# (test-side streams are fresh default_rng(seed) per setting by convention).
# run_anomaly_eval.py's historical val loop used enumerate() over the v1
# types, which coincides with these values — keep them frozen.
VAL_STREAM_INDEX = {
    "point_spike": 0, "contextual_outlier": 1, "level_shift": 2, "fgsm": 3,
    "flatline": 4, "drift": 5, "noise_burst": 6, "gap_imputation": 7,
    "clock_skew": 8,
}


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


def linf_fgsm_multichannel(
    x: np.ndarray, grad: np.ndarray, intensity: float,
    channel_mask=None, window: int = 24,
) -> tuple[np.ndarray, float]:
    """l-infinity FGSM over EVERY (attackable) input channel of (N, L, C).

    Same construction as ``linf_fgsm`` but the radius is computed
    channel-wise — ``intensity * that channel's local sigma`` — so covariates
    are perturbed in proportion to their own variability rather than the
    target's. ``channel_mask`` (bool per channel) limits the attack to
    physically attackable channels: calendar features are deterministic
    (an attacker cannot change the date), so they must stay clean.
    """
    x = np.asarray(x, dtype=np.float32)
    grad = np.asarray(grad)
    out = x.copy()
    mask = (np.ones(x.shape[2], dtype=bool) if channel_mask is None
            else np.asarray(channel_mask, dtype=bool))
    mags = []
    for c in np.flatnonzero(mask):
        eps = (intensity * _window_scale(x[:, :, c], window))[:, None]
        out[:, :, c] = x[:, :, c] + eps * np.sign(grad[:, :, c])
        mags.append(float(np.mean(eps)))
    return out, float(np.mean(mags)) if mags else 0.0


def inject_flatline(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    """Stuck sensor: the context tail freezes at the last pre-fault value.

    The most common real sensor fault (icing, power glitch). Intensity scales
    the *duration*: 6 h per unit (1/2/4 -> 6/12/24 h), always at the very end
    of the context so the fault reaches the forecast origin. The frozen
    segment erases the diurnal signal there. ``rng`` only draws nothing —
    kept in the signature for dispatch uniformity.
    """
    out = ctx.copy()
    n, length = out.shape
    k = min(length - 1, max(1, int(round(6 * intensity))))
    out[:, length - k:] = out[:, length - k - 1][:, None]
    mag = float(np.mean(np.abs(out - ctx)))
    return out.astype(np.float32), mag


def inject_drift(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    """Calibration drift: a linear bias ramp accumulating over the last 24 h.

    The slow sibling of the level shift. The ramp ends at
    ``intensity * window_sigma`` (random sign per window) at the forecast
    origin; reported magnitude is the mean absolute final offset, so it
    scales linearly with intensity.
    """
    out = ctx.copy()
    n, length = out.shape
    span = min(24, length)
    scale = _window_scale(out, window)
    sign = rng.choice([-1.0, 1.0], size=n)
    ramp = np.linspace(0.0, 1.0, span, dtype=float)[None, :]
    out[:, length - span:] += (sign * intensity * scale)[:, None] * ramp
    return out.astype(np.float32), float(np.mean(np.abs(intensity * scale)))


def inject_noise_burst(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator,
    window: int = 24, burst: int = 12,
) -> tuple[np.ndarray, float]:
    """Variance inflation: white noise over a 12 h segment in the tail.

    Electronic interference / turbulence. Standard-normal draws are made
    first and then scaled by ``intensity * local_sigma``, so with a fixed rng
    the perturbation pattern is identical across intensities and the realised
    magnitude is exactly linear in the coefficient.
    """
    out = ctx.copy()
    n, length = out.shape
    scale = local_rolling_std(out, window)
    start = rng.integers(max(0, length - 48), length - burst, size=n)
    z = rng.standard_normal((n, burst))
    total = 0.0
    for i in range(n):
        seg = slice(int(start[i]), int(start[i]) + burst)
        noise = intensity * scale[i, seg] * z[i]
        out[i, seg] += noise
        total += np.abs(noise).sum()
    return out.astype(np.float32), float(total / (n * burst))


def inject_gap_imputation(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    """Missing block filled by linear interpolation (imputation artefact).

    Mimics what the loading pipeline itself does to gaps: a 6 h-per-unit
    block (1/2/4 -> 6/12/24 h) in the context tail is replaced by the straight
    line between its endpoints — a realistic over-smoothing. Intensity scales
    the *duration*.
    """
    out = ctx.copy()
    n, length = out.shape
    k = min(length - 4, max(2, int(round(6 * intensity))))
    # start so the gap lies in the tail but keeps a right endpoint
    lo = max(1, length - 72)
    hi = max(lo + 1, length - k - 1)
    start = rng.integers(lo, hi, size=n)
    for i in range(n):
        s = int(start[i])
        e = s + k
        left, right = out[i, s - 1], out[i, e]
        out[i, s:e] = np.linspace(left, right, k + 2)[1:-1]
    mag = float(np.mean(np.abs(out - ctx)))
    return out.astype(np.float32), mag


def inject_clock_skew(
    ctx: np.ndarray, intensity: float, rng: np.random.Generator, window: int = 24
) -> tuple[np.ndarray, float]:
    """Logger clock error: the whole context is stale by 2 h per unit.

    The model sees data delayed by h = 2 * intensity hours (1/2/4 -> 2/4/8 h):
    a pure phase corruption of the diurnal cycle. The earliest h positions
    hold the oldest available value. Intensity scales the *delay*.
    """
    out = ctx.copy()
    n, length = out.shape
    h = min(length - 1, max(1, int(round(2 * intensity))))
    out[:, h:] = ctx[:, : length - h]
    out[:, :h] = ctx[:, 0][:, None]
    mag = float(np.mean(np.abs(out - ctx)))
    return out.astype(np.float32), mag


_DISPATCH = {
    "point_spike": inject_point_spike,
    "contextual_outlier": inject_contextual_outlier,
    "level_shift": inject_level_shift,
    "flatline": inject_flatline,
    "drift": inject_drift,
    "noise_burst": inject_noise_burst,
    "gap_imputation": inject_gap_imputation,
    "clock_skew": inject_clock_skew,
}


def apply_anomaly(
    ctx: np.ndarray, kind: str, intensity: float, rng: np.random.Generator,
    window: int = 24,
) -> tuple[np.ndarray, float]:
    """Dispatch for the non-gradient anomaly families (v1 + v2 catalog).

    FGSM is excluded here because it needs the model gradient; call
    ``linf_fgsm`` directly with a precomputed gradient.
    """
    try:
        fn = _DISPATCH[kind]
    except KeyError:
        raise ValueError(f"Unknown / non-dispatchable anomaly kind: {kind!r}") from None
    out, mag = fn(ctx, intensity, rng, window)
    return out.astype(np.float32), mag
