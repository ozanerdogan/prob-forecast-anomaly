"""Detect-and-clean input repair baseline.

A Hampel filter over the input context: each position is compared against the
median of its centered neighbourhood; positions further than ``n_sigmas``
robust standard deviations (MAD * 1.4826) from that median are replaced by it.
This is the cheap "sanitize the input" alternative to calibration-side repair:
it should fix isolated spikes/outliers well, while a sustained level shift
moves the local median itself and largely passes through — that contrast is
the point of including it.

Operates on (N, L) context arrays in standardised space, never in place.
"""
from __future__ import annotations

import numpy as np


def hampel_clean(
    ctx: np.ndarray, window: int = 12, n_sigmas: float = 3.0, eps: float = 1e-6
) -> np.ndarray:
    """Hampel-filtered copy of ``ctx`` (N, L); outliers -> local median."""
    ctx = np.asarray(ctx, dtype=np.float32)
    n, length = ctx.shape
    out = ctx.copy()
    for t in range(length):
        lo = max(0, t - window)
        hi = min(length, t + window + 1)
        seg = ctx[:, lo:hi]
        med = np.median(seg, axis=1)
        mad = np.median(np.abs(seg - med[:, None]), axis=1)
        sigma = np.maximum(1.4826 * mad, eps)
        mask = np.abs(ctx[:, t] - med) > n_sigmas * sigma
        out[mask, t] = med[mask]
    return out
