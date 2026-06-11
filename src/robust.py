"""Shared anomaly-augmentation for robust training (phase 4 generalisation).

Phase 3 showed anomaly-augmented training makes qLSTM degrade gracefully
under level shift (raw RMSE 9.0 -> 5.4). Phase 4 asks whether this is a
general recipe or qLSTM-specific, so the same augmenter is reused across
models. The factory returns a per-batch hook that injects a random anomaly
into a random subset of the target context, in-place-safe, seeded.

Handles both batch shapes:
  - (B, L)     target-only context (qLSTM)
  - (B, L, F)  multi-channel input -> corrupt the target channel (col 0) only
"""
from __future__ import annotations

import numpy as np
import torch

from src.anomaly import apply_anomaly

AUG_TYPES = ("point_spike", "contextual_outlier", "level_shift")


def make_augmenter(p: float = 0.5, seed: int = 42, intensities=(1.0, 2.0, 4.0)):
    """Per-batch on-the-fly target-context anomaly injection.

    p: fraction of rows corrupted per batch. The anomaly kind/intensity are
    drawn once per batch from a single seeded generator, so the whole training
    run is reproducible.
    """
    rng = np.random.default_rng(seed)

    def augment(xb: torch.Tensor) -> torch.Tensor:
        x = xb.numpy().copy()
        ctx = x[:, :, 0] if x.ndim == 3 else x  # target channel
        mask = rng.random(len(x)) < p
        if mask.any():
            kind = AUG_TYPES[rng.integers(len(AUG_TYPES))]
            inten = float(rng.choice(intensities))
            adv, _ = apply_anomaly(ctx[mask], kind, inten, rng)
            ctx[mask] = adv
            if x.ndim == 3:
                x[:, :, 0] = ctx
        return torch.from_numpy(x)

    return augment


def augment_windows(x: np.ndarray, p: float = 0.5, seed: int = 42,
                    intensities=(1.0, 2.0, 4.0), batch: int = 512) -> np.ndarray:
    """One-shot numpy variant of ``make_augmenter`` for batch-less trainers.

    Tree libraries (LightGBM, QRF) fit on a fixed table, so the per-batch
    torch hook cannot apply; instead the training windows are corrupted once
    up front, in seeded chunks that mirror the per-batch kind/intensity draws
    (p of each chunk corrupted; one kind + one intensity per chunk).
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(x, dtype=np.float32).copy()
    for i in range(0, len(out), batch):
        blk = out[i:i + batch]
        mask = rng.random(len(blk)) < p
        if mask.any():
            kind = AUG_TYPES[rng.integers(len(AUG_TYPES))]
            inten = float(rng.choice(intensities))
            adv, _ = apply_anomaly(blk[mask], kind, inten, rng)
            blk[mask] = adv
    return out
