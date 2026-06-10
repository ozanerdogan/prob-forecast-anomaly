"""LightGBM point + quantile forecaster (tree family, target-only).

One gradient-boosted model per quantile (pinball objective) plus one point
model (L2), all sharing a lag/rolling-statistic feature set derived from the
168-hour target context alone — so the model is evaluable on the same
(possibly corrupted) contexts as every other model in the anomaly grid.
Horizon position enters as a feature; rows are stacked over the 24 steps.

Tree-family robustness expectations this model exists to test: no gradient
(white-box FGSM undefined) and saturating extrapolation outside the training
range (a shifted context should not be 'followed' the way recurrent models
follow it).

Quantiles are sorted at predict time (same non-crossing convention as the
quantile Transformer).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUANTILES_7 = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
_LAGS = tuple(range(1, 25)) + (48, 72, 96, 120, 144, 168)


@dataclass
class LgbmConfig:
    lookback: int = 168
    horizon: int = 24
    quantiles: tuple = QUANTILES_7
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    train_stride: int = 1
    seed: int = 42
    feature_names: tuple = field(default=(), repr=False)


def context_features(ctx: np.ndarray, horizon: int) -> np.ndarray:
    """Stack per-(window, step) rows: (N, L) -> (N * horizon, F).

    Features: the last 24 hourly lags, the seasonal lags 48..168, rolling
    mean/std of the trailing 24 h and of the full window, and the horizon
    step. All derived from the context only.
    """
    ctx = np.asarray(ctx, dtype=np.float32)
    n, length = ctx.shape
    lags = np.column_stack([ctx[:, -lag] for lag in _LAGS])      # (N, 30)
    tail = ctx[:, -24:]
    stats = np.column_stack([
        tail.mean(axis=1), tail.std(axis=1),
        ctx.mean(axis=1), ctx.std(axis=1),
    ])                                                            # (N, 4)
    base = np.concatenate([lags, stats], axis=1)                  # (N, 34)
    rows = np.repeat(base, horizon, axis=0)                       # (N*H, 34)
    h = np.tile(np.arange(1, horizon + 1, dtype=np.float32), n)[:, None]
    return np.concatenate([rows, h], axis=1)                      # (N*H, 35)


def train_lgbm(x_ctx: np.ndarray, y: np.ndarray, cfg: LgbmConfig) -> dict:
    """Fit one point model + one model per quantile. Returns {name: booster}."""
    import lightgbm as lgb

    feats = context_features(x_ctx, cfg.horizon)
    target = np.asarray(y, dtype=np.float32).reshape(-1)
    common = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        random_state=cfg.seed,
        n_jobs=-1,
        verbose=-1,
    )
    models: dict = {}
    point = lgb.LGBMRegressor(objective="regression", **common)
    point.fit(feats, target)
    models["point"] = point
    for q in cfg.quantiles:
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(q), **common)
        m.fit(feats, target)
        models[f"q{q:g}"] = m
    return models


def predict_lgbm(models: dict, ctx: np.ndarray, cfg: LgbmConfig) -> tuple[np.ndarray, np.ndarray]:
    """(point (N, H), quantiles (N, H, Q)) from a context batch."""
    n = len(ctx)
    feats = context_features(ctx, cfg.horizon)
    point = models["point"].predict(feats).reshape(n, cfg.horizon)
    qs = np.stack(
        [models[f"q{q:g}"].predict(feats).reshape(n, cfg.horizon) for q in cfg.quantiles],
        axis=-1,
    )
    return point.astype(np.float32), np.sort(qs, axis=-1).astype(np.float32)
