"""Quantile Regression Forest (Meinshausen 2006) — tree-prob second opinion.

Random-forest quantiles via the ``quantile-forest`` package, on the same
context-only lag/rolling feature set as the LightGBM forecaster (so the two
tree models differ only in ensemble mechanism: bagged forest with leaf-
distribution quantiles vs boosted per-quantile pinball objectives, with no
crossing possible here by construction — all quantiles come from one leaf
distribution).

Cost control (documented as a deliberate budget choice): training rows are
windows at ``train_stride=3`` (neighbouring windows are ~96% overlapping, so
the information loss is negligible) and a 100-tree forest with a leaf-sample
cap; QRF stores leaf values for quantile extraction, so memory scales with
rows x trees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.lgbm_quantile import QUANTILES_7, context_features


@dataclass
class QrfConfig:
    lookback: int = 168
    horizon: int = 24
    quantiles: tuple = QUANTILES_7
    n_estimators: int = 100
    min_samples_leaf: int = 50
    max_samples_leaf: int = 32   # cap stored leaf values (memory/runtime)
    train_stride: int = 3
    seed: int = 42


def train_qrf(x_ctx: np.ndarray, y: np.ndarray, cfg: QrfConfig):
    from quantile_forest import RandomForestQuantileRegressor

    feats = context_features(x_ctx, cfg.horizon)
    target = np.asarray(y, dtype=np.float32).reshape(-1)
    model = RandomForestQuantileRegressor(
        n_estimators=cfg.n_estimators,
        min_samples_leaf=cfg.min_samples_leaf,
        max_samples_leaf=cfg.max_samples_leaf,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(feats, target)
    return model


def predict_qrf(model, ctx: np.ndarray, cfg: QrfConfig) -> np.ndarray:
    """Quantile forecasts (N, H, Q) from a context batch (leaf quantiles)."""
    n = len(ctx)
    feats = context_features(ctx, cfg.horizon)
    qs = model.predict(feats, quantiles=list(cfg.quantiles))  # (N*H, Q)
    return qs.reshape(n, cfg.horizon, len(cfg.quantiles)).astype(np.float32)
