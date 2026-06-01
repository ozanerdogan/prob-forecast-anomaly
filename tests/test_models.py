"""Unit tests for the probabilistic inference paths (DeepAR + quantile Transformer).

WHY THESE TESTS EXIST: these guard the sampling / quantile machinery that produces
every probabilistic number, and in particular they pin the DeepAR autoregressive
*conditioning alignment*. A subtle off-by-one once warmed the hidden state one
step too far and then fed the last lookback observation a second time as the first
rollout input, so inference no longer matched the teacher-forced training
alignment (it degraded CRPS/PICP). ``test_deepar_first_step_matches_teacher_forcing``
re-derives that invariant directly: with sampling forced to the predictive mean,
the first horizon step must equal the teacher-forced prediction for y[L]. If the
conditioning regresses, that test fails. The rest pin output shapes and the
no-quantile-crossing guarantee (predict_quantiles / quantiles_from_samples sort
along the quantile axis).

Models are tiny and random-initialised (no training, no dataset, CPU only).
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import torch

from src.models.deepar import DeepAR, DeepARConfig, quantiles_from_samples, sample_forecast
from src.models.quantile_transformer import (
    QUANTILES_7,
    QTransformerConfig,
    QuantileTransformer,
    predict_quantiles,
)


def _deepar(n_cov=2):
    torch.manual_seed(0)
    cfg = DeepARConfig(
        lookback=6, horizon=4, n_covariates=n_cov,
        hidden_size=8, num_layers=1, n_samples=8,
    )
    return DeepAR(cfg).eval(), cfg


def test_sample_forecast_shape_and_determinism():
    model, cfg = _deepar()
    n, L, H, C = 3, cfg.lookback, cfg.horizon, cfg.n_covariates
    y = np.random.default_rng(0).normal(size=(n, L + H)).astype(np.float32)
    cov = np.random.default_rng(1).normal(size=(n, L + H, C)).astype(np.float32)
    s1 = sample_forecast(model, y, cov, cfg)
    s2 = sample_forecast(model, y, cov, cfg)
    assert s1.shape == (n, cfg.n_samples, H)
    assert np.array_equal(s1, s2)  # reseeded internally -> identical across calls


def test_quantiles_from_samples_shape_and_sorted():
    samples = np.random.default_rng(0).normal(size=(5, 200, 4))
    q = quantiles_from_samples(samples, np.array([0.1, 0.5, 0.9]))
    assert q.shape == (5, 4, 3)
    assert (np.diff(q, axis=-1) >= 0).all()  # quantiles increasing


def test_deepar_first_step_matches_teacher_forcing():
    # D1 conditioning-alignment regression guard (see module docstring).
    model, cfg = _deepar(n_cov=2)
    n, L, H, C = 3, cfg.lookback, cfg.horizon, cfg.n_covariates
    y = np.random.default_rng(2).normal(size=(n, L + H)).astype(np.float32)
    cov = np.random.default_rng(3).normal(size=(n, L + H, C)).astype(np.float32)

    # Teacher-forced predictive mean for y[L] sits at output index L-1.
    with torch.no_grad():
        tf = model(torch.from_numpy(y[:, :-1]), torch.from_numpy(cov[:, 1:, :]))
    tf_mean_L = tf.mean[:, L - 1].numpy()

    # Force sampling -> distribution mean, so sample_forecast becomes deterministic
    # and propagates means; the first horizon step must then equal tf_mean_L.
    with mock.patch.object(
        torch.distributions.Normal, "sample", lambda self, *a, **k: self.mean
    ):
        samples = sample_forecast(model, y, cov, cfg)
    first_step = samples[:, 0, 0]  # all samples identical under mean-sampling
    assert np.allclose(first_step, tf_mean_L, atol=1e-5)


def test_qtransformer_predict_shape_and_no_crossing():
    torch.manual_seed(0)
    cfg = QTransformerConfig(
        lookback=6, horizon=4, n_features=3, quantiles=QUANTILES_7,
        d_model=16, nhead=2, num_layers=1, dim_ff=32,
    )
    model = QuantileTransformer(cfg).eval()
    x = np.random.default_rng(0).normal(size=(5, cfg.lookback, cfg.n_features)).astype(np.float32)
    q = predict_quantiles(model, x)
    assert q.shape == (5, cfg.horizon, cfg.n_quantiles)
    assert (np.diff(q, axis=-1) >= 0).all()  # sorted -> no crossing
