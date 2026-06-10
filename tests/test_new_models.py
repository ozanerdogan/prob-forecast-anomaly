"""Unit tests for the phase-2 model additions (qLSTM, GRU cell, DLinear,
QRF, shared feature builder). Tiny CPU models, no dataset, seconds."""
import numpy as np
import pytest
import torch

from src.baselines.lstm_baseline import LstmConfig, LstmForecaster
from src.models.dlinear import DLinearConfig, DLinear, dlinear_context_grad, predict_dlinear
from src.models.lgbm_quantile import context_features
from src.models.qlstm import QLstmConfig, QuantileLstm, predict_qlstm, qlstm_context_grad

LEVELS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def _ctx(n=6, length=168, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, length)).astype(np.float32)


def test_qlstm_shapes_and_sorted_quantiles():
    cfg = QLstmConfig(hidden_size=8, num_layers=1)
    torch.manual_seed(0)
    model = QuantileLstm(cfg)
    x = _ctx()
    out = model(torch.from_numpy(x))
    assert out.shape == (6, cfg.horizon, cfg.n_quantiles)
    q = predict_qlstm(model, x)
    assert q.shape == (6, cfg.horizon, cfg.n_quantiles)
    assert (np.diff(q, axis=-1) >= 0).all()  # non-crossing by sorting


def test_qlstm_context_grad_shape():
    cfg = QLstmConfig(hidden_size=8, num_layers=1)
    torch.manual_seed(0)
    model = QuantileLstm(cfg)
    x = _ctx(4)
    y = np.zeros((4, cfg.horizon), dtype=np.float32)
    g = qlstm_context_grad(model, x, y, LEVELS)
    assert g.shape == x.shape
    assert np.isfinite(g).all() and np.abs(g).sum() > 0


def test_gru_cell_option_builds_gru_and_default_is_lstm():
    lstm = LstmForecaster(LstmConfig(hidden_size=8, num_layers=1))
    gru = LstmForecaster(LstmConfig(hidden_size=8, num_layers=1, cell="gru"))
    assert isinstance(lstm.lstm, torch.nn.LSTM)
    assert isinstance(gru.lstm, torch.nn.GRU)
    x = torch.from_numpy(_ctx(3))
    assert gru(x).shape == (3, 24)


def test_dlinear_point_and_quantile_shapes():
    torch.manual_seed(0)
    point = DLinear(DLinearConfig())
    x = _ctx(5)
    assert point(torch.from_numpy(x)).shape == (5, 24)
    quant = DLinear(DLinearConfig(quantiles=tuple(LEVELS)))
    q = predict_dlinear(quant, x)
    assert q.shape == (5, 24, len(LEVELS))
    assert (np.diff(q, axis=-1) >= 0).all()


def test_dlinear_grad_shapes_for_both_losses():
    torch.manual_seed(0)
    x = _ctx(4)
    y = np.zeros((4, 24), dtype=np.float32)
    g_mse = dlinear_context_grad(DLinear(DLinearConfig()), x, y)
    g_pin = dlinear_context_grad(
        DLinear(DLinearConfig(quantiles=tuple(LEVELS))), x, y, LEVELS)
    assert g_mse.shape == x.shape and g_pin.shape == x.shape


def test_dlinear_attack_damage_bounded_by_weight_l1():
    # Linear model => |f(x+d) - f(x)| <= eps * ||w||_1 for ||d||_inf <= eps.
    torch.manual_seed(0)
    model = DLinear(DLinearConfig())
    x = _ctx(4)
    eps = 0.3
    delta = eps * np.sign(np.random.default_rng(1).standard_normal(x.shape)).astype(np.float32)
    with torch.no_grad():
        f0 = model(torch.from_numpy(x)).numpy()
        f1 = model(torch.from_numpy(x + delta)).numpy()
    w = (model.trend.weight.abs() + model.seasonal.weight.abs()).sum(dim=1).max().item()
    assert np.abs(f1 - f0).max() <= eps * w + 1e-4


def test_context_features_shape_and_horizon_column():
    x = _ctx(3)
    f = context_features(x, horizon=24)
    assert f.shape == (3 * 24, 35)
    np.testing.assert_allclose(f[:24, -1], np.arange(1, 25))  # h feature tiles per window


def test_qrf_fit_predict_noncrossing():
    qf = pytest.importorskip("quantile_forest")  # noqa: F841
    from src.models.qrf import QrfConfig, predict_qrf, train_qrf

    cfg = QrfConfig(n_estimators=5, min_samples_leaf=5, train_stride=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 168)).astype(np.float32)
    y = rng.standard_normal((40, 24)).astype(np.float32)
    model = train_qrf(x, y, cfg)
    q = predict_qrf(model, x[:8], cfg)
    assert q.shape == (8, 24, 7)
    assert (np.diff(q, axis=-1) >= -1e-9).all()  # leaf quantiles: monotone
