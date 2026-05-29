"""Shared experiment plumbing reused by the anomaly, ablation and error-analysis
scripts.

Centralises three things so the runner scripts stay small and consistent:
  - data preparation (feature frame -> chronological split -> standardise), with
    an optional exogenous-covariate mode,
  - training thin-wrappers for the LSTM baseline, DeepAR and the quantile
    Transformer, and
  - uniform prediction helpers that turn a (possibly perturbed) input context
    into forecasts, plus the context-gradient needed for the FGSM attack.

Everything works in the standardised space; callers inverse-transform with the
returned Standardizer when they need physical units.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.baselines.lstm_baseline import LstmConfig, LstmForecaster, train_lstm
from src.data_loader import load_hourly
from src.features import build_feature_frame
from src.models.deepar import (
    DeepAR,
    DeepARConfig,
    quantiles_from_samples,
    sample_forecast,
    train_deepar,
)
from src.models.quantile_transformer import (
    QTransformerConfig,
    QuantileTransformer,
    pinball_loss_torch,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import Standardizer, TARGET, chronological_split
from src.seq_data import make_ar_windows, make_encoder_windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Data:
    scaler: Standardizer
    train: np.ndarray  # (T_tr, F) standardised, target in column 0
    val: np.ndarray
    test: np.ndarray
    n_features: int
    train_target_raw: np.ndarray  # unscaled training target (for MASE scale)
    test_index: object  # DatetimeIndex of the test split (for error analysis)


def prepare(use_covariates: bool = False, covariate_cols: list[str] | None = None,
            data_root: Path | None = None) -> Data:
    root = data_root or Path(__file__).resolve().parents[1] / "data" / "processed"
    df = load_hourly(root)
    feat = build_feature_frame(df, TARGET, use_covariates, covariate_cols)
    splits = chronological_split(feat)
    scaler = Standardizer.fit(splits.train)
    return Data(
        scaler=scaler,
        train=scaler.transform(splits.train).to_numpy().astype(np.float32),
        val=scaler.transform(splits.val).to_numpy().astype(np.float32),
        test=scaler.transform(splits.test).to_numpy().astype(np.float32),
        n_features=feat.shape[1],
        train_target_raw=splits.train[TARGET].to_numpy(),
        test_index=splits.test.index,
    )


# --------------------------------------------------------------------------- #
# Training wrappers
# --------------------------------------------------------------------------- #
def fit_lstm(data: Data, cfg: LstmConfig) -> LstmForecaster:
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    # LSTM baseline is target-only: keep the target channel.
    model, _ = train_lstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
    return model


def fit_deepar(data: Data, cfg: DeepARConfig) -> DeepAR:
    y_tr, cov_tr = make_ar_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    y_va, cov_va = make_ar_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    model, _ = train_deepar(y_tr, cov_tr, y_va, cov_va, cfg, device=DEVICE)
    return model


def fit_qtransformer(data: Data, cfg: QTransformerConfig) -> QuantileTransformer:
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    model, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)
    return model


# --------------------------------------------------------------------------- #
# Prediction helpers on (possibly perturbed) contexts
# --------------------------------------------------------------------------- #
def deepar_predict(
    model: DeepAR, cfg: DeepARConfig, y_seq: np.ndarray, cov_seq: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_preds (N,H,Q), samples (N,S,H)) in standardised space."""
    samples = sample_forecast(model, y_seq, cov_seq, cfg, device=DEVICE, batch_size=128)
    return quantiles_from_samples(samples, quantiles), samples


def qtransformer_predict(model: QuantileTransformer, x: np.ndarray) -> np.ndarray:
    """Return q_preds (N,H,Q) in standardised space (sorted quantiles)."""
    return predict_quantiles(model, x, device=DEVICE)


def lstm_predict(model: LstmForecaster, x_target: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Point forecast (N,H) in standardised space from target-only context."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_target), batch_size):
            xb = torch.from_numpy(x_target[i : i + batch_size]).to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Context gradients for the FGSM attack (d loss / d target-context)
# --------------------------------------------------------------------------- #
def lstm_context_grad(model: LstmForecaster, x_target: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.tensor(x_target, device=DEVICE, requires_grad=True)
    y = torch.tensor(y_true, device=DEVICE)
    # cuDNN RNN backward is unsupported in eval mode; disable it for the grad pass.
    with torch.backends.cudnn.flags(enabled=False):
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
    return x.grad.detach().cpu().numpy()


def qtransformer_context_grad(
    model: QuantileTransformer, x: np.ndarray, y_true: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Gradient of the pinball loss wrt the target channel (column 0)."""
    model.eval()
    xt = torch.tensor(x, device=DEVICE, requires_grad=True)
    y = torch.tensor(y_true, device=DEVICE)
    q = torch.tensor(quantiles, device=DEVICE, dtype=torch.float32)
    loss = pinball_loss_torch(model(xt), y, q)
    loss.backward()
    return xt.grad.detach().cpu().numpy()[:, :, 0]


def deepar_context_grad(
    model: DeepAR, cfg: DeepARConfig, y_seq: np.ndarray, cov_seq: np.ndarray
) -> np.ndarray:
    """Gradient of the teacher-forced horizon NLL wrt the conditioning target."""
    model.eval()
    L = cfg.lookback
    ctx = torch.tensor(y_seq[:, :L], device=DEVICE, requires_grad=True)
    future = torch.tensor(y_seq[:, L:], device=DEVICE)
    cov = torch.tensor(cov_seq, device=DEVICE)
    full_y = torch.cat([ctx, future], dim=1)
    with torch.backends.cudnn.flags(enabled=False):
        dist = model(full_y[:, :-1], cov[:, 1:, :])
        nll = -dist.log_prob(full_y[:, 1:])[:, L - 1 :].mean()
        nll.backward()
    return ctx.grad.detach().cpu().numpy()
