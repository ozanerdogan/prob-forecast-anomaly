"""Quantile-head Transformer trained with pinball loss.

An encoder-only Transformer reads the lookback window (target + optional
covariates) and a linear head emits, for every forecast step, a fixed set of
quantiles. Training minimises the averaged pinball (quantile) loss over all
steps and quantile levels. At inference the predicted quantiles are sorted along
the quantile axis to remove crossing.

Defaults match the LSTM / DeepAR setup (lookback=168, horizon=24, seed=42). The
encoder layer is implemented explicitly (rather than nn.TransformerEncoderLayer)
so the first layer's self-attention weights can be read out for the attention
sanity-check figure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Fixed quantile sets. The ablation switches between the 3- and 7-level sets.
QUANTILES_7 = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
QUANTILES_3 = (0.1, 0.5, 0.9)


@dataclass
class QTransformerConfig:
    lookback: int = 168
    horizon: int = 24
    n_features: int = 1  # 1 (target-only) or 1 + n_covariates
    quantiles: tuple[float, ...] = QUANTILES_7
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 12
    lr: float = 1e-3
    seed: int = 42

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)


class _EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer that caches its attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x


def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(length, d_model)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, L, d_model)


class QuantileTransformer(nn.Module):
    def __init__(self, cfg: QTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.register_buffer("pos_enc", _sinusoidal_pe(cfg.lookback, cfg.d_model))
        self.layers = nn.ModuleList(
            [_EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.head = nn.Linear(cfg.d_model, cfg.horizon * cfg.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, H, Q)
        h = self.input_proj(x) + self.pos_enc
        for layer in self.layers:
            h = layer(h)
        pooled = h[:, -1, :]  # last-step representation
        out = self.head(pooled)
        return out.view(-1, self.cfg.horizon, self.cfg.n_quantiles)

    def attention_first_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Return the first layer's (B, L, L) attention map for a batch.

        Restores the prior train/eval mode so this read-out has no side effect.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(x)
        if was_training:
            self.train()
        return self.layers[0].last_attn


def pinball_loss_torch(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Averaged pinball loss.

    pred:   (B, H, Q)
    target: (B, H)
    quantiles: (Q,)
    """
    diff = target.unsqueeze(-1) - pred  # (B, H, Q)
    q = quantiles.view(1, 1, -1)
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def train_qtransformer(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_va: np.ndarray, y_va: np.ndarray,
    cfg: QTransformerConfig,
    device: str = "cpu",
    augment_fn=None,
) -> tuple[QuantileTransformer, list[dict]]:
    """Train the quantile Transformer.

    ``augment_fn`` (phase-4 robust training): called per CPU batch as
    ``augment_fn(xb) -> xb`` BEFORE the device transfer (on-the-fly
    target-channel anomaly injection). Defaults to None -> bit-identical to
    the standard path.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = QuantileTransformer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    quantiles = torch.tensor(cfg.quantiles, device=device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=cfg.batch_size,
    )

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            if augment_fn is not None:
                xb = augment_fn(xb)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = pinball_loss_torch(pred, yb, quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            tr_n += len(xb)
        tr_loss /= max(tr_n, 1)

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = pinball_loss_torch(model(xb), yb, quantiles)
                va_loss += loss.item() * len(xb)
                va_n += len(xb)
        va_loss /= max(va_n, 1)
        history.append({"epoch": epoch, "train_pinball": tr_loss, "val_pinball": va_loss})
        print(f"  epoch {epoch:2d} | train_pinball {tr_loss:.4f} | val_pinball {va_loss:.4f}")

    return model, history


@torch.no_grad()
def predict_quantiles(
    model: QuantileTransformer,
    x: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Return sorted quantile forecasts of shape (N, H, Q) (scaled space)."""
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    preds = np.concatenate(out, axis=0)
    return np.sort(preds, axis=-1)  # enforce non-crossing quantiles
