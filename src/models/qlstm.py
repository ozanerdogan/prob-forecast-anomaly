"""Quantile-LSTM: the deterministic LSTM's probabilistic twin.

Identical backbone to the phase-1 LSTM baseline (same lookback, hidden size,
layers, epochs, lr, seed) — only the head differs: it emits horizon x Q
values trained by pinball loss instead of a single MSE point. The pair
isolates "what does the probabilistic head buy" from architecture, which is
the controlled deterministic-vs-probabilistic comparison the model roster is
built around. MC dropout was deliberately rejected (under-dispersed).

Quantiles are sorted at predict time (the shared non-crossing convention).
Training clips gradients at 5.0 like the other probabilistic models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.quantile_transformer import pinball_loss_torch

QUANTILES_7 = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class QLstmConfig:
    lookback: int = 168
    horizon: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 8          # parity with the deterministic LSTM twin
    lr: float = 1e-3
    quantiles: tuple = QUANTILES_7
    seed: int = 42

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)


class QuantileLstm(nn.Module):
    def __init__(self, cfg: QLstmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon * cfg.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lookback) target-only
        out, _ = self.lstm(x.unsqueeze(-1))
        last = out[:, -1, :]
        return self.head(last).view(-1, self.cfg.horizon, self.cfg.n_quantiles)


def train_qlstm(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    cfg: QLstmConfig, device: str = "cpu",
) -> tuple[QuantileLstm, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = QuantileLstm(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    q = torch.tensor(cfg.quantiles, device=device, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=cfg.batch_size,
    )

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = pinball_loss_torch(model(xb), yb, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            tr_n += len(xb)
        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += pinball_loss_torch(model(xb), yb, q).item() * len(xb)
                va_n += len(xb)
        history.append({"epoch": epoch, "train_pinball": tr_loss / max(tr_n, 1),
                        "val_pinball": va_loss / max(va_n, 1)})
        print(f"  epoch {epoch:2d} | train_pinball {history[-1]['train_pinball']:.4f} "
              f"| val_pinball {history[-1]['val_pinball']:.4f}")
    return model, history


@torch.no_grad()
def predict_qlstm(
    model: QuantileLstm, x: np.ndarray, batch_size: int = 512, device: str = "cpu"
) -> np.ndarray:
    """Sorted quantile forecasts (N, H, Q) from target-only contexts (N, L)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i: i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.sort(np.concatenate(out, axis=0), axis=-1)


def qlstm_context_grad(
    model: QuantileLstm, ctx: np.ndarray, y_true: np.ndarray,
    quantiles: np.ndarray, device: str = "cpu",
) -> np.ndarray:
    """d(pinball)/d(context) for the white-box FGSM column."""
    model.eval()
    x = torch.tensor(ctx, device=device, requires_grad=True)
    y = torch.tensor(y_true, device=device)
    q = torch.tensor(quantiles, device=device, dtype=torch.float32)
    with torch.backends.cudnn.flags(enabled=False):
        loss = pinball_loss_torch(model(x), y, q)
        loss.backward()
    return x.grad.detach().cpu().numpy()
