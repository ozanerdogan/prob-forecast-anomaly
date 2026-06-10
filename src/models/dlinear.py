"""DLinear point + quantile forecasters (linear family, target-only).

Zeng et al. 2023-style decomposition-linear model: the context is split by a
moving average into trend + seasonal parts, each mapped to the horizon by its
own linear layer, and the two are summed. The quantile twin (qDLinear) is the
same backbone with a horizon x Q output trained by pinball loss — the linear
member of the paired deterministic-vs-probabilistic design.

Deliberately NO RevIN / last-value normalisation: it helps under natural
regime change but *follows* an injected level shift (re-anchoring to the
corrupted level), which is exactly wrong for this study's metric — that
trade-off is report material.

Robustness bonus of linearity: an L-inf FGSM attack's damage is analytically
bounded by eps * ||w||_1 per output, so weight regularisation is a direct
robustness knob.
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
class DLinearConfig:
    lookback: int = 168
    horizon: int = 24
    kernel: int = 25          # moving-average window for the trend split
    batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-3
    quantiles: tuple | None = None  # None -> point (MSE); tuple -> pinball
    seed: int = 42

    @property
    def n_out(self) -> int:
        return self.horizon * (len(self.quantiles) if self.quantiles else 1)


class DLinear(nn.Module):
    def __init__(self, cfg: DLinearConfig) -> None:
        super().__init__()
        self.cfg = cfg
        pad = (cfg.kernel - 1) // 2
        self.avg = nn.AvgPool1d(cfg.kernel, stride=1, padding=pad, count_include_pad=False)
        self.trend = nn.Linear(cfg.lookback, cfg.n_out)
        self.seasonal = nn.Linear(cfg.lookback, cfg.n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lookback) target-only
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        out = self.trend(trend) + self.seasonal(x - trend)
        if self.cfg.quantiles:
            return out.view(-1, self.cfg.horizon, len(self.cfg.quantiles))
        return out  # (B, horizon)


def train_dlinear(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    cfg: DLinearConfig, device: str = "cpu",
) -> tuple[DLinear, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = DLinear(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    q = (torch.tensor(cfg.quantiles, device=device, dtype=torch.float32)
         if cfg.quantiles else None)

    def loss_fn(pred, yb):
        if q is None:
            return nn.functional.mse_loss(pred, yb)
        return pinball_loss_torch(pred, yb, q)

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
            loss = loss_fn(model(xb), yb)
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
                va_loss += loss_fn(model(xb), yb).item() * len(xb)
                va_n += len(xb)
        history.append({"epoch": epoch, "train_loss": tr_loss / max(tr_n, 1),
                        "val_loss": va_loss / max(va_n, 1)})
        print(f"  epoch {epoch:2d} | train {history[-1]['train_loss']:.4f} "
              f"| val {history[-1]['val_loss']:.4f}")
    return model, history


@torch.no_grad()
def predict_dlinear(
    model: DLinear, x: np.ndarray, batch_size: int = 2048, device: str = "cpu"
) -> np.ndarray:
    """Point (N, H) or sorted quantiles (N, H, Q) from contexts (N, L)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i: i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    preds = np.concatenate(out, axis=0)
    if model.cfg.quantiles:
        return np.sort(preds, axis=-1)
    return preds


def dlinear_context_grad(
    model: DLinear, ctx: np.ndarray, y_true: np.ndarray,
    quantiles: np.ndarray | None = None, device: str = "cpu",
) -> np.ndarray:
    """d(loss)/d(context) for the white-box FGSM column (MSE or pinball)."""
    model.eval()
    x = torch.tensor(ctx, device=device, requires_grad=True)
    y = torch.tensor(y_true, device=device)
    if model.cfg.quantiles:
        q = torch.tensor(quantiles, device=device, dtype=torch.float32)
        loss = pinball_loss_torch(model(x), y, q)
    else:
        loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    return x.grad.detach().cpu().numpy()
