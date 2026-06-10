"""Direct multi-horizon LSTM baseline (deterministic point forecast).

Phase 1 scope: a small LSTM trained to map a 168-hour lookback window to a
24-hour forecast horizon. The whole horizon is emitted in one shot by a linear
head (not autoregressive). Outputs a point forecast — no distributional head.
DeepAR / probabilistic Transformer come in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LstmConfig:
    lookback: int = 168
    horizon: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 8
    lr: float = 1e-3
    seed: int = 42
    cell: str = "lstm"  # "lstm" | "gru" — same module serves the GRU twin


class LstmForecaster(nn.Module):
    def __init__(self, cfg: LstmConfig) -> None:
        super().__init__()
        self.cfg = cfg
        rnn_cls = nn.LSTM if cfg.cell == "lstm" else nn.GRU
        self.lstm = rnn_cls(
            input_size=1,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, lookback)
        out, _ = self.lstm(x.unsqueeze(-1))
        last = out[:, -1, :]
        return self.head(last)  # (B, horizon)


def _make_loaders(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_va: np.ndarray, y_va: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def train_lstm(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    cfg: LstmConfig,
    device: str = "cpu",
) -> tuple[LstmForecaster, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = LstmForecaster(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_loader, val_loader = _make_loaders(
        x_train, y_train, x_val, y_val, cfg.batch_size
    )

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
            train_count += len(xb)
        train_loss /= max(train_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * len(xb)
                val_count += len(xb)
        val_loss /= max(val_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f}")

    return model, history


@torch.no_grad()
def predict(model: LstmForecaster, x: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)
