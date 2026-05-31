"""DeepAR-style probabilistic forecaster.

An autoregressive LSTM that emits, at every step, the parameters of a predictive
distribution for the next target value. Trained by minimising the negative
log-likelihood (NLL) with teacher forcing over the whole (lookback+horizon)
window; at inference it conditions on the lookback with teacher forcing and then
rolls forward by sampling, producing sample trajectories that are summarised
into quantiles.

Likelihoods:
  - "gaussian"  -> Normal(mu, sigma)
  - "studentt"  -> StudentT(df=nu, loc=mu, scale=sigma)   (heavier tails; df>2)

Defaults match the Phase-1 LSTM baseline (lookback=168, horizon=24) so the
probabilistic vs deterministic comparison is on equal footing. Training and
inference sampling are both seeded with cfg.seed (42) so a fixed model produces
the same samples — hence the same PICP/CRPS — on every run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LIKELIHOODS = ("gaussian", "studentt")


@dataclass
class DeepARConfig:
    lookback: int = 168
    horizon: int = 24
    n_covariates: int = 0  # extra input channels beyond the autoregressive target
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    likelihood: str = "gaussian"
    batch_size: int = 256
    epochs: int = 12
    lr: float = 1e-3
    seed: int = 42
    n_samples: int = 200  # trajectories drawn at inference

    @property
    def input_size(self) -> int:
        # previous target value + covariates at the current step
        return 1 + self.n_covariates

    @property
    def n_params(self) -> int:
        return 3 if self.likelihood == "studentt" else 2


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) + 1e-4


class DeepAR(nn.Module):
    def __init__(self, cfg: DeepARConfig) -> None:
        super().__init__()
        if cfg.likelihood not in LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {LIKELIHOODS}")
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.n_params)

    def _dist(self, params: torch.Tensor) -> torch.distributions.Distribution:
        mu = params[..., 0]
        sigma = _softplus(params[..., 1])
        if self.cfg.likelihood == "studentt":
            nu = _softplus(params[..., 2]) + 2.0  # df > 2 -> finite variance
            return torch.distributions.StudentT(df=nu, loc=mu, scale=sigma)
        return torch.distributions.Normal(mu, sigma)

    def forward(
        self, prev_target: torch.Tensor, cov: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Teacher-forced pass over a full window.

        prev_target: (B, T)        target shifted by one (input at step t is y[t-1])
        cov:         (B, T, C)     covariates aligned to the output step
        returns a distribution batched over (B, T).
        """
        x = prev_target.unsqueeze(-1)
        if cov.shape[-1] > 0:
            x = torch.cat([x, cov], dim=-1)
        out, _ = self.lstm(x)
        return self._dist(self.head(out))


def _nll(dist: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    return -dist.log_prob(target).mean()


def train_deepar(
    y_tr: np.ndarray, cov_tr: np.ndarray,
    y_va: np.ndarray, cov_va: np.ndarray,
    cfg: DeepARConfig,
    device: str = "cpu",
) -> tuple[DeepAR, list[dict]]:
    """Train DeepAR by NLL with teacher forcing.

    y_*  : (N, L+H) target sequences (scaled)
    cov_*: (N, L+H, C) covariates (C may be 0)
    Loss is the per-step NLL over steps 1..L+H-1 (input y[t-1] -> predict y[t]).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = DeepAR(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def make_dataset(y, cov):
        return TensorDataset(torch.from_numpy(y), torch.from_numpy(cov))

    train_loader = DataLoader(
        make_dataset(y_tr, cov_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(make_dataset(y_va, cov_va), batch_size=cfg.batch_size)

    history: list[dict] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for yb, cb in train_loader:
            yb, cb = yb.to(device), cb.to(device)
            # input target is y shifted right by one step; predict steps 1..T-1
            prev = yb[:, :-1]
            cov_in = cb[:, 1:, :]
            tgt = yb[:, 1:]
            opt.zero_grad()
            dist = model(prev, cov_in)
            loss = _nll(dist, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_n += len(yb)
        tr_loss /= max(tr_n, 1)

        model.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for yb, cb in val_loader:
                yb, cb = yb.to(device), cb.to(device)
                dist = model(yb[:, :-1], cb[:, 1:, :])
                loss = _nll(dist, yb[:, 1:])
                va_loss += loss.item() * len(yb)
                va_n += len(yb)
        va_loss /= max(va_n, 1)
        history.append({"epoch": epoch, "train_nll": tr_loss, "val_nll": va_loss})
        print(f"  epoch {epoch:2d} | train_nll {tr_loss:.4f} | val_nll {va_loss:.4f}")

    return model, history


@torch.no_grad()
def sample_forecast(
    model: DeepAR,
    y_seq: np.ndarray,
    cov_seq: np.ndarray,
    cfg: DeepARConfig,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Draw sample trajectories over the forecast horizon.

    y_seq:   (N, L+H) — only the first L (lookback) values are used to condition;
             the trailing H are ignored (filled at evaluation by the true target).
    cov_seq: (N, L+H, C)
    Returns samples of shape (N, n_samples, H) in scaled space.

    The RNG is reseeded to ``cfg.seed`` before drawing trajectories, so a fixed
    model evaluated on fixed data yields identical samples — and therefore
    identical quantiles / PICP / CRPS — on every run and across scripts.
    """
    model.eval()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    L, H = cfg.lookback, cfg.horizon
    S = cfg.n_samples
    out: list[np.ndarray] = []

    for i in range(0, len(y_seq), batch_size):
        yb = torch.from_numpy(y_seq[i : i + batch_size]).to(device)
        cb = torch.from_numpy(cov_seq[i : i + batch_size]).to(device)
        B = yb.shape[0]

        # Warm the hidden state on the lookback, but stop one step early: the
        # last observed pair (y[L-1], cov[L]) is fed as the FIRST rollout step
        # below, not here, so it is consumed exactly once -- matching training,
        # where the output predicting y[L] has seen (y[0],cov[1])..(y[L-1],cov[L])
        # once each. Conditioning therefore covers k in 0..L-2: inputs y[0..L-2]
        # paired with cov[1..L-1]. (Target-only mode has no covariates and skips
        # the cat branch.)
        x = yb[:, : L - 1].unsqueeze(-1)
        if cb.shape[-1] > 0:
            x = torch.cat([x, cb[:, 1:L, :]], dim=-1)
        _, hidden = model.lstm(x)

        # Expand state across S sample paths.
        h, c = hidden
        h = h.repeat_interleave(S, dim=1)
        c = c.repeat_interleave(S, dim=1)
        last = yb[:, L - 1].repeat_interleave(S)  # (B*S,)
        cov_future = cb[:, L:, :].repeat_interleave(S, dim=0)  # (B*S, H, C)

        traj = torch.empty(B * S, H, device=device)
        for t in range(H):
            step_in = last.unsqueeze(-1)
            if cov_future.shape[-1] > 0:
                step_in = torch.cat([step_in, cov_future[:, t, :]], dim=-1)
            o, (h, c) = model.lstm(step_in.unsqueeze(1), (h, c))
            dist = model._dist(model.head(o[:, -1, :]))
            draw = dist.sample()
            traj[:, t] = draw
            last = draw

        out.append(traj.view(B, S, H).cpu().numpy())

    return np.concatenate(out, axis=0)


def quantiles_from_samples(samples: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Samples (N, S, H) -> quantile forecasts (N, H, Q)."""
    q = np.quantile(samples, quantiles, axis=1)  # (Q, N, H)
    return np.transpose(q, (1, 2, 0))
