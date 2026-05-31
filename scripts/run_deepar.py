"""Train + evaluate the DeepAR probabilistic forecaster on Jena Climate.

Target-only, lookback=168, horizon=24 (matching the Phase-1 LSTM baseline).
Produces probabilistic scores (CRPS, pinball, PICP, MPIW, MIS) alongside point
scores (RMSE/MAE off the predictive median, sMAPE, MASE). CRPS is reported via
the quantile approximation (2x mean pinball loss) so it is directly comparable
to the quantile Transformer, which is scored the same way.

  python scripts/run_deepar.py            # full run
  python scripts/run_deepar.py --smoke    # tiny end-to-end sanity run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_hourly  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import (  # noqa: E402
    mase,
    report_probabilistic,
    smape,
)
from src.models.deepar import (  # noqa: E402
    DeepARConfig,
    quantiles_from_samples,
    sample_forecast,
    train_deepar,
)
from src.models.quantile_transformer import QUANTILES_7  # noqa: E402
from src.preprocessing import (  # noqa: E402
    Standardizer,
    TARGET,
    chronological_split,
)
from src.seq_data import make_ar_windows  # noqa: E402

# Shared 7-level quantile grid (same as the quantile Transformer, for fair
# DeepAR-vs-QT scoring).
QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1  # central 90% interval from the 0.05 / 0.95 quantiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny fast sanity run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--likelihood", choices=("gaussian", "studentt"), default="gaussian")
    args = parser.parse_args()

    df = load_hourly(ROOT / "data" / "processed")
    feat = build_feature_frame(df, TARGET, use_covariates=False)
    splits = chronological_split(feat)

    scaler = Standardizer.fit(splits.train)
    tr = scaler.transform(splits.train).to_numpy().astype(np.float32)
    va = scaler.transform(splits.val).to_numpy().astype(np.float32)
    te = scaler.transform(splits.test).to_numpy().astype(np.float32)

    cfg = DeepARConfig(likelihood=args.likelihood)
    if args.smoke:
        cfg.epochs = 1
        cfg.n_samples = 50
        tr = tr[: cfg.lookback + cfg.horizon + 2000]
        va = va[: cfg.lookback + cfg.horizon + 500]
        te = te[: cfg.lookback + cfg.horizon + 500]
    if args.epochs is not None:
        cfg.epochs = args.epochs

    y_tr, cov_tr = make_ar_windows(tr, cfg.lookback, cfg.horizon, stride=1)
    y_va, cov_va = make_ar_windows(va, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    y_te, cov_te = make_ar_windows(te, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    print(f"shapes  train {y_tr.shape} val {y_va.shape} test {y_te.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_deepar(y_tr, cov_tr, y_va, cov_va, cfg, device=device)

    samples = sample_forecast(model, y_te, cov_te, cfg, device=device, batch_size=128)
    # inverse-transform to original temperature scale
    samples = scaler.inverse_target(samples, TARGET)  # (N, S, H)
    y_true = scaler.inverse_target(y_te[:, cfg.lookback :], TARGET)  # (N, H)

    q_preds = quantiles_from_samples(samples, QUANTILES)  # (N, H, Q)
    q_flat = q_preds.reshape(-1, len(QUANTILES))
    y_flat = y_true.reshape(-1)

    # CRPS via the quantile approximation (2x mean pinball), matching the
    # quantile Transformer so DeepAR-vs-QT comparisons are on equal footing.
    metrics = report_probabilistic(y_flat, q_flat, QUANTILES, alpha=ALPHA)
    metrics["smape"] = smape(y_flat, q_preds[:, :, np.argmin(np.abs(QUANTILES - 0.5))].reshape(-1))
    metrics["mase"] = mase(y_flat, q_flat[:, np.argmin(np.abs(QUANTILES - 0.5))],
                           splits.train[TARGET].to_numpy(), season=24)
    metrics.update(
        model="deepar",
        likelihood=cfg.likelihood,
        target=TARGET,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        epochs=cfg.epochs,
        n_samples=cfg.n_samples,
        quantiles=QUANTILES.tolist(),
        n_predictions=int(len(y_flat)),
        device=device,
        smoke=bool(args.smoke),
        history=history,
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "deepar.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
