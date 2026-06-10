"""Train + evaluate the quantile-head Transformer on Jena Climate.

Target-only, lookback=168, horizon=24, fixed 7-quantile set, pinball-loss
training (matching the DeepAR / LSTM setup). Produces the same probabilistic +
point score block as run_deepar.py so the two probabilistic models are directly
comparable. Quantiles are sorted at inference to avoid crossing.

  python scripts/run_qtransformer.py            # full run
  python scripts/run_qtransformer.py --smoke    # tiny end-to-end sanity run
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
from src.metrics import mase, report_probabilistic, smape  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QUANTILES_7,
    QTransformerConfig,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import Standardizer, TARGET, chronological_split  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

ALPHA = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny fast sanity run")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    df = load_hourly(ROOT / "data" / "processed")
    feat = build_feature_frame(df, TARGET, use_covariates=False)
    splits = chronological_split(feat)

    scaler = Standardizer.fit(splits.train)
    tr = scaler.transform(splits.train).to_numpy().astype(np.float32)
    va = scaler.transform(splits.val).to_numpy().astype(np.float32)
    te = scaler.transform(splits.test).to_numpy().astype(np.float32)

    cfg = QTransformerConfig(n_features=tr.shape[1], quantiles=QUANTILES_7)
    if args.smoke:
        cfg.epochs = 1
        tr = tr[: cfg.lookback + cfg.horizon + 2000]
        va = va[: cfg.lookback + cfg.horizon + 500]
        te = te[: cfg.lookback + cfg.horizon + 500]
    if args.epochs is not None:
        cfg.epochs = args.epochs

    x_tr, y_tr = make_encoder_windows(tr, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(va, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    x_te, y_te = make_encoder_windows(te, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    print(f"shapes  train {x_tr.shape} val {x_va.shape} test {x_te.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=device)

    q_pred_z = predict_quantiles(model, x_te, device=device)  # (N, H, Q) scaled
    q_preds = scaler.inverse_target(q_pred_z, TARGET)
    y_true = scaler.inverse_target(y_te, TARGET)

    quantiles = np.array(cfg.quantiles)
    q_flat = q_preds.reshape(-1, cfg.n_quantiles)
    y_flat = y_true.reshape(-1)
    med = q_flat[:, int(np.argmin(np.abs(quantiles - 0.5)))]

    metrics = report_probabilistic(y_flat, q_flat, quantiles, alpha=ALPHA)
    metrics["smape"] = smape(y_flat, med)
    metrics["mase"] = mase(y_flat, med, splits.train[TARGET].to_numpy(), season=24)
    metrics.update(
        model="qtransformer",
        target=TARGET,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        epochs=cfg.epochs,
        quantiles=quantiles.tolist(),
        n_predictions=int(len(y_flat)),
        device=device,
        smoke=bool(args.smoke),
        history=history,
    )

    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qtransformer.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
