"""Train + evaluate the LSTM baseline on Jena Climate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.lstm_baseline import LstmConfig, predict, train_lstm  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import report  # noqa: E402
from src.preprocessing import (  # noqa: E402
    Standardizer,
    TARGET,
    chronological_split,
    make_windows,
)


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)

    scaler = Standardizer.fit(splits.train[[TARGET]])
    train_z = scaler.transform(splits.train[[TARGET]])[TARGET].to_numpy()
    val_z = scaler.transform(splits.val[[TARGET]])[TARGET].to_numpy()
    test_z = scaler.transform(splits.test[[TARGET]])[TARGET].to_numpy()

    cfg = LstmConfig()
    x_tr, y_tr = make_windows(train_z, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_windows(val_z, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    x_te, y_te = make_windows(test_z, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    print(f"shapes  train {x_tr.shape} val {x_va.shape} test {x_te.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_lstm(x_tr, y_tr, x_va, y_va, cfg, device=device)

    pred_z = predict(model, x_te, device=device)
    y_pred = scaler.inverse_target(pred_z, TARGET).reshape(-1)
    y_true = scaler.inverse_target(y_te, TARGET).reshape(-1)

    metrics = report(y_true, y_pred)
    metrics.update(
        model="lstm",
        target=TARGET,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        epochs=cfg.epochs,
        n_predictions=int(len(y_true)),
        device=device,
        history=history,
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lstm.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
