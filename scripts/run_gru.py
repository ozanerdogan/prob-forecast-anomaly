"""Train + evaluate the deterministic GRU baseline (recurrent family).

Identical to the phase-1 LSTM in every respect except the recurrent cell
(LstmConfig(cell="gru")) — the second point model of the recurrent family,
and the project-task counterpart of the GRU already validated against
Bari 2025 in the scratch area. Dumps the frozen-forecast grid (white-box
FGSM included) and writes results/base/gru.json.

  python scripts/run_gru.py
  python scripts/run_gru.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.baselines.lstm_baseline import LstmConfig, train_lstm  # noqa: E402
from src.metrics import mase, smape  # noqa: E402
from src.model_eval import evaluate_and_dump  # noqa: E402
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = LstmConfig(cell="gru")
    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    if args.smoke:
        cfg.epochs = 1
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    print("Training GRU ...")
    model, history = train_lstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)

    predict_fn = lambda x: {"point": E.lstm_predict(model, x[:, :, 0])}  # noqa: E731
    grad_fn = lambda x, y: E.lstm_context_grad(model, x[:, :, 0], y)  # noqa: E731

    metrics = evaluate_and_dump(
        "gru", data, predict_fn, root=ROOT, grad_fn=grad_fn, smoke=args.smoke,
    )

    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, "gru", "test", "clean"))
    p_flat, y_flat = d["point"].reshape(-1), d["y_true"].reshape(-1)
    metrics["rmse"] = metrics.pop("point_rmse")
    metrics["mae"] = metrics.pop("point_mae")
    metrics["smape"] = smape(y_flat, p_flat)
    metrics["mase"] = mase(y_flat, p_flat, data.train_target_raw, season=24)
    metrics.update(
        model="gru", target=TARGET, lookback=cfg.lookback, horizon=cfg.horizon,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, epochs=cfg.epochs,
        cell=cfg.cell, seed=cfg.seed, device=DEVICE, smoke=bool(args.smoke),
        history=history,
    )
    out = ROOT / "results" / "base" / "gru.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
