"""Train + evaluate the Quantile-LSTM (the deterministic LSTM's twin).

Same backbone/budget as the phase-1 LSTM (hidden 64, 2 layers, 8 epochs,
seed 42); only the head/loss differ (horizon x 7 quantiles, pinball). Dumps
the full frozen-forecast grid (val/test, clean + injected + white-box FGSM +
hampel-cleaned) via the shared runner and writes results/base/qlstm.json.

  python scripts/models/run_qlstm.py
  python scripts/models/run_qlstm.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.anomaly import FAULT_TYPES_V2  # noqa: E402
from src.metrics import mase, smape  # noqa: E402
from src.model_eval import NONGRAD_V1, evaluate_and_dump  # noqa: E402
from src.models.qlstm import (  # noqa: E402
    QLstmConfig,
    QUANTILES_7,
    predict_qlstm,
    qlstm_context_grad,
    train_qlstm,
)
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--catalog", choices=("v1", "v2"), default="v1",
                        help="v2 adds the phase-2 fault families to the sweep")
    args = parser.parse_args()
    nongrad = NONGRAD_V1 + (FAULT_TYPES_V2 if args.catalog == "v2" else ())

    cfg = QLstmConfig()
    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    if args.smoke:
        cfg.epochs = 1
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    print("Training Quantile-LSTM ...")
    model, history = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)

    predict_fn = lambda x: {"quantiles": predict_qlstm(model, x[:, :, 0], device=DEVICE)}  # noqa: E731
    grad_fn = lambda x, y: qlstm_context_grad(model, x[:, :, 0], y, QUANTILES, device=DEVICE)  # noqa: E731

    metrics = evaluate_and_dump(
        "qlstm", data, predict_fn, root=ROOT, grad_fn=grad_fn,
        quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
    )

    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, "qlstm", "test", "clean"))
    med = d["quantiles"][..., int(np.argmin(np.abs(QUANTILES - 0.5)))].reshape(-1)
    y_flat = d["y_true"].reshape(-1)
    metrics["smape"] = smape(y_flat, med)
    metrics["mase"] = mase(y_flat, med, data.train_target_raw, season=24)
    metrics.update(
        model="qlstm", target=TARGET, lookback=cfg.lookback, horizon=cfg.horizon,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, epochs=cfg.epochs,
        quantiles=QUANTILES.tolist(), seed=cfg.seed, device=DEVICE,
        smoke=bool(args.smoke), history=history,
        pair_note="probabilistic twin of the phase-1 LSTM (same backbone/budget, pinball head)",
    )
    out = ROOT / "results" / "base" / "qlstm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
