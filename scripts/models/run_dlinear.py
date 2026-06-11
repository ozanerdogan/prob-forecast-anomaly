"""Train + evaluate the linear family: DLinear (point) and qDLinear (quantile).

Decomposition-linear models (Zeng et al. 2023) on the target context alone,
deliberately without RevIN/last-value normalisation (it would follow an
injected level shift — see the module docstring). One script trains both
twins so they share data preparation; each dumps its own frozen-forecast
grid (white-box FGSM included — the attack's damage on a linear model is
analytically eps * ||w||_1) and writes results/base/{dlinear,qdlinear}.json.

  python scripts/models/run_dlinear.py
  python scripts/models/run_dlinear.py --smoke
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
from src.metrics import mase, smape  # noqa: E402
from src.model_eval import evaluate_and_dump  # noqa: E402
from src.models.dlinear import (  # noqa: E402
    DLinearConfig,
    QUANTILES_7,
    dlinear_context_grad,
    predict_dlinear,
    train_dlinear,
)
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _finish(name, metrics, cfg, data, history, smoke, extra):
    pred_dir = ROOT / "results" / ("predictions_smoke" if smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, name, "test", "clean"))
    if "quantiles" in d:
        flat = d["quantiles"][..., int(np.argmin(np.abs(QUANTILES - 0.5)))].reshape(-1)
    else:
        flat = d["point"].reshape(-1)
        metrics["rmse"] = metrics.pop("point_rmse")
        metrics["mae"] = metrics.pop("point_mae")
    y_flat = d["y_true"].reshape(-1)
    metrics["smape"] = smape(y_flat, flat)
    metrics["mase"] = mase(y_flat, flat, data.train_target_raw, season=24)
    metrics.update(
        model=name, target=TARGET, lookback=cfg.lookback, horizon=cfg.horizon,
        kernel=cfg.kernel, epochs=cfg.epochs, seed=cfg.seed, device=DEVICE,
        smoke=bool(smoke), history=history, **extra,
    )
    out = ROOT / "results" / "base" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"Saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    data = E.prepare(use_covariates=False)
    base_cfg = DLinearConfig()
    x_tr, y_tr = make_encoder_windows(data.train, base_cfg.lookback, base_cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, base_cfg.lookback, base_cfg.horizon,
                                      stride=base_cfg.horizon)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    # ---- DLinear (point, MSE) -------------------------------------------- #
    cfg_p = DLinearConfig()
    if args.smoke:
        cfg_p.epochs = 1
    print("Training DLinear (point) ...")
    model_p, hist_p = train_dlinear(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg_p, device=DEVICE)
    m_p = evaluate_and_dump(
        "dlinear", data,
        lambda x: {"point": predict_dlinear(model_p, x[:, :, 0], device=DEVICE)},
        root=ROOT,
        grad_fn=lambda x, y: dlinear_context_grad(model_p, x[:, :, 0], y, device=DEVICE),
        smoke=args.smoke,
    )
    _finish("dlinear", m_p, cfg_p, data, hist_p, args.smoke,
            {"loss": "mse", "pair_note": "deterministic linear twin (no RevIN, deliberate)"})

    # ---- qDLinear (quantiles, pinball) ----------------------------------- #
    cfg_q = DLinearConfig(quantiles=QUANTILES_7)
    if args.smoke:
        cfg_q.epochs = 1
    print("Training qDLinear (quantile) ...")
    model_q, hist_q = train_dlinear(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg_q, device=DEVICE)
    m_q = evaluate_and_dump(
        "qdlinear", data,
        lambda x: {"quantiles": predict_dlinear(model_q, x[:, :, 0], device=DEVICE)},
        root=ROOT,
        grad_fn=lambda x, y: dlinear_context_grad(model_q, x[:, :, 0], y, QUANTILES, device=DEVICE),
        quantiles=QUANTILES, smoke=args.smoke,
    )
    _finish("qdlinear", m_q, cfg_q, data, hist_q, args.smoke,
            {"loss": "pinball", "quantiles": QUANTILES.tolist(),
             "pair_note": "probabilistic linear twin of dlinear (same backbone/budget)"})


if __name__ == "__main__":
    main()
