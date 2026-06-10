"""Robust (anomaly-augmented) quantile Transformer -- phase-4 generalisation.

Tests whether the phase-3 robust-training win (qLSTM) generalises to the
headline probabilistic model. Same augmenter (p=0.5 per-batch target-context
injection, seed 42) as run_qlstm_robust.py; dumps the frozen v2 grid as
"qtransformer_robust" so the stage-2 calibrators and the 4-corner combined
study pick it up.

  python scripts/run_qt_robust.py
  python scripts/run_qt_robust.py --smoke
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

from src import experiment as E  # noqa: E402
from src.anomaly import FAULT_TYPES_V2  # noqa: E402
from src.metrics import mase, smape  # noqa: E402
from src.model_eval import NONGRAD_V1, evaluate_and_dump  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    QUANTILES_7,
    predict_quantiles,
    train_qtransformer,
)
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.robust import make_augmenter  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--catalog", choices=("v1", "v2"), default="v2")
    args = ap.parse_args()
    nongrad = NONGRAD_V1 + (FAULT_TYPES_V2 if args.catalog == "v2" else ())

    cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_7)
    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon, stride=1)
    x_va, y_va = make_encoder_windows(data.val, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    if args.smoke:
        cfg.epochs = 1
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    print("Training robust (anomaly-augmented) QTransformer ...")
    model, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE,
                                  augment_fn=make_augmenter(p=0.5))

    predict_fn = lambda x: {"quantiles": predict_quantiles(model, x, device=DEVICE)}  # noqa: E731
    grad_fn = lambda x, y: E.qtransformer_context_grad(model, x, y, QUANTILES)  # noqa: E731

    metrics = evaluate_and_dump(
        "qtransformer_robust", data, predict_fn, root=ROOT, grad_fn=grad_fn,
        quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
    )

    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, "qtransformer_robust", "test", "clean"))
    med = d["quantiles"][..., int(np.argmin(np.abs(QUANTILES - 0.5)))].reshape(-1)
    y_flat = d["y_true"].reshape(-1)
    metrics["smape"] = smape(y_flat, med)
    metrics["mase"] = mase(y_flat, med, data.train_target_raw, season=24)
    metrics.update(model="qtransformer_robust", target=TARGET, epochs=cfg.epochs,
                   quantiles=QUANTILES.tolist(), seed=cfg.seed, device=DEVICE,
                   smoke=bool(args.smoke),
                   augment="p=0.5 per batch {spike,outlier,level_shift}x{1,2,4}",
                   note="anomaly-augmented QT; phase-4 robust generalisation test")
    out = ROOT / "results" / "base" / "qtransformer_robust.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
