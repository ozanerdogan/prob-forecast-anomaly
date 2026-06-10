"""Train + evaluate the Quantile Regression Forest (tree-prob second opinion).

Same 35 context-only features as LightGBM; quantiles come from leaf
distributions (Meinshausen 2006), so crossing is impossible by construction.
Budget choices (stride-3 training rows, 100 trees, leaf-sample cap) are
documented in the module. No gradient -> no white-box FGSM (transfer column
arrives with the phase-4 wave). Dumps the frozen grid and writes
results/base/qrf.json.

  python scripts/run_qrf.py
  python scripts/run_qrf.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.metrics import mase, smape  # noqa: E402
from src.model_eval import evaluate_and_dump  # noqa: E402
from src.models.qrf import QrfConfig, predict_qrf, train_qrf  # noqa: E402
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = QrfConfig()
    quantiles = np.array(cfg.quantiles)
    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, cfg.lookback, cfg.horizon,
                                      stride=cfg.train_stride)
    if args.smoke:
        cfg.n_estimators = 10
        x_tr, y_tr = x_tr[:1000], y_tr[:1000]
    print(f"Training QRF on {len(x_tr)} windows x {cfg.horizon} rows ...")
    model = train_qrf(x_tr[:, :, 0], y_tr, cfg)

    predict_fn = lambda x: {"quantiles": predict_qrf(model, x[:, :, 0], cfg)}  # noqa: E731
    metrics = evaluate_and_dump(
        "qrf", data, predict_fn, root=ROOT, quantiles=quantiles, smoke=args.smoke,
    )

    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, "qrf", "test", "clean"))
    med = d["quantiles"][..., int(np.argmin(np.abs(quantiles - 0.5)))].reshape(-1)
    y_flat = d["y_true"].reshape(-1)
    metrics["smape"] = smape(y_flat, med)
    metrics["mase"] = mase(y_flat, med, data.train_target_raw, season=24)
    metrics.update(
        model="qrf", target=TARGET, lookback=cfg.lookback, horizon=cfg.horizon,
        n_estimators=cfg.n_estimators, min_samples_leaf=cfg.min_samples_leaf,
        max_samples_leaf=cfg.max_samples_leaf, train_stride=cfg.train_stride,
        quantiles=quantiles.tolist(), seed=cfg.seed, smoke=bool(args.smoke),
    )
    out = ROOT / "results" / "base" / "qrf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
