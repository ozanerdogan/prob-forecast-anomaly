"""Stage-2 repair, detect-and-clean contrast baseline (hampel input repair).

Scores the ``<setting>__cleaned`` frozen forecasts (produced by
run_anomaly_eval.py via hampel-filtered contexts) against their uncleaned
twins, for every model — point models get RMSE/MAE, probabilistic models get
interval metrics. This is the input-side alternative to calibration-side
repair: it should fix isolated spikes and largely pass level shifts through.
Writes results/calibrated/detect_clean/<model>.json.

  python scripts/calibrate_detect_clean.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrators import interval_metrics  # noqa: E402
from src.predictions_io import list_settings, load_predictions, prediction_path  # noqa: E402


def _point_metrics(y_true: np.ndarray, point: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(point, dtype=float).reshape(-1)
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


def _scores(d: dict, alpha: float) -> dict:
    if "quantiles" in d:
        return interval_metrics(d["y_true"], d["quantiles"], d["levels"], alpha)
    return _point_metrics(d["y_true"], d["point"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "calibrated" / "detect_clean"))
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted({p.stem.split("__", 1)[0] for p in pred_dir.glob("*__test__*.npz")})
    if not models:
        raise SystemExit(f"no prediction dumps in {pred_dir} (run scripts/run_anomaly_eval.py)")

    for model in models:
        settings = list_settings(pred_dir, model, "test")
        cleaned = [s for s in settings if s.endswith("__cleaned")]
        if not cleaned:
            continue
        out = {"model": model, "method": "detect_clean", "alpha": args.alpha,
               "filter": "hampel(window=12, n_sigmas=3.0)", "settings": {}}
        for s_cl in cleaned:
            base = s_cl[: -len("__cleaned")]
            if not prediction_path(pred_dir, model, "test", base).exists():
                continue
            d0 = load_predictions(prediction_path(pred_dir, model, "test", base))
            d1 = load_predictions(prediction_path(pred_dir, model, "test", s_cl))
            out["settings"][base] = {"before": _scores(d0, args.alpha),
                                     "after": _scores(d1, args.alpha)}
            b, a = out["settings"][base]["before"], out["settings"][base]["after"]
            key = "picp" if "picp" in b else "rmse"
            print(f"  {model:13s} {base:26s} {key} {b[key]:.3f} -> {a[key]:.3f}")
        out_path = out_dir / f"{model}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
