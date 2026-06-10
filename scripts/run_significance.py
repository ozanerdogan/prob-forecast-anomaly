"""Significance tests for the headline model comparisons (stage 1.3).

Reads the frozen clean-test predictions (results/predictions/), forms the
point forecasts of every available model (probabilistic models contribute
their median; the LSTM+QT average forms the ensemble) and runs, for each
pair: a Diebold-Mariano test on per-window MSEs and a paired bootstrap for
the pooled-RMSE delta. Writes results/base/significance.json. No model is
run.

  python scripts/run_significance.py
  python scripts/run_significance.py --pred-dir results/predictions_smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.significance import dm_test, paired_bootstrap_rmse, window_mse  # noqa: E402


def _point_forecast(d: dict) -> np.ndarray:
    if "point" in d:
        return np.asarray(d["point"], dtype=float)
    levels = np.asarray(d["levels"])
    med = int(np.argmin(np.abs(levels - 0.5)))
    return np.asarray(d["quantiles"], dtype=float)[..., med]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    preds: dict[str, np.ndarray] = {}
    y_true = None
    for model in ("lstm", "naive_seasonal", "deepar", "qtransformer", "lgbm"):
        p = prediction_path(pred_dir, model, "test", "clean")
        if not p.exists():
            continue
        d = load_predictions(p)
        preds[model] = _point_forecast(d)
        y_true = np.asarray(d["y_true"], dtype=float)
    if y_true is None:
        raise SystemExit(f"no clean test dumps in {pred_dir} (run scripts/run_anomaly_eval.py)")
    if {"lstm", "qtransformer"} <= preds.keys():
        preds["ensemble_lstm_qt"] = 0.5 * (preds["lstm"] + preds["qtransformer"])

    rmse = {m: float(np.sqrt(np.mean((p - y_true) ** 2))) for m, p in preds.items()}
    losses = {m: window_mse(y_true, p) for m, p in preds.items()}

    out = {
        "n_windows": int(len(y_true)),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "note": "delta/dm convention: (a - b); negative favours a",
        "rmse": rmse,
        "pairs": {},
    }
    for a, b in itertools.combinations(sorted(preds), 2):
        key = f"{a}__vs__{b}"
        dm = dm_test(losses[a], losses[b])
        bs = paired_bootstrap_rmse(y_true, preds[a], preds[b],
                                   n_boot=args.n_boot, seed=args.seed)
        out["pairs"][key] = {"dm": dm, "bootstrap": bs}
        print(f"  {key:42s} dRMSE {bs['delta_rmse']:+.4f} "
              f"ci [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}] "
              f"DM p={dm['p_value']:.4f}")

    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "significance.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
